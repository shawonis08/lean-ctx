use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use rmcp::model::Tool;
use rmcp::ErrorData;
use serde_json::{json, Map, Value};

use crate::server::tool_trait::{
    get_bool, get_int, get_str, require_resolved_path, McpTool, ToolContext, ToolOutput,
};
use crate::tool_defs::tool_def;

/// Per-file lock that serializes concurrent reads of the same path.
///
/// When multiple subagents read sequentially through a shared set of files,
/// they tend to hit the same path at the same time. Without per-file locking
/// they all contend on the global cache write lock while doing redundant I/O.
/// This lock ensures only one thread reads a given file from disk; the others
/// wait cheaply on the per-file mutex, then hit the warm cache.
fn per_file_lock(path: &str) -> Arc<Mutex<()>> {
    static FILE_LOCKS: std::sync::OnceLock<Mutex<HashMap<String, Arc<Mutex<()>>>>> =
        std::sync::OnceLock::new();
    let map = FILE_LOCKS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut map = map.lock().unwrap();

    const MAX_ENTRIES: usize = 500;
    if map.len() > MAX_ENTRIES {
        map.retain(|_, v| Arc::strong_count(v) > 1);
    }

    map.entry(path.to_string())
        .or_insert_with(|| Arc::new(Mutex::new(())))
        .clone()
}

pub struct CtxReadTool;

impl McpTool for CtxReadTool {
    fn name(&self) -> &'static str {
        "ctx_read"
    }

    fn tool_def(&self) -> Tool {
        tool_def(
            "ctx_read",
            "Read file (cached, compressed). Cached re-reads can be ~13 tok when unchanged. Auto-selects optimal mode. \
Modes: full|map|signatures|diff|aggressive|entropy|task|reference|lines:N-M. fresh=true forces a disk re-read.",
            json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Absolute file path to read" },
                    "mode": {
                        "type": "string",
                        "description": "Compression mode (default: full). Use 'map' for context-only files. For line ranges: 'lines:N-M' (e.g. 'lines:400-500')."
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "Read from this line number to end of file. Implies fresh=true (disk re-read) to avoid stale snippets."
                    },
                    "fresh": {
                        "type": "boolean",
                        "description": "Bypass cache and force a full re-read. Use when running as a subagent that may not have the parent's context."
                    }
                },
                "required": ["path"]
            }),
        )
    }

    fn handle(
        &self,
        args: &Map<String, Value>,
        ctx: &ToolContext,
    ) -> Result<ToolOutput, ErrorData> {
        let path = require_resolved_path(ctx, args, "path")?;

        self.handle_inner(args, ctx, &path)
    }
}

impl CtxReadTool {
    #[allow(clippy::unused_self)]
    fn handle_inner(
        &self,
        args: &Map<String, Value>,
        ctx: &ToolContext,
        path: &str,
    ) -> Result<ToolOutput, ErrorData> {
        let session_lock = ctx
            .session
            .as_ref()
            .ok_or_else(|| ErrorData::internal_error("session not available", None))?;
        let cache_lock = ctx
            .cache
            .as_ref()
            .ok_or_else(|| ErrorData::internal_error("cache not available", None))?;

        let current_task = {
            let session = session_lock.blocking_read();
            session.task.as_ref().map(|t| t.description.clone())
        };
        let task_ref = current_task.as_deref();

        let profile = crate::core::profiles::active_profile();
        let mut mode = if let Some(m) = get_str(args, "mode") {
            m
        } else if profile.read.default_mode_effective() == "auto" {
            // Non-blocking: if the cache write-lock is held by a timed-out zombie
            // thread, blocking_read() would hang with NO timeout protection.
            if let Ok(cache) = cache_lock.try_read() {
                crate::tools::ctx_smart_read::select_mode_with_task(&cache, path, task_ref)
            } else {
                tracing::debug!(
                    "cache lock contested during auto-mode selection for {path}; \
                     falling back to full"
                );
                "full".to_string()
            }
        } else {
            profile.read.default_mode_effective().to_string()
        };

        let mut fresh = get_bool(args, "fresh").unwrap_or(false);
        let start_line = get_int(args, "start_line");
        if let Some(sl) = start_line {
            let sl = sl.max(1_i64);
            mode = format!("lines:{sl}-999999");
            fresh = true;
        }

        let pressure_action = ctx.pressure_snapshot.as_ref().map(|p| &p.recommendation);
        let resolved_agent_id = ctx
            .agent_id
            .as_ref()
            .and_then(|a| a.blocking_read().clone());
        let gate_result = crate::server::context_gate::pre_dispatch_read_for_agent(
            path,
            &mode,
            task_ref,
            Some(&ctx.project_root),
            pressure_action,
            resolved_agent_id.as_deref(),
        );
        if gate_result.budget_blocked {
            let msg = gate_result
                .budget_warning
                .unwrap_or_else(|| "Agent token budget exceeded".to_string());
            return Err(ErrorData::invalid_params(msg, None));
        }
        let budget_warning = gate_result.budget_warning.clone();
        if let Some(overridden) = gate_result.overridden_mode {
            mode = overridden;
        }

        let mode = if crate::tools::ctx_read::is_instruction_file(path) {
            "full".to_string()
        } else {
            auto_degrade_read_mode(&mode)
        };

        if mode.starts_with("lines:") {
            fresh = true;
        }

        if crate::core::binary_detect::is_binary_file(path) {
            let msg = crate::core::binary_detect::binary_file_message(path);
            return Err(ErrorData::invalid_params(msg, None));
        }
        {
            let cap = crate::core::limits::max_read_bytes() as u64;
            if let Ok(meta) = std::fs::metadata(path) {
                if meta.len() > cap {
                    let msg = format!(
                        "File too large ({} bytes, limit {} bytes via LCTX_MAX_READ_BYTES). \
                         Use mode=\"lines:1-100\" for partial reads or increase the limit.",
                        meta.len(),
                        cap
                    );
                    return Err(ErrorData::invalid_params(msg, None));
                }
            }
        }

        // Fast path: if both per-file lock and cache write-lock are immediately
        // available, execute inline without spawning a thread. This avoids thread +
        // channel overhead for the ~90% of calls that are cache hits.
        let read_timeout = std::time::Duration::from_secs(30);
        let cancelled = Arc::new(AtomicBool::new(false));
        let (output, resolved_mode, original, is_cache_hit, file_ref, cache_stats) = {
            let crp_mode = ctx.crp_mode;
            let task_ref = current_task.as_deref();

            let fast_result = 'fast: {
                let file_lock = per_file_lock(path);
                let Some(_file_guard) = file_lock.try_lock().ok() else {
                    break 'fast None;
                };
                let Some(mut cache) = cache_lock.try_write().ok() else {
                    break 'fast None;
                };
                let read_output = if fresh {
                    crate::tools::ctx_read::handle_fresh_with_task_resolved(
                        &mut cache, path, &mode, crp_mode, task_ref,
                    )
                } else {
                    crate::tools::ctx_read::handle_with_task_resolved(
                        &mut cache, path, &mode, crp_mode, task_ref,
                    )
                };
                let content = read_output.content;
                let rmode = read_output.resolved_mode;
                let orig = cache.get(path).map_or(0, |e| e.original_tokens);
                let hit = content.contains(" cached ")
                    || content.contains("[unchanged")
                    || content.contains("[delta:");
                let fref = cache.file_ref_map().get(path).cloned();
                let stats = cache.get_stats();
                let stats_snapshot = (stats.total_reads, stats.cache_hits);
                Some((content, rmode, orig, hit, fref, stats_snapshot))
            };

            if let Some(result) = fast_result {
                result
            } else {
                // Slow path: spawn thread with bounded timeout for contended locks.
                let cache_lock = cache_lock.clone();
                let mode = mode.clone();
                let task_owned = current_task.clone();
                let path_owned = path.to_string();
                let cancel_flag = cancelled.clone();
                let (tx, rx) = std::sync::mpsc::sync_channel(1);
                std::thread::spawn(move || {
                    let file_lock = per_file_lock(&path_owned);

                    // Bounded per-file lock: if a zombie thread still holds it, don't
                    // wait forever. 25s keeps us inside the 30s recv_timeout.
                    let _file_guard = {
                        let deadline =
                            std::time::Instant::now() + std::time::Duration::from_secs(25);
                        loop {
                            if cancel_flag.load(Ordering::Relaxed) {
                                return;
                            }
                            if let Ok(guard) = file_lock.try_lock() {
                                break guard;
                            }
                            if std::time::Instant::now() >= deadline {
                                tracing::error!(
                                    "ctx_read: per-file lock timeout after 25s for {path_owned}"
                                );
                                return;
                            }
                            std::thread::sleep(std::time::Duration::from_millis(50));
                        }
                    };

                    if cancel_flag.load(Ordering::Relaxed) {
                        return;
                    }

                    // Bounded cache write-lock: avoids indefinite block when a zombie
                    // thread from a previous timed-out call still holds the lock.
                    let mut cache = {
                        let deadline =
                            std::time::Instant::now() + std::time::Duration::from_secs(25);
                        loop {
                            if cancel_flag.load(Ordering::Relaxed) {
                                return;
                            }
                            if let Ok(guard) = cache_lock.try_write() {
                                break guard;
                            }
                            if std::time::Instant::now() >= deadline {
                                tracing::error!(
                                    "ctx_read: cache write-lock timeout after 25s for {path_owned}"
                                );
                                return;
                            }
                            std::thread::sleep(std::time::Duration::from_millis(50));
                        }
                    };

                    let task_ref = task_owned.as_deref();
                    let read_output = if fresh {
                        crate::tools::ctx_read::handle_fresh_with_task_resolved(
                            &mut cache,
                            &path_owned,
                            &mode,
                            crp_mode,
                            task_ref,
                        )
                    } else {
                        crate::tools::ctx_read::handle_with_task_resolved(
                            &mut cache,
                            &path_owned,
                            &mode,
                            crp_mode,
                            task_ref,
                        )
                    };
                    let content = read_output.content;
                    let rmode = read_output.resolved_mode;
                    let orig = cache.get(&path_owned).map_or(0, |e| e.original_tokens);
                    let hit = content.contains(" cached ");
                    let fref = cache.file_ref_map().get(path_owned.as_str()).cloned();
                    let stats = cache.get_stats();
                    let stats_snapshot = (stats.total_reads, stats.cache_hits);
                    let _ = tx.send((content, rmode, orig, hit, fref, stats_snapshot));
                });
                if let Ok(result) = rx.recv_timeout(read_timeout) {
                    result
                } else {
                    cancelled.store(true, Ordering::Relaxed);
                    tracing::error!("ctx_read timed out after {read_timeout:?} for {path}");
                    let msg = format!(
                        "ERROR: ctx_read timed out after {}s reading {path}. \
                     The file may be very large or a blocking I/O issue occurred. \
                     Try mode=\"lines:1-100\" for a partial read.",
                        read_timeout.as_secs()
                    );
                    return Err(ErrorData::internal_error(msg, None));
                }
            } // end else (slow path)
        };

        // Convert error results to proper MCP ErrorData instead of success body
        if resolved_mode == "error" {
            return Err(ErrorData::invalid_params(output, None));
        }

        let output_tokens = crate::core::tokens::count_tokens(&output);
        let saved = original.saturating_sub(output_tokens);

        // Session updates (short lock)
        let mut ensured_root: Option<String> = None;
        let project_root_snapshot;
        {
            let mut session = session_lock.blocking_write();
            session.touch_file(path, file_ref.as_deref(), &resolved_mode, original);
            if is_cache_hit {
                session.record_cache_hit();
            }
            if session.active_structured_intent.is_none() && session.files_touched.len() >= 2 {
                let touched: Vec<String> = session
                    .files_touched
                    .iter()
                    .map(|f| f.path.clone())
                    .collect();
                let inferred =
                    crate::core::intent_engine::StructuredIntent::from_file_patterns(&touched);
                if inferred.confidence >= 0.4 {
                    session.active_structured_intent = Some(inferred);
                }
            }
            let root_missing = session
                .project_root
                .as_deref()
                .is_none_or(|r| r.trim().is_empty());
            if root_missing {
                if let Some(root) = crate::core::protocol::detect_project_root(path) {
                    session.project_root = Some(root.clone());
                    ensured_root = Some(root);
                }
            }
            project_root_snapshot = session
                .project_root
                .clone()
                .unwrap_or_else(|| ".".to_string());
        }

        if let Some(root) = ensured_root.as_deref() {
            crate::core::index_orchestrator::ensure_all_background(root);
        }

        crate::core::heatmap::record_file_access(path, original, saved);

        // Mode predictor + feedback — no locks needed, uses snapshots from above
        {
            let sig = crate::core::mode_predictor::FileSignature::from_path(path, original);
            let density = if output_tokens > 0 {
                original as f64 / output_tokens as f64
            } else {
                1.0
            };
            let outcome = crate::core::mode_predictor::ModeOutcome {
                mode: resolved_mode.clone(),
                tokens_in: original,
                tokens_out: output_tokens,
                density: density.min(1.0),
            };
            let mut predictor = crate::core::mode_predictor::ModePredictor::new();
            predictor.set_project_root(&project_root_snapshot);
            predictor.record(sig, outcome);
            predictor.save();

            let ext = std::path::Path::new(path)
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("")
                .to_string();
            let thresholds = crate::core::adaptive_thresholds::thresholds_for_path(path);
            let feedback_outcome = crate::core::feedback::CompressionOutcome {
                session_id: format!("{}", std::process::id()),
                language: ext,
                entropy_threshold: thresholds.bpe_entropy,
                jaccard_threshold: thresholds.jaccard,
                total_turns: cache_stats.0 as u32,
                tokens_saved: saved as u64,
                tokens_original: original as u64,
                cache_hits: cache_stats.1 as u32,
                total_reads: cache_stats.0 as u32,
                task_completed: true,
                timestamp: chrono::Local::now().to_rfc3339(),
            };
            let mut store = crate::core::feedback::FeedbackStore::load();
            store.project_root = Some(project_root_snapshot.clone());
            store.record_outcome(feedback_outcome);
        }

        if let Some(aid) = resolved_agent_id.as_deref() {
            crate::core::agent_budget::record_consumption(aid, output_tokens);
        }

        let final_output = if let Some(ref warning) = budget_warning {
            format!("{output}\n\n{warning}")
        } else {
            output
        };

        Ok(ToolOutput {
            text: final_output,
            original_tokens: original,
            saved_tokens: saved,
            mode: Some(resolved_mode),
            path: Some(path.to_string()),
            changed: false,
        })
    }
}

fn auto_degrade_read_mode(mode: &str) -> String {
    use crate::core::degradation_policy::DegradationVerdictV1;
    let profile = crate::core::profiles::active_profile();
    if !profile.degradation.enforce_effective() {
        return mode.to_string();
    }
    let policy = crate::core::degradation_policy::evaluate_v1_for_tool("ctx_read", None);
    match policy.decision.verdict {
        DegradationVerdictV1::Ok => mode.to_string(),
        DegradationVerdictV1::Warn => match mode {
            "full" => "map".to_string(),
            other => other.to_string(),
        },
        DegradationVerdictV1::Throttle => match mode {
            "full" | "map" => "signatures".to_string(),
            other => other.to_string(),
        },
        DegradationVerdictV1::Block => "signatures".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn per_file_lock_same_path_returns_same_mutex() {
        let lock_a1 = per_file_lock("/tmp/test_same_path.txt");
        let lock_a2 = per_file_lock("/tmp/test_same_path.txt");
        assert!(Arc::ptr_eq(&lock_a1, &lock_a2));
    }

    #[test]
    fn per_file_lock_different_paths_return_different_mutexes() {
        let lock_a = per_file_lock("/tmp/test_path_a.txt");
        let lock_b = per_file_lock("/tmp/test_path_b.txt");
        assert!(!Arc::ptr_eq(&lock_a, &lock_b));
    }

    #[test]
    fn per_file_lock_serializes_concurrent_access() {
        let counter = Arc::new(AtomicUsize::new(0));
        let max_concurrent = Arc::new(AtomicUsize::new(0));
        let path = "/tmp/test_concurrent_serialization.txt";
        let mut handles = Vec::new();

        for _ in 0..5 {
            let counter = counter.clone();
            let max_concurrent = max_concurrent.clone();
            let path = path.to_string();
            handles.push(std::thread::spawn(move || {
                let lock = per_file_lock(&path);
                let _guard = lock.lock().unwrap();
                let active = counter.fetch_add(1, Ordering::SeqCst) + 1;
                max_concurrent.fetch_max(active, Ordering::SeqCst);
                std::thread::sleep(std::time::Duration::from_millis(10));
                counter.fetch_sub(1, Ordering::SeqCst);
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        assert_eq!(max_concurrent.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn per_file_lock_allows_parallel_different_paths() {
        let counter = Arc::new(AtomicUsize::new(0));
        let max_concurrent = Arc::new(AtomicUsize::new(0));
        let mut handles = Vec::new();

        for i in 0..4 {
            let counter = counter.clone();
            let max_concurrent = max_concurrent.clone();
            let path = format!("/tmp/test_parallel_{i}.txt");
            handles.push(std::thread::spawn(move || {
                let lock = per_file_lock(&path);
                let _guard = lock.lock().unwrap();
                let active = counter.fetch_add(1, Ordering::SeqCst) + 1;
                max_concurrent.fetch_max(active, Ordering::SeqCst);
                std::thread::sleep(std::time::Duration::from_millis(50));
                counter.fetch_sub(1, Ordering::SeqCst);
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        assert!(max_concurrent.load(Ordering::SeqCst) > 1);
    }

    /// Regression test for Issue #229: a zombie thread holding the cache write-lock
    /// must not block subsequent reads indefinitely. The try_write() loop inside
    /// the spawned thread should respect its 25s deadline and the cancellation flag.
    #[test]
    fn zombie_thread_does_not_block_subsequent_cache_access() {
        let cache: Arc<tokio::sync::RwLock<u32>> = Arc::new(tokio::sync::RwLock::new(0));

        // Simulate a zombie: hold the write-lock on a background thread for 2s.
        let zombie_lock = cache.clone();
        let _zombie = std::thread::spawn(move || {
            let _guard = zombie_lock.blocking_write();
            std::thread::sleep(std::time::Duration::from_secs(2));
        });
        std::thread::sleep(std::time::Duration::from_millis(50));

        // A try_read() must fail immediately (zombie holds write-lock).
        assert!(cache.try_read().is_err());

        // A try_write() loop with cancellation must exit promptly.
        let cancel = Arc::new(AtomicBool::new(false));
        let cancel2 = cancel.clone();
        let lock2 = cache.clone();
        let waiter = std::thread::spawn(move || {
            let start = std::time::Instant::now();
            loop {
                if cancel2.load(Ordering::Relaxed) {
                    return (false, start.elapsed());
                }
                if let Ok(_guard) = lock2.try_write() {
                    return (true, start.elapsed());
                }
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
        });

        // Set cancellation after 200ms — the loop should exit quickly.
        std::thread::sleep(std::time::Duration::from_millis(200));
        cancel.store(true, Ordering::Relaxed);

        let (acquired, elapsed) = waiter.join().unwrap();
        assert!(
            !acquired,
            "should not have acquired lock while zombie holds it"
        );
        assert!(
            elapsed < std::time::Duration::from_secs(1),
            "cancellation should have stopped the loop promptly"
        );
    }
}
