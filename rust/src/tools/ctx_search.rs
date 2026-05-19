use std::collections::HashSet;
use std::path::Path;
use std::path::PathBuf;

use ignore::WalkBuilder;
use regex::RegexBuilder;

use crate::core::protocol;
use crate::core::symbol_map::{self, SymbolMap};
use crate::core::tokens::count_tokens;
use crate::tools::CrpMode;

const MAX_FILE_SIZE: u64 = 512_000;
const MAX_WALK_DEPTH: usize = 20;

/// Searches files for a regex pattern with compressed output and monorepo scope hints.
pub fn handle(
    pattern: &str,
    dir: &str,
    ext_filter: Option<&str>,
    max_results: usize,
    _crp_mode: CrpMode,
    respect_gitignore: bool,
    allow_secret_paths: bool,
) -> (String, usize) {
    const MAX_PATTERN_LEN: usize = 1024;
    const MAX_REGEX_SIZE: usize = 1 << 20; // 1 MiB DFA limit

    let redact = crate::core::redaction::redaction_enabled_for_active_role();
    if pattern.len() > MAX_PATTERN_LEN {
        return (
            format!(
                "ERROR: pattern too long ({} > {MAX_PATTERN_LEN} chars)",
                pattern.len()
            ),
            0,
        );
    }
    let re = match RegexBuilder::new(pattern)
        .size_limit(MAX_REGEX_SIZE)
        .dfa_size_limit(MAX_REGEX_SIZE)
        .build()
    {
        Ok(r) => r,
        Err(e) => return (format!("ERROR: invalid regex: {e}"), 0),
    };

    let root = Path::new(dir);
    if !root.exists() {
        return (format!("ERROR: {dir} does not exist"), 0);
    }

    let walker = WalkBuilder::new(root)
        .hidden(true)
        .max_depth(Some(MAX_WALK_DEPTH))
        .git_ignore(respect_gitignore)
        .git_global(respect_gitignore)
        .git_exclude(respect_gitignore)
        .build();

    let mut files: Vec<PathBuf> = Vec::new();
    let mut matches = Vec::new();
    let mut raw_tokens_accum: usize = 0;
    let mut files_searched = 0u32;
    let mut files_skipped_size = 0u32;
    let mut files_skipped_encoding = 0u32;
    let mut files_skipped_boundary = 0u32;

    for entry in walker.filter_map(std::result::Result::ok) {
        if entry.file_type().is_none_or(|ft| ft.is_dir()) {
            continue;
        }

        if entry.file_type().is_some_and(|ft| ft.is_symlink()) {
            continue;
        }

        let path = entry.path();

        if is_binary_ext(path) || is_generated_file(path) {
            continue;
        }

        if !allow_secret_paths && crate::core::io_boundary::is_secret_like(path).is_some() {
            files_skipped_boundary += 1;
            continue;
        }

        if let Some(ext) = ext_filter {
            let file_ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
            if file_ext != ext {
                continue;
            }
        }

        if let Ok(meta) = std::fs::metadata(path) {
            if meta.len() > MAX_FILE_SIZE {
                files_skipped_size += 1;
                continue;
            }
        }

        files.push(path.to_path_buf());
    }

    // Deterministic search: stable file ordering makes max_results truncation reproducible.
    files.sort_unstable_by(|a, b| a.as_os_str().cmp(b.as_os_str()));

    for path in &files {
        if matches.len() >= max_results {
            break;
        }

        let Ok(content) = std::fs::read_to_string(path) else {
            files_skipped_encoding += 1;
            continue;
        };

        files_searched += 1;

        for (i, line) in content.lines().enumerate() {
            if re.is_match(line) {
                let short_path = protocol::shorten_path(&path.to_string_lossy());
                // Count raw tokens incrementally (avoids separate Vec + join)
                raw_tokens_accum += count_tokens(line.trim()) + 2;
                let shown = if redact {
                    crate::core::redaction::redact_text(line.trim())
                } else {
                    line.trim().to_string()
                };
                matches.push(format!("{short_path}:{} {}", i + 1, shown));
                if matches.len() >= max_results {
                    break;
                }
            }
        }
    }

    if matches.is_empty() {
        let mut msg = format!("0 matches for '{pattern}' in {files_searched} files");
        if files_skipped_size > 0 {
            msg.push_str(&format!(" ({files_skipped_size} large files skipped)"));
        }
        if files_skipped_encoding > 0 {
            msg.push_str(&format!(
                " ({files_skipped_encoding} files skipped: binary/encoding)"
            ));
        }
        if files_skipped_boundary > 0 {
            msg.push_str(&format!(
                " ({files_skipped_boundary} secret-like files skipped by boundary policy)"
            ));
        }
        return (msg, 0);
    }

    // Prefix-cache-friendly: structural file list before per-query match content
    let matched_files: Vec<&str> = {
        let mut seen = HashSet::new();
        matches
            .iter()
            .filter_map(|m| {
                let file = m.split(':').next()?;
                if seen.insert(file) {
                    Some(file)
                } else {
                    None
                }
            })
            .collect()
    };

    let mut result = format!("{} matches in {} files", matches.len(), files_searched);
    if matched_files.len() > 1 {
        result.push_str(" [");
        result.push_str(&matched_files.join(", "));
        result.push(']');
    }
    result.push_str(":\n");
    result.push_str(&matches.join("\n"));

    if files_skipped_size > 0 {
        result.push_str(&format!("\n({files_skipped_size} files >512KB skipped)"));
    }
    if files_skipped_encoding > 0 {
        result.push_str(&format!(
            "\n({files_skipped_encoding} files skipped: binary/encoding)"
        ));
    }
    if files_skipped_boundary > 0 {
        result.push_str(&format!(
            "\n({files_skipped_boundary} secret-like files skipped by boundary policy)"
        ));
    }

    let scope_hint = monorepo_scope_hint(&matches, dir);

    {
        let file_ext = ext_filter.unwrap_or("rs");
        let mut sym = SymbolMap::new();
        let idents = symbol_map::extract_identifiers(&result, file_ext);
        for ident in &idents {
            sym.register(ident);
        }
        if sym.len() >= 3 {
            let sym_table = sym.format_table();
            let compressed = sym.apply(&result);
            let original_tok = count_tokens(&result);
            let compressed_tok = count_tokens(&compressed) + count_tokens(&sym_table);
            let net_saving = original_tok.saturating_sub(compressed_tok);
            if original_tok > 0 && net_saving * 100 / original_tok >= 5 {
                result = format!("{compressed}{sym_table}");
            }
        }
    }

    if let Some(hint) = scope_hint {
        result.push_str(&hint);
    }

    let sent = count_tokens(&result);

    // The "original" cost is what a native grep with context lines would produce.
    // rg defaults to showing full paths + 2 context lines per match. We estimate
    // the native cost as ~3x the raw match output (context + separators + headers).
    let native_estimate = (raw_tokens_accum as f64 * 2.5).ceil() as usize;
    let original = native_estimate.max(raw_tokens_accum);
    let savings = protocol::format_savings(original, sent);

    (format!("{result}\n{savings}"), original)
}

fn is_binary_ext(path: &Path) -> bool {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    matches!(
        ext,
        "png"
            | "jpg"
            | "jpeg"
            | "gif"
            | "webp"
            | "ico"
            | "svg"
            | "woff"
            | "woff2"
            | "ttf"
            | "eot"
            | "pdf"
            | "zip"
            | "tar"
            | "gz"
            | "br"
            | "zst"
            | "bz2"
            | "xz"
            | "mp3"
            | "mp4"
            | "webm"
            | "ogg"
            | "wasm"
            | "so"
            | "dylib"
            | "dll"
            | "exe"
            | "lock"
            | "map"
            | "snap"
            | "patch"
            | "db"
            | "sqlite"
            | "parquet"
            | "arrow"
            | "bin"
            | "o"
            | "a"
            | "class"
            | "pyc"
            | "pyo"
    )
}

fn is_generated_file(path: &Path) -> bool {
    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
    name.ends_with(".min.js")
        || name.ends_with(".min.css")
        || name.ends_with(".bundle.js")
        || name.ends_with(".chunk.js")
        || name.ends_with(".d.ts")
        || name.ends_with(".js.map")
        || name.ends_with(".css.map")
}

fn monorepo_scope_hint(matches: &[String], search_dir: &str) -> Option<String> {
    let top_dirs: HashSet<&str> = matches
        .iter()
        .filter_map(|m| {
            let path = m.split(':').next()?;
            let relative = path.strip_prefix("./").unwrap_or(path);
            let relative = relative.strip_prefix(search_dir).unwrap_or(relative);
            let relative = relative.strip_prefix('/').unwrap_or(relative);
            relative.split('/').next()
        })
        .collect();

    if top_dirs.len() > 3 {
        let mut dirs: Vec<&&str> = top_dirs.iter().collect();
        dirs.sort();
        let dir_list: Vec<String> = dirs.iter().take(6).map(|d| format!("'{d}'")).collect();
        let extra = if top_dirs.len() > 6 {
            format!(", +{} more", top_dirs.len() - 6)
        } else {
            String::new()
        };
        Some(format!(
            "\n\nResults span {} directories ({}{}). \
             Use the 'path' parameter to scope to a specific service, \
             e.g. path=\"{}/\".",
            top_dirs.len(),
            dir_list.join(", "),
            extra,
            dirs[0]
        ))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::CrpMode;

    #[test]
    fn search_results_are_deterministically_ordered_by_path() {
        let dir = tempfile::tempdir().unwrap();
        let a = dir.path().join("a.txt");
        let b = dir.path().join("b.txt");
        std::fs::write(&b, "match\n").unwrap();
        std::fs::write(&a, "match\n").unwrap();

        let (out, _orig) = handle(
            "match",
            dir.path().to_string_lossy().as_ref(),
            Some("txt"),
            10,
            CrpMode::Off,
            true,
            true,
        );

        let mut match_lines: Vec<&str> = out
            .lines()
            .filter(|l| l.contains(".txt:") && l.contains("match"))
            .collect();
        // Expect exactly the 2 match lines, ordered a.txt then b.txt.
        match_lines.truncate(2);
        assert_eq!(match_lines.len(), 2);
        assert!(
            match_lines[0].contains("a.txt:"),
            "first match should come from a.txt, got: {}",
            match_lines[0]
        );
        assert!(
            match_lines[1].contains("b.txt:"),
            "second match should come from b.txt, got: {}",
            match_lines[1]
        );
    }

    #[test]
    fn secret_like_files_are_skipped_by_default() {
        let dir = tempfile::tempdir().unwrap();
        let secret = dir.path().join("key.pem");
        let ok = dir.path().join("ok.txt");
        std::fs::write(&secret, "match\n").unwrap();
        std::fs::write(&ok, "match\n").unwrap();

        let (out, _orig) = handle(
            "match",
            dir.path().to_string_lossy().as_ref(),
            None,
            10,
            CrpMode::Off,
            true,
            false,
        );

        assert!(out.contains("ok.txt:"), "expected ok.txt match, got: {out}");
        assert!(
            !out.contains("key.pem:"),
            "secret-like file should be skipped, got: {out}"
        );
        assert!(
            out.contains("secret-like files skipped"),
            "expected boundary skip note, got: {out}"
        );
    }
}
