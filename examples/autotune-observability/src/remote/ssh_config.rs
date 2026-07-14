use std::path::PathBuf;

/// The subset of `~/.ssh/config` we resolve for a host alias.
#[derive(Default)]
pub(crate) struct HostConfig {
    pub hostname: Option<String>,
    pub user: Option<String>,
    pub port: Option<u16>,
    pub identity_file: Option<PathBuf>,
}

/// Look up `alias` in `~/.ssh/config`, honouring OpenSSH's "first matching value wins" rule across
/// `Host` blocks (including `Host *` defaults). Returns an empty config if the file is absent, so a
/// plain `user@host` still works without any config.
pub(crate) fn lookup(alias: &str) -> HostConfig {
    let mut cfg = HostConfig::default();
    let Some(home) = std::env::var_os("HOME") else {
        return cfg;
    };
    let Ok(text) = std::fs::read_to_string(PathBuf::from(home).join(".ssh").join("config")) else {
        return cfg;
    };

    let mut matching = false;
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((key, value)) = split_kv(line) else {
            continue;
        };
        if key.eq_ignore_ascii_case("Host") {
            matching = value.split_whitespace().any(|pat| matches_pattern(pat, alias));
        } else if matching {
            fill(&mut cfg, &key.to_ascii_lowercase(), value);
        }
    }
    cfg
}

fn fill(cfg: &mut HostConfig, key: &str, value: &str) {
    match key {
        "hostname" if cfg.hostname.is_none() => cfg.hostname = Some(value.to_string()),
        "user" if cfg.user.is_none() => cfg.user = Some(value.to_string()),
        "port" if cfg.port.is_none() => cfg.port = value.parse().ok(),
        "identityfile" if cfg.identity_file.is_none() => cfg.identity_file = Some(expand(value)),
        _ => {}
    }
}

/// Split a config line into `(key, value)`, accepting either `Key value` or `Key = value`.
fn split_kv(line: &str) -> Option<(&str, &str)> {
    let (key, rest) = line.split_once(|c: char| c.is_whitespace() || c == '=')?;
    let value = rest.trim_start_matches(['=', ' ', '\t']).trim();
    if value.is_empty() {
        None
    } else {
        Some((key.trim(), value))
    }
}

/// Match an OpenSSH host pattern against an alias, supporting `*` and a single leading/trailing
/// wildcard — enough for the common `Host *` and `Host prefix*` cases.
fn matches_pattern(pattern: &str, alias: &str) -> bool {
    if pattern == "*" || pattern == alias {
        return true;
    }
    if let Some(suffix) = pattern.strip_prefix('*') {
        return alias.ends_with(suffix);
    }
    if let Some(prefix) = pattern.strip_suffix('*') {
        return alias.starts_with(prefix);
    }
    false
}

fn expand(path: &str) -> PathBuf {
    if let Some(rest) = path.strip_prefix("~/")
        && let Some(home) = std::env::var_os("HOME")
    {
        return PathBuf::from(home).join(rest);
    }
    PathBuf::from(path)
}

#[cfg(test)]
mod tests {
    use super::matches_pattern;

    #[test]
    fn patterns() {
        assert!(matches_pattern("tower", "tower"));
        assert!(matches_pattern("*", "tower"));
        assert!(matches_pattern("gpu*", "gpu-box"));
        assert!(matches_pattern("*.lan", "tower.lan"));
        assert!(!matches_pattern("tower", "other"));
        assert!(!matches_pattern("gpu*", "cpu-box"));
    }
}
