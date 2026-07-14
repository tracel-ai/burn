use std::path::PathBuf;

use crate::example_dir;

/// Where the remote settings are persisted. The password is intentionally never written here.
fn config_path() -> PathBuf {
    example_dir().join("remote-config.txt")
}

/// Connection + layout settings for running the workload on a remote host over SSH.
#[derive(Clone, Default)]
pub(crate) struct RemoteConfig {
    /// Run on the remote instead of locally.
    pub enabled: bool,
    /// `user@host` or `user@host:port`.
    pub host: String,
    /// Blank means key/agent auth; otherwise password auth. Kept in memory only, never persisted.
    pub password: String,
    /// Base directory on the remote holding the synced repos. Blank means "the remote's own temp
    /// dir" (`/tmp` on unix, `%TEMP%` on Windows), resolved per-OS after connecting.
    pub base_dir: String,
}

impl RemoteConfig {
    pub fn load() -> Self {
        let mut cfg = Self::default();
        let Ok(text) = std::fs::read_to_string(config_path()) else {
            return cfg;
        };
        for line in text.lines() {
            let Some((key, value)) = line.split_once('=') else {
                continue;
            };
            let value = value.trim().to_string();
            match key.trim() {
                "enabled" => cfg.enabled = value == "true",
                "host" => cfg.host = value,
                "base_dir" if !value.is_empty() => cfg.base_dir = value,
                _ => {}
            }
        }
        cfg
    }

    pub fn save(&self) {
        let text = format!(
            "enabled = {}\nhost = {}\nbase_dir = {}\n",
            self.enabled, self.host, self.base_dir
        );
        let _ = std::fs::write(config_path(), text);
    }
}
