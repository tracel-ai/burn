pub(crate) fn now_millis() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0)
}

/// Whether the UI itself is a release build, so the runner is compiled with the same profile
/// (and cargo reuses the UI's own build artifacts instead of a second full compile).
pub(crate) fn is_release() -> bool {
    std::env::current_exe()
        .ok()
        .map(|p| p.components().any(|c| c.as_os_str() == "release"))
        .unwrap_or(false)
}
