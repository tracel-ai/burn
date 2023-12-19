use std::path::Path;
use tracing_core::{Level, LevelFilter};
use tracing_subscriber::filter::filter_fn;
use tracing_subscriber::prelude::*;
use tracing_subscriber::{registry, Layer};

/// If a global tracing subscriber is not already configured, set up logging to a file,
/// and add our custom panic hook.
pub(crate) fn install_file_logger(file_path: &str) {
    let path = Path::new(file_path);
    let writer = tracing_appender::rolling::never(
        path.parent().unwrap_or_else(|| Path::new(".")),
        path.file_name()
            .unwrap_or_else(|| panic!("The path '{file_path}' to point to a file.")),
    );
    let layer = tracing_subscriber::fmt::layer()
        .with_ansi(false)
        .with_writer(writer)
        .with_filter(LevelFilter::INFO)
        .with_filter(filter_fn(|m| {
            if let Some(path) = m.module_path() {
                // The wgpu crate is logging too much, so we skip `info` level.
                if path.starts_with("wgpu") && *m.level() >= Level::INFO {
                    return false;
                }
            }
            true
        }));

    if registry().with(layer).try_init().is_ok() {
        update_panic_hook(file_path);
    }
}

fn update_panic_hook(file_path: &str) {
    let hook = std::panic::take_hook();
    let file_path = file_path.to_owned();

    std::panic::set_hook(Box::new(move |info| {
        log::error!("PANIC => {}", info.to_string());
        eprintln!(
            "=== PANIC ===\nA fatal error happened, you can check the experiment logs here => \
             '{file_path}'\n============="
        );
        hook(info);
    }));
}
