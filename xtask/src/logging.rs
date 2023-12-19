use std::io::Write;

/// Initialise and create a `env_logger::Builder` which follows the
/// GitHub Actions logging syntax when running on CI.
pub fn init_logger() -> env_logger::Builder {
    let mut builder = env_logger::Builder::from_default_env();
    builder.target(env_logger::Target::Stdout);

    // Find and setup the correct log level
    builder.filter(None, get_log_level());
    builder.write_style(env_logger::WriteStyle::Always);

    // Custom Formatter for Github Actions
    if std::env::var("CI").is_ok() {
        builder.format(|buf, record| match record.level().as_str() {
            "DEBUG" => writeln!(buf, "::debug:: {}", record.args()),
            "WARN" => writeln!(buf, "::warning:: {}", record.args()),
            "ERROR" => {
                writeln!(buf, "::error:: {}", record.args())
            }
            _ => writeln!(buf, "{}", record.args()),
        });
    }

    builder
}

/// Determine the LogLevel for the logger
fn get_log_level() -> log::LevelFilter {
    // DEBUG
    match std::env::var("DEBUG") {
        Ok(_value) => return log::LevelFilter::Debug,
        Err(_err) => (),
    }
    // ACTIONS_RUNNER_DEBUG
    match std::env::var("ACTIONS_RUNNER_DEBUG") {
        Ok(_value) => return log::LevelFilter::Debug,
        Err(_err) => (),
    };

    log::LevelFilter::Info
}

/// Group Macro
#[macro_export]
macro_rules! group {
    // group!()
    ($($arg:tt)*) => {
        let title = format!($($arg)*);
        if std::env::var("CI").is_ok() {
            log!(log::Level::Info, "::group::{}", title)
        } else {
            log!(log::Level::Info, "{}", title)
        }
    };
}

/// End Group Macro
#[macro_export]
macro_rules! endgroup {
    // endgroup!()
    () => {
        if std::env::var("CI").is_ok() {
            log!(log::Level::Info, "::endgroup::")
        }
    };
}
