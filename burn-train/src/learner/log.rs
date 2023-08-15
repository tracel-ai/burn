use log::LevelFilter;
use log4rs::{
    append::file::FileAppender,
    config::{Appender, Config, Root},
    encode::pattern::PatternEncoder,
    Handle,
};
use std::{cell::Cell, sync::Mutex};

static LOGGER_HANDLER: Mutex<Cell<Option<Handle>>> = Mutex::new(Cell::new(None));

pub fn update_log_file(file_path: &str) {
    let config = create_config(file_path);
    set_config(config, file_path);
}

fn create_config(file_path: &str) -> Config {
    let experiment = FileAppender::builder()
        .encoder(Box::new(PatternEncoder::new(
            "[{d(%+)(utc)} - {h({l})} - {f}:{L}] {m}{n}",
        )))
        .build(file_path)
        .unwrap();

    /// The wgpu crate is logging too much, so we skip `info` level.
    #[derive(Debug)]
    struct WgpuFilter;

    impl log4rs::filter::Filter for WgpuFilter {
        fn filter(&self, record: &log::Record) -> log4rs::filter::Response {
            if !matches!(record.level(), log::Level::Info) {
                return log4rs::filter::Response::Accept;
            }

            match record.module_path_static() {
                Some(path) => {
                    if path.starts_with("wgpu") {
                        log4rs::filter::Response::Reject
                    } else {
                        log4rs::filter::Response::Accept
                    }
                }
                None => log4rs::filter::Response::Accept,
            }
        }
    }

    Config::builder()
        .appender(
            Appender::builder()
                .filter(Box::new(WgpuFilter))
                .build("experiment", Box::new(experiment)),
        )
        .build(
            Root::builder()
                .appender("experiment")
                .build(LevelFilter::Info),
        )
        .unwrap()
}

fn set_config(config: Config, file_path: &str) {
    let mut cell = LOGGER_HANDLER.lock().unwrap();

    match cell.get_mut() {
        Some(handler) => {
            handler.set_config(config);
        }
        None => {
            let handler = log4rs::init_config(config).unwrap();
            update_panic_hook(file_path);
            cell.replace(Some(handler));
        }
    }
}

fn update_panic_hook(file_path: &str) {
    let hook = std::panic::take_hook();
    let file_path = file_path.to_owned();

    std::panic::set_hook(Box::new(move |info| {
        log::error!("PANIC => {}", info.to_string());
        eprintln!(
            "=== PANIC ===\nA fatal error happened, you can check the experiment logs here => '{file_path}'\n============="
        );
        hook(info);
    }));
}
