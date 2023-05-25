use log::{LevelFilter, SetLoggerError};
use log4rs::{
    append::console::ConsoleAppender,
    config::{Appender, Root},
    encode::pattern::PatternEncoder,
    Config,
};

pub fn init_log() -> Result<(), SetLoggerError> {
    let stdout = ConsoleAppender::builder()
        .encoder(Box::new(PatternEncoder::new("[{h({l})} - {f}:{L}] {m}{n}")))
        .build();
    let appender = Appender::builder().build("stdout", Box::new(stdout));

    log4rs::init_config(
        Config::builder()
            .appender(appender)
            .build(Root::builder().appender("stdout").build(LevelFilter::Debug))
            .unwrap(),
    )?;
    update_panic_hook();

    Ok(())
}

fn update_panic_hook() {
    let hook = std::panic::take_hook();

    std::panic::set_hook(Box::new(move |info| {
        log::error!("PANIC => {}", info.to_string());
        hook(info);
    }));
}
