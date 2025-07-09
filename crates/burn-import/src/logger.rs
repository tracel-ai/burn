#![allow(dead_code)]

use std::error::Error;
use tracing_core::LevelFilter;

pub fn init_log() -> Result<(), Box<dyn Error + Send + Sync>> {
    let result = tracing_subscriber::fmt()
        .with_max_level(LevelFilter::DEBUG)
        .without_time()
        .try_init();

    if result.is_ok() {
        update_panic_hook();
    }
    result
}

fn update_panic_hook() {
    let hook = std::panic::take_hook();

    std::panic::set_hook(Box::new(move |info| {
        log::error!("PANIC => {info}");
        hook(info);
    }));
}
