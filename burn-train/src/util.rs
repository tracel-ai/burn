#![allow(missing_docs)]

#[cfg(feature = "browser")]
use crate::pool::WORKER_POOL;
#[cfg(feature = "browser")]
use wasm_bindgen::prelude::*;

#[cfg(not(feature = "browser"))]
pub fn spawn<F>(f: F) -> Box<dyn FnOnce() -> Result<(), ()>>
where
    F: FnOnce(),
    F: Send + 'static,
{
    let handle = std::thread::spawn(f);
    Box::new(move || handle.join().map_err(|_| ()))
}

// High level description at https://www.tweag.io/blog/2022-11-24-wasm-threads-and-messages/
// Mostly copied from https://github.com/tweag/rust-wasm-threads/blob/main/shared-memory/src/lib.rs
#[cfg(feature = "browser")]
pub fn spawn<F>(f: F) -> Box<dyn FnOnce() -> Result<(), ()>>
where
    F: FnOnce(),
    F: Send + 'static,
{
    WORKER_POOL.run(f).unwrap();
    Box::new(|| Ok(()))
}

#[cfg(feature = "browser")]
pub static WORKER_URL: std::sync::OnceLock<String> = std::sync::OnceLock::new();

#[cfg(feature = "browser")]
pub fn init(worker_url: String, worker_count: usize) -> Result<(), JsValue> {
    WORKER_URL
        .set(worker_url)
        .map_err(|worker_url| -> JsValue {
            format!(
                "You can only call `init` once. You tried to `init` with: {:?}",
                worker_url
            )
            .into()
        })?;
    crate::pool::init(worker_count)
}
