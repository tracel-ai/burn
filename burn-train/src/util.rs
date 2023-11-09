#![allow(missing_docs)]

#[cfg(feature = "browser")]
use wasm_bindgen::prelude::*;

#[cfg(not(feature = "browser"))]
pub fn spawn<F>(f: F) -> Box<dyn FnOnce() -> Result<(), ()>>
where
    F: FnOnce() -> (),
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
    F: FnOnce() -> (),
    F: Send + 'static,
{
    let mut worker_options = web_sys::WorkerOptions::new();
    worker_options.type_(web_sys::WorkerType::Module);
    // Double-boxing because `dyn FnOnce` is unsized and so `Box<dyn FnOnce()>` has
    let w = web_sys::Worker::new_with_options(
        WORKER_URL
            .get()
            .expect("You must first call `init` with the worker's url."),
        &worker_options,
    )
    .expect(&format!("Error initializing worker at {:?}", WORKER_URL));
    // an undefined layout (although I think in practice its a pointer and a length?).
    let ptr = Box::into_raw(Box::new(Box::new(f) as Box<dyn FnOnce()>));

    // See `worker.js` for the format of this message.
    let msg: js_sys::Array = [
        &wasm_bindgen::module(),
        &wasm_bindgen::memory(),
        &JsValue::from(ptr as u32),
    ]
    .into_iter()
    .collect();
    if let Err(e) = w.post_message(&msg) {
        // We expect the worker to deallocate the box, but if there was an error then
        // we'll do it ourselves.
        let _ = unsafe { Box::from_raw(ptr) };
        panic!("Error initializing worker during post_message: {:?}", e)
    } else {
        Box::new(move || Ok(w.terminate()))
    }
}

#[cfg(feature = "browser")]
static WORKER_URL: std::sync::OnceLock<String> = std::sync::OnceLock::new();

#[cfg(feature = "browser")]
pub fn init(worker_url: String) -> Result<(), String> {
    WORKER_URL.set(worker_url)
}
