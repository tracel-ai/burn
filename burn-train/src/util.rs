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
    rayon::spawn(f);
    Box::new(|| Ok(()))
}

#[cfg(feature = "browser")]
pub use wasm_bindgen_rayon::init_thread_pool;
