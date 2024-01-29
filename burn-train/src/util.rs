#![allow(missing_docs)]

pub trait AsyncTask {
    fn join(self: Box<Self>) -> Result<(), ()>;
}

pub type AsyncTaskBoxed = Box<dyn AsyncTask>;

struct Thread {
    join: Box<dyn FnOnce() -> Result<(), ()>>,
}

impl AsyncTask for Thread {
    fn join(self: Box<Self>) -> Result<(), ()> {
        (self.join)()
    }
}

impl Thread {
    fn new(join: Box<dyn FnOnce() -> Result<(), ()>>) -> Self {
        Thread { join }
    }
}

#[cfg(not(feature = "browser"))]
pub fn spawn<F>(f: F) -> AsyncTaskBoxed
where
    F: FnOnce(),
    F: Send + 'static,
{
    let handle = std::thread::spawn(f);
    Box::new(Thread::new(Box::new(move || handle.join().map_err(|_| ()))))
}

#[cfg(feature = "browser")]
pub fn spawn<F>(f: F) -> AsyncTaskBoxed
where
    F: FnOnce(),
    F: Send + 'static,
{
    rayon::spawn(f);
    Box::new(Thread::new(Box::new(|| Ok(()))))
}

#[cfg(feature = "browser")]
pub use wasm_bindgen_rayon::init_thread_pool;
