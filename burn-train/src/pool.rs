//! A small module that's intended to provide an example of creating a pool of
//! web workers which can be used to execute `rayon`-style work.

use crate::util::WORKER_URL;
use log::info;
use std::borrow::BorrowMut;
use std::{cell::RefCell, sync::Arc};
use wasm_bindgen::prelude::*;
use web_sys::{ErrorEvent, Event, MessageEvent, Worker};

// "This is only safe because wasm is currently single-threaded." https://github.com/rustwasm/wasm-bindgen/issues/1505#issuecomment-489300331
unsafe impl Send for PoolState {}
unsafe impl Sync for PoolState {}

lazy_static! {
    pub static ref WORKER_POOL: WorkerPool = WorkerPool {
        state: Arc::new(PoolState {
            workers: RefCell::new(vec!()),
            callback: Closure::new(|event: Event| {
                info!("unhandled event: {:?}", &event);
            })
        })
    };
}

#[wasm_bindgen]
pub struct WorkerPool {
    state: Arc<PoolState>,
}

struct PoolState {
    workers: RefCell<Vec<Worker>>,
    callback: Closure<dyn FnMut(Event)>,
}

/// Creates a new `WorkerPool` which immediately creates `initial` workers.
///
/// The pool created here can be used over a long period of time, and it
/// will be initially primed with `initial` workers. Currently workers are
/// never released or gc'd until the whole pool is destroyed.
///
/// # Errors
///
/// Returns any error that may happen while a JS web worker is created and a
/// message is sent to it.
pub fn init(initial: usize) -> Result<(), JsValue> {
    let pool = WorkerPool {
        state: Arc::new(PoolState {
            workers: RefCell::new(Vec::with_capacity(initial)),
            callback: Closure::new(|event: Event| {
                info!("unhandled event: {:?}", &event);
            }),
        }),
    };

    let workers: Result<_, _> = [0..initial].map(|_| pool.spawn()).into_iter().collect();
    WORKER_POOL.state.workers.replace(workers?);
    Ok(())
}

#[wasm_bindgen]
impl WorkerPool {
    /// Unconditionally spawns a new worker
    ///
    /// The worker isn't registered with this `WorkerPool` but is capable of
    /// executing work for this wasm module.
    ///
    /// # Errors
    ///
    /// Returns any error that may happen while a JS web worker is created and a
    /// message is sent to it.
    fn spawn(&self) -> Result<Worker, JsValue> {
        let mut worker_options = web_sys::WorkerOptions::new();
        worker_options.type_(web_sys::WorkerType::Module);
        web_sys::Worker::new_with_options(
            WORKER_URL
                .get()
                .expect("You must first call `init` with the worker's url."),
            &worker_options,
        )
    }

    /// Fetches a worker from this pool, spawning one if necessary.
    ///
    /// This will attempt to pull an already-spawned web worker from our cache
    /// if one is available, otherwise it will spawn a new worker and return the
    /// newly spawned worker.
    ///
    /// # Errors
    ///
    /// Returns any error that may happen while a JS web worker is created and a
    /// message is sent to it.
    fn worker(&self) -> Result<Worker, JsValue> {
        match self.state.workers.borrow_mut().pop() {
            Some(worker) => Ok(worker),
            None => self.spawn(),
        }
    }

    /// Executes the work `f` in a web worker, spawning a web worker if
    /// necessary.
    ///
    /// This will acquire a web worker and then send the closure `f` to the
    /// worker to execute. The worker won't be usable for anything else while
    /// `f` is executing, and no callbacks are registered for when the worker
    /// finishes.
    ///
    /// # Errors
    ///
    /// Returns any error that may happen while a JS web worker is created and a
    /// message is sent to it.
    fn execute(&self, f: impl FnOnce() + Send + 'static) -> Result<Worker, JsValue> {
        let worker = self.worker()?;

        // Double-boxing because `dyn FnOnce` is unsized and so `Box<dyn FnOnce()>` has
        // an undefined layout (although I think in practice its a pointer and a length?).
        let ptr = Box::into_raw(Box::new(Box::new(f) as Box<dyn FnOnce()>));

        // See `worker.ts` for the format of this message.
        let msg: js_sys::Array = [
            &wasm_bindgen::module(),
            &wasm_bindgen::memory(),
            &JsValue::from(ptr as u32),
        ]
        .into_iter()
        .collect();
        if let Err(e) = worker.post_message(&msg) {
            // We expect the worker to deallocate the box, but if there was an error then
            // we'll do it ourselves.
            let _ = unsafe { Box::from_raw(ptr) };
            Err(format!("Error initializing worker during post_message: {:?}", e).into())
        } else {
            Ok(worker)
        }
    }

    /// Configures an `onmessage` callback for the `worker` specified for the
    /// web worker to be reclaimed and re-inserted into this pool when a message
    /// is received.
    ///
    /// Currently this `WorkerPool` abstraction is intended to execute one-off
    /// style work where the work itself doesn't send any notifications and
    /// whatn it's done the worker is ready to execute more work. This method is
    /// used for all spawned workers to ensure that when the work is finished
    /// the worker is reclaimed back into this pool.
    fn reclaim_on_message(&self, worker: Worker) {
        let state = Arc::downgrade(&self.state);
        let worker2 = worker.clone();
        let mut reclaim_slot = Arc::new(RefCell::new(None));
        let mut slot2 = reclaim_slot.clone();
        let reclaim = Closure::<dyn FnMut(_)>::new(move |event: Event| {
            if let Some(error) = event.dyn_ref::<ErrorEvent>() {
                info!("error in worker: {}", error.message());
                // TODO: this probably leaks memory somehow? It's sort of
                // unclear what to do about errors in workers right now.
                return;
            }

            // If this is a completion event then can deallocate our own
            // callback by clearing out `slot2` which contains our own closure.
            if let Some(_msg) = event.dyn_ref::<MessageEvent>() {
                if let Some(state) = state.upgrade() {
                    state.push(worker2.clone());
                }
                *slot2.borrow_mut() = Arc::new(RefCell::new(None));
                return;
            }

            info!("unhandled event: {:?}", &event);
            // TODO: like above, maybe a memory leak here?
        });
        worker.set_onmessage(Some(reclaim.as_ref().unchecked_ref()));
        *reclaim_slot.borrow_mut() = Arc::new(RefCell::new(Some(reclaim)));
    }
}

impl WorkerPool {
    /// Executes `f` in a web worker.
    ///
    /// This pool manages a set of web workers to draw from, and `f` will be
    /// spawned quickly into one if the worker is idle. If no idle workers are
    /// available then a new web worker will be spawned.
    ///
    /// Once `f` returns the worker assigned to `f` is automatically reclaimed
    /// by this `WorkerPool`. This method provides no method of learning when
    /// `f` completes, and for that you'll need to use `run_notify`.
    ///
    /// # Errors
    ///
    /// If an error happens while spawning a web worker or sending a message to
    /// a web worker, that error is returned.
    pub fn run(&self, f: impl FnOnce() + Send + 'static) -> Result<(), JsValue> {
        let worker = self.execute(f)?;
        self.reclaim_on_message(worker);
        Ok(())
    }
}

impl PoolState {
    fn push(&self, worker: Worker) {
        worker.set_onmessage(Some(self.callback.as_ref().unchecked_ref()));
        worker.set_onerror(Some(self.callback.as_ref().unchecked_ref()));
        let mut workers = self.workers.borrow_mut();
        for prev in workers.iter() {
            let prev: &JsValue = prev;
            let worker: &JsValue = &worker;
            assert!(prev != worker);
        }
        workers.push(worker);
    }
}

/// Entry point invoked by `worker.js`
#[wasm_bindgen]
pub fn child_entry_point(ptr: u32) {
    let work = unsafe { Box::from_raw(ptr as *mut Box<dyn FnOnce()>) };
    (*work)();
}
