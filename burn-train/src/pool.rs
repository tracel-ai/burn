// Silences warnings from the compiler about Work.func and child_entry_point
// being unused when the target is not wasm.
#![cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]

//! A small module that's intended to provide an example of creating a pool of
//! web workers which can be used to execute `rayon`-style work.

use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::prelude::*;
use web_sys::{DedicatedWorkerGlobalScope, MessageEvent};
use web_sys::{ErrorEvent, Event, Worker};

#[wasm_bindgen]
pub struct WorkerPool {
    state: Rc<PoolState>,
}

struct PoolState {
    workers: RefCell<Vec<Worker>>,
    callback: Closure<dyn FnMut(Event)>,
}

struct Work {
    func: Box<dyn FnOnce() + Send>,
}

#[wasm_bindgen]
impl WorkerPool {
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
    #[wasm_bindgen(constructor)]
    pub fn new(initial: usize) -> Result<WorkerPool, JsValue> {
        let pool = WorkerPool {
            state: Rc::new(PoolState {
                workers: RefCell::new(Vec::with_capacity(initial)),
                callback: Closure::new(|event: Event| {
                    console_log!("unhandled event: {}", event.type_());
                    crate::logv(&event);
                }),
            }),
        };
        for _ in 0..initial {
            let worker = pool.spawn()?;
            pool.state.push(worker);
        }

        Ok(pool)
    }

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
        console_log!("spawning new worker");
        // TODO: what do do about `./worker.js`:
        //
        // * the path is only known by the bundler. How can we, as a
        //   library, know what's going on?
        // * How do we not fetch a script N times? It internally then
        //   causes another script to get fetched N times...
        let worker = Worker::new("./worker.js")?;

        // With a worker spun up send it the module/memory so it can start
        // instantiating the wasm module. Later it might receive further
        // messages about code to run on the wasm module.
        let array = js_sys::Array::new();
        array.push(&wasm_bindgen::module());
        array.push(&wasm_bindgen::memory());
        worker.post_message(&array)?;

        Ok(worker)
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
        let work = Box::new(Work { func: Box::new(f) });
        let ptr = Box::into_raw(work);
        match worker.post_message(&JsValue::from(ptr as u32)) {
            Ok(()) => Ok(worker),
            Err(e) => {
                unsafe {
                    drop(Box::from_raw(ptr));
                }
                Err(e)
            }
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
        let state = Rc::downgrade(&self.state);
        let worker2 = worker.clone();
        let reclaim_slot = Rc::new(RefCell::new(None));
        let slot2 = reclaim_slot.clone();
        let reclaim = Closure::<dyn FnMut(_)>::new(move |event: Event| {
            if let Some(error) = event.dyn_ref::<ErrorEvent>() {
                console_log!("error in worker: {}", error.message());
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
                *slot2.borrow_mut() = None;
                return;
            }

            console_log!("unhandled event: {}", event.type_());
            crate::logv(&event);
            // TODO: like above, maybe a memory leak here?
        });
        worker.set_onmessage(Some(reclaim.as_ref().unchecked_ref()));
        *reclaim_slot.borrow_mut() = Some(reclaim);
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

/// Entry point invoked by `worker.js`, a bit of a hack but see the "TODO" above
/// about `worker.js` in general.
#[wasm_bindgen]
pub fn child_entry_point(ptr: u32) -> Result<(), JsValue> {
    let ptr = unsafe { Box::from_raw(ptr as *mut Work) };
    let global = js_sys::global().unchecked_into::<DedicatedWorkerGlobalScope>();
    (ptr.func)();
    global.post_message(&JsValue::undefined())?;
    Ok(())
}
