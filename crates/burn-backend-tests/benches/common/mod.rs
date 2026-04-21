//! Shared TestBackend setup for benches.
//!
//! Mirrors `tests/common/backend.rs`: `TestBackend = burn_dispatch::Dispatch`, with a `#[ctor]`
//! that pins the default float/int dtypes to `f32`/`i32` so backends that advertise a different
//! default (e.g. bf16) don't silently convert bench inputs.

use std::cell::Cell;
use std::panic::{self, AssertUnwindSafe, Location};
use std::sync::Mutex;

use burn_tensor::backend::Backend;
use ctor::ctor;

pub type FloatElem = f32;
pub type IntElem = i32;
pub type TestBackend = burn_dispatch::Dispatch;

#[ctor]
fn init_device_settings() {
    let device = burn_dispatch::DispatchDevice::default();
    burn_tensor::set_default_dtypes::<TestBackend>(
        &device,
        <FloatElem as burn_tensor::Element>::dtype(),
        <IntElem as burn_tensor::Element>::dtype(),
    )
    .unwrap();
}

/// Block until all outstanding ops on the default device complete.
///
/// GPU backends (cuda, wgpu, rocm, metal, vulkan) dispatch ops asynchronously; without a sync
/// barrier inside the timed region a bench would measure dispatch latency, not execution time.
/// On CPU backends (flex, ndarray) this is a no-op via the `Backend::sync` default.
#[inline]
pub fn sync() {
    TestBackend::sync(&Default::default()).unwrap();
}

// --- Panic-tolerant bench execution -----------------------------------------
//
// Some ops are not implemented on every backend (quantization on tch, deformable conv on metal,
// etc.). Without handling, the first panic in a bench binary aborts the whole run. We pre-flight
// each bench under `catch_unwind`; on panic we record the failure, skip the bench loop, and let
// subsequent benches in the same binary continue. `report_failures` at the end of `main()`
// prints a summary.

thread_local! {
    static SUPPRESS_PANIC_OUTPUT: Cell<bool> = const { Cell::new(false) };
}

struct BenchFailure {
    location: String,
    message: String,
}

static FAILURES: Mutex<Vec<BenchFailure>> = Mutex::new(Vec::new());

#[ctor]
fn install_panic_hook() {
    // Chain onto the existing hook rather than replacing it so truly unexpected panics (outside
    // `bench_synced`'s catch_unwind window) still print normally.
    let default_hook = panic::take_hook();
    panic::set_hook(Box::new(move |info| {
        if !SUPPRESS_PANIC_OUTPUT.with(|s| s.get()) {
            default_hook(info);
        }
    }));
}

fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        s.to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "(non-string panic payload)".to_string()
    }
}

/// Run a setup closure tolerantly: if it panics (typically because the backend doesn't support
/// an op used during setup), returns `None` and records the failure location. Lets benches that
/// rely on op-heavy setup (e.g. `make_qtensor` calls `quantize_dynamic`, fft inverse benches call
/// `rfft` to produce the input) fall through to a no-op without taking down the whole binary.
#[track_caller]
pub fn try_setup<T>(f: impl FnOnce() -> T) -> Option<T> {
    let loc = Location::caller();
    SUPPRESS_PANIC_OUTPUT.with(|s| s.set(true));
    let r = panic::catch_unwind(AssertUnwindSafe(f));
    SUPPRESS_PANIC_OUTPUT.with(|s| s.set(false));
    match r {
        Ok(v) => Some(v),
        Err(payload) => {
            FAILURES.lock().unwrap().push(BenchFailure {
                location: format!("{}:{} (setup)", loc.file(), loc.line()),
                message: panic_message(payload),
            });
            None
        }
    }
}

/// Print a summary of benches that panicked during this run, if any. Call from `main()` after
/// `divan::main()`.
pub fn report_failures() {
    let failures = FAILURES.lock().unwrap();
    if failures.is_empty() {
        return;
    }
    eprintln!();
    eprintln!("=== {} bench(es) skipped due to panic ===", failures.len());
    for f in failures.iter() {
        eprintln!("  [{}] {}", f.location, f.message);
    }
}

/// Extension trait adding a synced, panic-tolerant variant of `Bencher::bench`.
///
/// `bench_synced` runs the op then forces a device sync before returning, so the timed region
/// covers actual execution on async backends. If the op panics (typically "not implemented" on a
/// backend that doesn't support it), the failure is recorded and the bench is replaced with a
/// no-op so later benches in the same binary keep running.
pub trait BencherExt<'a, 'b> {
    fn bench_synced<O, F>(self, benched: F)
    where
        F: Fn() -> O + Sync;
}

impl<'a, 'b> BencherExt<'a, 'b> for divan::Bencher<'a, 'b> {
    #[track_caller]
    fn bench_synced<O, F>(self, benched: F)
    where
        F: Fn() -> O + Sync,
    {
        let loc = Location::caller();

        SUPPRESS_PANIC_OUTPUT.with(|s| s.set(true));
        let result = panic::catch_unwind(AssertUnwindSafe(|| {
            let r = benched();
            sync();
            drop(r);
        }));
        SUPPRESS_PANIC_OUTPUT.with(|s| s.set(false));

        match result {
            Ok(()) => {
                self.bench(move || {
                    let r = benched();
                    sync();
                    r
                });
            }
            Err(payload) => {
                FAILURES.lock().unwrap().push(BenchFailure {
                    location: format!("{}:{}", loc.file(), loc.line()),
                    message: panic_message(payload),
                });
                self.bench(|| ());
            }
        }
    }
}
