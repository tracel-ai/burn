//! Burn's own Tokio runtime, shared by native clients and servers.

use core::fmt;
use std::{
    panic::{AssertUnwindSafe, catch_unwind, resume_unwind},
    sync::OnceLock,
};

use tokio::runtime::Runtime;

/// Burn's own Tokio runtime, which binds the Iroh endpoints Burn owns, runs every native client
/// session and hosts a blocking `serve`.
///
/// Sessions never run on the caller's runtime: a current-thread runtime blocked in a synchronous
/// call could not drive them, and a runtime the caller shuts down would take them along.
pub(crate) fn blocking_runtime() -> &'static Runtime {
    static RUNTIME: OnceLock<Runtime> = OnceLock::new();
    RUNTIME.get_or_init(|| {
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("Can build the Burn Remote blocking runtime")
    })
}

/// The runtime dropped the work before it ran to completion.
#[derive(Debug)]
pub(crate) struct Interrupted;

impl fmt::Display for Interrupted {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Burn Remote's runtime dropped the work before it finished")
    }
}

impl std::error::Error for Interrupted {}

/// Run `work` on a blocking thread of Burn's runtime and block the calling thread until it
/// finishes. A plain channel, so a caller on a runtime thread blocks it rather than panicking.
pub(crate) fn wait<T: Send + 'static>(
    work: impl FnOnce() -> T + Send + 'static,
) -> Result<T, Interrupted> {
    let (sender, receiver) = std::sync::mpsc::channel();
    blocking_runtime().spawn_blocking(move || {
        let _ = sender.send(catch_unwind(AssertUnwindSafe(work)));
    });
    match receiver.recv() {
        Ok(Ok(value)) => Ok(value),
        Ok(Err(panic)) => resume_unwind(panic),
        Err(_) => Err(Interrupted),
    }
}

/// Run `work` on a blocking thread of Burn's runtime, awaited from any executor.
#[cfg(feature = "client")]
pub(crate) async fn run<T: Send + 'static>(
    work: impl FnOnce() -> T + Send + 'static,
) -> Result<T, Interrupted> {
    match blocking_runtime().spawn_blocking(work).await {
        Ok(value) => Ok(value),
        Err(err) => match err.try_into_panic() {
            Ok(panic) => resume_unwind(panic),
            Err(_) => Err(Interrupted),
        },
    }
}
