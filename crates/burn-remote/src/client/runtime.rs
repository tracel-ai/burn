//! The client session runtime.

#[cfg(not(target_family = "wasm"))]
use crate::runtime::blocking_runtime;

/// Executor for a remote session's writer and response-demux tasks: [`blocking_runtime`] on
/// native, and the JS event loop in the browser, where tasks are spawned with `spawn_local` and
/// blocking calls are unavailable.
#[derive(Clone, Debug)]
pub(crate) enum Executor {
    #[cfg(not(target_family = "wasm"))]
    Tokio(tokio::runtime::Handle),
    #[cfg(target_family = "wasm")]
    WasmLocal,
}

/// Handle to a spawned session task. Joinable on native; a no-op in the browser where tasks
/// run on the event loop and cannot be awaited.
pub(crate) struct SpawnHandle {
    #[cfg(not(target_family = "wasm"))]
    inner: tokio::task::JoinHandle<()>,
}

impl Executor {
    #[cfg(not(target_family = "wasm"))]
    pub(crate) fn session() -> Self {
        Self::Tokio(blocking_runtime().handle().clone())
    }

    #[cfg(target_family = "wasm")]
    pub(crate) fn session() -> Self {
        Self::WasmLocal
    }

    pub(crate) fn block_on<F: core::future::Future>(&self, future: F) -> F::Output {
        match self {
            #[cfg(not(target_family = "wasm"))]
            Self::Tokio(handle) => handle.block_on(future),
            #[cfg(target_family = "wasm")]
            Self::WasmLocal => {
                core::mem::drop(future);
                panic!(
                    "Blocking remote calls are not supported on wasm. Connect with \
                     `Device::remote_options(&host).init_async().await` and read tensors with \
                     `into_data_async().await`."
                )
            }
        }
    }

    #[cfg(not(target_family = "wasm"))]
    pub(crate) fn spawn<F>(&self, future: F) -> SpawnHandle
    where
        F: core::future::Future<Output = ()> + Send + 'static,
    {
        match self {
            Self::Tokio(handle) => SpawnHandle {
                inner: handle.spawn(future),
            },
        }
    }

    /// Spawn a session task on the browser event loop. The Iroh streams these tasks own are not
    /// `Send`, which is why the wasm path uses `spawn_local` rather than the native `spawn`.
    #[cfg(target_family = "wasm")]
    pub(crate) fn spawn<F>(&self, future: F) -> SpawnHandle
    where
        F: core::future::Future<Output = ()> + 'static,
    {
        wasm_bindgen_futures::spawn_local(future);
        SpawnHandle {}
    }

    /// Wait for a spawned task to finish. No-op in the browser.
    #[cfg(not(target_family = "wasm"))]
    pub(crate) fn join(&self, handle: SpawnHandle) {
        let _ = self.block_on(handle.inner);
    }
}
