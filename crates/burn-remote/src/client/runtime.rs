//! The client session runtime.
//!
//! A remote device captures a runtime handle ([`Executor`]) at construction — in whatever runtime
//! owns the transport — and the device registry carries it per device. The session's writer and
//! response-demux tasks are then spawned later (on the device-runner thread, off cubecl's global
//! device-registry lock) without depending on an ambient runtime being present at that point.

/// Process-global Tokio runtime for sessions opened outside an ambient runtime.
///
/// Fallback for synchronous callers (scripts, REPLs, notebooks) and the legacy WebSocket path. A
/// native Iroh node also binds on this runtime so its endpoint and session tasks share one executor.
/// When a device is built inside an existing runtime, that one is used instead and this fallback is
/// never created.
#[cfg(not(target_family = "wasm"))]
pub(crate) fn blocking_runtime() -> &'static tokio::runtime::Runtime {
    use std::sync::OnceLock;
    static RUNTIME: OnceLock<tokio::runtime::Runtime> = OnceLock::new();
    RUNTIME.get_or_init(|| {
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("Can build the Burn Remote blocking runtime")
    })
}

/// Executor for a remote session's writer and response-demux tasks.
///
/// Captured once at device construction and carried by the device registry, so the session reuses
/// whatever runtime owns its transport. On native this wraps a Tokio runtime handle: the ambient
/// runtime if one is active, otherwise the shared [`blocking_runtime`]. In the browser Iroh runs on
/// the JS event loop; tasks are spawned with `spawn_local` and blocking calls are unavailable.
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
    /// Capture the executor for a new session at device-construction time.
    ///
    /// Native: the ambient Tokio runtime if one is active, otherwise the shared [`blocking_runtime`].
    /// Browser: the JS event loop. The result is stored in the device registry.
    #[cfg(not(target_family = "wasm"))]
    pub(crate) fn capture() -> Self {
        match tokio::runtime::Handle::try_current() {
            Ok(handle) => Self::Tokio(handle),
            Err(_) => Self::Tokio(blocking_runtime().handle().clone()),
        }
    }

    #[cfg(target_family = "wasm")]
    pub(crate) fn capture() -> Self {
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
                    "Blocking remote calls are not supported on wasm. Establish the session with \
                     `RemoteDevice::connect_async(...).await` and read tensors with \
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
