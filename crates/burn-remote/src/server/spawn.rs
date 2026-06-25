use core::future::Future;

#[cfg(not(target_family = "wasm"))]
pub(crate) fn spawn_detached<F>(future: F)
where
    F: Future<Output = ()> + Send + 'static,
{
    tokio::spawn(future);
}

#[cfg(target_family = "wasm")]
pub(crate) fn spawn_detached<F>(future: F)
where
    F: Future<Output = ()> + 'static,
{
    wasm_bindgen_futures::spawn_local(future);
}
