use core::{future::Future, time::Duration};

#[cfg(not(target_family = "wasm"))]
pub(crate) async fn sleep(duration: Duration) {
    tokio::time::sleep(duration).await;
}

#[cfg(target_family = "wasm")]
pub(crate) async fn sleep(duration: Duration) {
    gloo_timers::future::sleep(duration).await;
}

/// `Err(())` if `duration` elapses before `future` completes.
#[cfg(not(target_family = "wasm"))]
pub(crate) async fn timeout<F: Future>(duration: Duration, future: F) -> Result<F::Output, ()> {
    tokio::time::timeout(duration, future).await.map_err(|_| ())
}

#[cfg(target_family = "wasm")]
pub(crate) async fn timeout<F: Future>(duration: Duration, future: F) -> Result<F::Output, ()> {
    use futures_util::future::{Either, select};

    let timer = gloo_timers::future::sleep(duration);
    futures_util::pin_mut!(future, timer);
    match select(future, timer).await {
        Either::Left((output, _)) => Ok(output),
        Either::Right(((), _)) => Err(()),
    }
}
