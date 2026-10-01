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

/// Detached tasks that each send one response. Each holds a response sender, and the response
/// queue stays open until every sender drops, so a failed session has to be able to stop them.
#[derive(Debug, Default)]
pub(crate) struct ResponseTasks {
    #[cfg(not(target_family = "wasm"))]
    running: Vec<tokio::task::AbortHandle>,
}

impl ResponseTasks {
    #[cfg(not(target_family = "wasm"))]
    pub(crate) fn spawn<F>(&mut self, future: F)
    where
        F: Future<Output = ()> + Send + 'static,
    {
        self.running.retain(|task| !task.is_finished());
        self.running.push(tokio::spawn(future).abort_handle());
    }

    #[cfg(target_family = "wasm")]
    pub(crate) fn spawn<F>(&mut self, future: F)
    where
        F: Future<Output = ()> + 'static,
    {
        spawn_detached(future);
    }

    /// Cancel every task still running; each drops its sender when the runtime next runs it.
    #[cfg(not(target_family = "wasm"))]
    pub(crate) fn abort_all(&mut self) {
        for task in self.running.drain(..) {
            task.abort();
        }
    }
}

/// Resolve when the process is asked to stop (Ctrl+C, or `SIGTERM` on Unix).
///
/// The single shutdown trigger shared by the turnkey WebSocket and Iroh server entry points.
#[cfg(all(
    not(target_family = "wasm"),
    any(feature = "websocket", feature = "iroh")
))]
pub(crate) async fn os_shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install signal handler")
            .recv()
            .await;
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
}

#[cfg(all(test, not(target_family = "wasm")))]
mod tests {
    use super::*;
    use std::time::Duration;
    use tokio::{sync::mpsc, time::timeout};

    #[tokio::test]
    async fn aborting_a_stalled_task_releases_its_response_sender() {
        let (sender, mut responses) = mpsc::channel::<()>(1);
        let mut tasks = ResponseTasks::default();
        tasks.spawn(async move {
            let _held = sender;
            std::future::pending::<()>().await
        });

        tokio::task::yield_now().await;
        tasks.abort_all();

        let closed = timeout(Duration::from_secs(10), responses.recv()).await;
        assert_eq!(closed, Ok(None));
    }
}
