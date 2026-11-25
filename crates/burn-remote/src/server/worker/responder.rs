use std::future::Future;

use burn_backend::ops::ReadFuture;
use burn_std::ExecutionError;
use tokio::sync::{mpsc, oneshot};

use crate::server::spawn::{CatchPanic, ResponseTasks};
use crate::shared::{RequestId, SessionId, TaskResponse, TaskResponseContent};

/// Answers a session's requests, at once or once what they wait on resolves.
///
/// # Assumptions
///
/// Off the browser, runs inside a tokio runtime.
///
/// # Guarantees
///
/// The response queue stays open while an answer still pending holds a sender.
pub struct SessionResponder {
    session_id: SessionId,
    sender: mpsc::Sender<TaskResponse>,
    /// The answers still waiting on the device.
    answers: ResponseTasks,
    /// The reads still copying to the host, which a capture waits for.
    reads: ResponseTasks,
}

impl SessionResponder {
    pub fn new(session_id: SessionId, sender: mpsc::Sender<TaskResponse>) -> Self {
        Self {
            session_id,
            sender,
            answers: ResponseTasks::default(),
            reads: ResponseTasks::default(),
        }
    }

    /// A response receiver already dropped is logged as the task's failure.
    pub async fn respond(&self, request_id: RequestId, content: TaskResponseContent) {
        let response = TaskResponse {
            content,
            id: request_id,
        };
        if self.sender.send(response).await.is_err() {
            log::error!(
                "Task on session {} failed: Response receiver dropped before result for request \
                 {request_id} could be sent",
                self.session_id
            );
        }
    }

    /// Answers once `ready` resolves, without holding up the tasks after it.
    pub fn respond_when_ready<Value: Send + 'static>(
        &mut self,
        request_id: RequestId,
        ready: impl ReadFuture<Value> + 'static,
        into_content: fn(Result<Value, ExecutionError>) -> TaskResponseContent,
    ) {
        let answer = self.answer(request_id, ready, into_content);
        self.answers.spawn(answer);
    }

    /// [`respond_when_ready`](Self::respond_when_ready) for a read, which
    /// [`settle_reads`](Self::settle_reads) waits for.
    pub fn respond_after_read<Value: Send + 'static>(
        &mut self,
        request_id: RequestId,
        read: impl ReadFuture<Value> + 'static,
        into_content: fn(Result<Value, ExecutionError>) -> TaskResponseContent,
    ) {
        let answer = self.answer(request_id, read, into_content);
        self.reads.spawn(answer);
    }

    /// Hands `read` on once it resolves, as a read [`settle_reads`](Self::settle_reads) waits for.
    pub fn track_read<Data: Send + 'static>(
        &mut self,
        read: impl ReadFuture<Data> + 'static,
    ) -> impl ReadFuture<Data> + 'static {
        let (read_done, read_back) = oneshot::channel();
        self.reads.spawn(async move {
            let _ = read_done.send(read.await);
        });
        async move {
            read_back.await.unwrap_or_else(|_| {
                Err(ExecutionError::generic(
                    "the session ended before the tensor was read",
                ))
            })
        }
    }

    /// # Guarantees
    ///
    /// Returns once every read handed on or answered so far has finished, and its answer's send
    /// has been attempted. Waits for no other answer.
    pub async fn settle_reads(&mut self) {
        self.reads.settle().await;
    }

    /// Cancels every answer and read still pending, since one may never resolve and would hold
    /// the response queue open.
    ///
    /// # Guarantees
    ///
    /// Each drops its sender when the runtime next runs it.
    #[cfg(not(target_family = "wasm"))]
    pub fn abort_pending(&mut self) {
        self.answers.abort_all();
        self.reads.abort_all();
    }
}

impl SessionResponder {
    fn answer<Value: Send + 'static>(
        &self,
        request_id: RequestId,
        ready: impl ReadFuture<Value> + 'static,
        into_content: fn(Result<Value, ExecutionError>) -> TaskResponseContent,
    ) -> impl Future<Output = ()> + Send + 'static {
        let sender = self.sender.clone();
        async move {
            let response = TaskResponse {
                content: into_content(panic_as_error(CatchPanic::new(ready)).await),
                id: request_id,
            };
            if sender.send(response).await.is_err() {
                log::warn!("Response receiver dropped before answering request {request_id}");
            }
        }
    }
}

/// The result `caught` resolves to, or the server's panic polling it as the error.
pub async fn panic_as_error<T>(
    caught: CatchPanic<Result<T, ExecutionError>>,
) -> Result<T, ExecutionError> {
    caught.await.unwrap_or_else(|message| {
        Err(ExecutionError::generic(format!(
            "The server panicked: {message}"
        )))
    })
}

#[cfg(all(test, not(target_family = "wasm")))]
mod tests {
    use super::*;
    use burn_backend::TensorData;
    use std::pin::pin;
    use std::task::{Context, Waker};
    use std::time::Duration;
    use tokio::time::timeout;

    #[tokio::test]
    async fn a_read_that_panics_is_answered_with_an_error() {
        fn failing_read() -> Result<Vec<TensorData>, ExecutionError> {
            panic!("the backend failed")
        }
        let (sender, mut responses) = mpsc::channel(1);
        let mut responder = SessionResponder::new(SessionId::new(), sender);

        let request_id: RequestId = 7;
        responder.respond_after_read(
            request_id,
            async { failing_read() },
            TaskResponseContent::ReadTensors,
        );

        let response = timeout(Duration::from_secs(10), responses.recv())
            .await
            .expect("a panicking read still answers")
            .unwrap();
        assert_eq!(response.id, request_id);
        match response.content {
            TaskResponseContent::ReadTensors(Err(error)) => {
                assert!(error.to_string().contains("the backend failed"), "{error}")
            }
            other => panic!("expected a read error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn settling_waits_for_a_handed_on_read() {
        settling_waits_for(|responder, released| drop(responder.track_read(gate(released)))).await;
    }

    #[tokio::test]
    async fn settling_waits_for_an_answered_read() {
        settling_waits_for(|responder, released| {
            responder.respond_after_read(
                1,
                async move { gate(released).await.map(|()| Vec::new()) },
                TaskResponseContent::ReadTensors,
            )
        })
        .await;
    }

    /// `read` gates its read on `released`, beside a profile answer that never resolves.
    async fn settling_waits_for(read: impl FnOnce(&mut SessionResponder, oneshot::Receiver<()>)) {
        let (sender, _responses) = mpsc::channel(4);
        let mut responder = SessionResponder::new(SessionId::new(), sender);
        responder.respond_when_ready(
            2,
            std::future::pending::<Result<Option<Duration>, ExecutionError>>(),
            TaskResponseContent::ProfileEnd,
        );
        let (release, released) = oneshot::channel();
        read(&mut responder, released);

        let mut settled = pin!(responder.settle_reads());
        let polled = settled
            .as_mut()
            .poll(&mut Context::from_waker(Waker::noop()));
        assert!(polled.is_pending(), "settled before the read finished");
        release.send(()).unwrap();
        timeout(Duration::from_secs(10), settled)
            .await
            .expect("the pending profile answer held the settle");
    }

    async fn gate(released: oneshot::Receiver<()>) -> Result<(), ExecutionError> {
        released
            .await
            .map_err(|_| ExecutionError::generic("the gate was dropped"))
    }
}
