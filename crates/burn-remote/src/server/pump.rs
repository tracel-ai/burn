//! The transport-agnostic session pump.
//!
//! One session is one duplex link. [`drive_session`] reads the init handshake, authorizes the peer,
//! binds the session, replies with the device settings, then concurrently drains task responses to
//! the sink (a detached writer) while forwarding submitted task batches from the source to the
//! session worker. This is the single implementation both transports (iroh, websocket) drive — the
//! per-transport modules only build the [`FrameSource`]/[`FrameSink`] halves and the authorizer.

use std::sync::Arc;

use crate::PeerId;
use crate::server::service::{SessionChannels, SessionService, parse_init_handshake};
use crate::server::spawn::spawn_detached;
use crate::shared::{
    PROTOCOL_VERSION, RemoteMessage, SessionId, SessionInfo, SessionInit, Task, TaskResponse,
    TaskResponseContent,
};
use crate::transport::link::{FrameSink, FrameSource};
use tokio::sync::mpsc;

/// Drive one session to completion over a duplex link.
///
/// `authorize` runs once, after the init handshake is parsed and before the session is bound — it
/// is where a transport with an authenticated peer identity (iroh) enforces its policy; transports
/// without one (websocket) pass an allow-all closure. `server_peer_id` is echoed to the client in
/// the handshake response (the server's own identity, or `None` for websocket).
///
/// Returns `Err` on a protocol violation or a failed read; the caller logs it. A clean client
/// `Close` (or stream end) returns `Ok(())`. A failed response send, the handshake reply included,
/// only stops the writer and is logged.
pub(crate) async fn drive_session<Src, Snk, S, A>(
    mut source: Src,
    mut sink: Snk,
    service: Arc<S>,
    server_peer_id: Option<PeerId>,
    authorize: A,
) -> Result<(), String>
where
    Src: FrameSource,
    Snk: FrameSink,
    S: SessionService,
    A: FnOnce(&SessionInit) -> Result<(), String>,
{
    // The session stream opens with exactly one `Init` frame.
    let handshake = source
        .recv()
        .await?
        .ok_or_else(|| "Session stream closed before initialization".to_string())?;
    let init = parse_init_handshake(&handshake)?;

    // Authorize before any session state is created.
    authorize(&init)?;

    // Reply with the selected device's settings + this server's identity, so the client can fill in
    // `RemoteDevice::defaults`/`enumerate` without an extra round-trip.
    let info = TaskResponse {
        id: 0,
        content: TaskResponseContent::Init(SessionInfo {
            version: PROTOCOL_VERSION,
            settings: service.device_settings(init.device_index),
            device_count: service.device_count(),
            peer_id: server_peer_id,
        }),
    };
    let info = rmp_serde::to_vec(&info)
        .map_err(|err| format!("Failed to encode session handshake response: {err}"))?;

    let SessionChannels {
        tasks: task_sender,
        mut responses,
    } = service.bind(init.session_id, init.device_index).await?;

    // Sends the handshake reply itself, so a failed send cannot skip the teardown.
    let (writer_done, writer_result) = tokio::sync::oneshot::channel();
    spawn_detached(async move {
        let result = async {
            sink.send(info.into()).await?;
            while let Some(response) = responses.recv().await {
                let bytes = rmp_serde::to_vec(&response)
                    .map_err(|err| format!("Failed to encode task response: {err}"))?;
                sink.send(bytes.into()).await?;
            }
            sink.close().await
        }
        .await;
        let _ = writer_done.send(result);
    });

    let result = forward_tasks(source, &task_sender, init.session_id).await;

    // Teardown: drop our task sender and close the session so its worker drains and exits, which
    // closes the response queue and ends the writer; then await the writer so we don't tear the
    // runtime down mid-send.
    drop(task_sender);
    service.close(init.session_id).await;
    match writer_result.await {
        Ok(Ok(())) => {}
        Ok(Err(err)) => log::warn!("Session response writer failed: {err}"),
        Err(_) => log::warn!("Session response writer stopped before finishing"),
    }
    result
}

/// Forward each submitted task batch to the session worker in arrival order, until the client
/// closes the session or its stream ends.
async fn forward_tasks(
    mut source: impl FrameSource,
    task_sender: &mpsc::Sender<Task>,
    session_id: SessionId,
) -> Result<(), String> {
    while let Some(frame) = source.recv().await? {
        let messages: Vec<RemoteMessage> = rmp_serde::from_slice(&frame)
            .map_err(|err| format!("Invalid remote task batch: {err}"))?;
        for message in messages {
            match message {
                RemoteMessage::Task(task) => task_sender
                    .send(task)
                    .await
                    .map_err(|_| "Session worker stopped".to_string())?,
                RemoteMessage::Close(id) if id == session_id => return Ok(()),
                RemoteMessage::Close(id) => {
                    return Err(format!(
                        "Session {session_id} attempted to close unrelated session {id}"
                    ));
                }
                RemoteMessage::Init(_) => {
                    return Err("A session stream cannot be initialized twice".into());
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_std::{BoolDType, DeviceSettings, FloatDType, IntDType};
    use bytes::Bytes;
    use std::{collections::VecDeque, sync::Mutex};

    #[derive(Default)]
    struct FakeService {
        /// The worker's end of the response queue. Dropping it on close ends the writer, as the
        /// worker exiting does.
        responses: Mutex<Option<mpsc::Sender<TaskResponse>>>,
        /// The worker's end of the task queue, kept so a task sent in a test is not refused.
        tasks: Mutex<Option<mpsc::Receiver<Task>>>,
        closed: Mutex<Vec<SessionId>>,
        bound_elsewhere: bool,
    }

    impl SessionService for FakeService {
        async fn bind(
            &self,
            session_id: SessionId,
            _device_index: u32,
        ) -> Result<SessionChannels, String> {
            if self.bound_elsewhere {
                return Err(format!(
                    "Session {session_id} is already bound to another stream"
                ));
            }
            let (tasks, worker) = mpsc::channel(1);
            let (sender, responses) = mpsc::channel(1);
            *self.tasks.lock().unwrap() = Some(worker);
            *self.responses.lock().unwrap() = Some(sender);
            Ok(SessionChannels { tasks, responses })
        }

        fn device_settings(&self, _device_index: u32) -> DeviceSettings {
            DeviceSettings::with_dtypes(FloatDType::F32, IntDType::I32, BoolDType::Native)
        }

        fn device_count(&self) -> u32 {
            1
        }

        async fn close(&self, session_id: SessionId) {
            self.closed.lock().unwrap().push(session_id);
            self.responses.lock().unwrap().take();
        }
    }

    struct ScriptedSource(VecDeque<Result<Option<Bytes>, String>>);

    impl FrameSource for ScriptedSource {
        async fn recv(&mut self) -> Result<Option<Bytes>, String> {
            self.0.pop_front().unwrap_or(Ok(None))
        }
    }

    struct DiscardingSink;

    impl FrameSink for DiscardingSink {
        async fn send(&mut self, _frame: Bytes) -> Result<(), String> {
            Ok(())
        }

        async fn close(&mut self) -> Result<(), String> {
            Ok(())
        }
    }

    struct FailingSink;

    impl FrameSink for FailingSink {
        async fn send(&mut self, _frame: Bytes) -> Result<(), String> {
            Err("connection reset".into())
        }

        async fn close(&mut self) -> Result<(), String> {
            Ok(())
        }
    }

    fn handshake(session_id: SessionId) -> Bytes {
        let init = vec![RemoteMessage::Init(SessionInit::new(session_id, 0, vec![]))];
        rmp_serde::to_vec(&init).unwrap().into()
    }

    #[tokio::test]
    async fn a_stream_that_fails_still_closes_its_session() {
        let service = Arc::new(FakeService::default());
        let session_id = SessionId::new();
        let source = ScriptedSource(
            [
                Ok(Some(handshake(session_id))),
                Err("connection reset".into()),
            ]
            .into(),
        );

        let result = drive_session(source, DiscardingSink, service.clone(), None, |_| Ok(())).await;

        assert_eq!(result, Err("connection reset".to_string()));
        assert_eq!(*service.closed.lock().unwrap(), [session_id]);
    }

    #[tokio::test]
    async fn a_handshake_reply_that_fails_still_closes_its_session() {
        let service = Arc::new(FakeService::default());
        let session_id = SessionId::new();
        let source = ScriptedSource([Ok(Some(handshake(session_id)))].into());

        let _ = drive_session(source, FailingSink, service.clone(), None, |_| Ok(())).await;

        assert_eq!(*service.closed.lock().unwrap(), [session_id]);
    }

    #[tokio::test]
    async fn a_stream_that_cannot_bind_leaves_the_session_alone() {
        let service = Arc::new(FakeService {
            bound_elsewhere: true,
            ..Default::default()
        });
        let session_id = SessionId::new();
        let source = ScriptedSource([Ok(Some(handshake(session_id)))].into());

        let result = drive_session(source, DiscardingSink, service.clone(), None, |_| Ok(())).await;

        assert!(result.is_err());
        assert!(service.closed.lock().unwrap().is_empty());
    }
}
