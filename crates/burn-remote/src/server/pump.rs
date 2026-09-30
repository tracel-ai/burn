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
    PROTOCOL_VERSION, RemoteMessage, SessionId, SessionInfo, SessionInit, SessionRefusal, Task,
    TaskResponse, TaskResponseContent,
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
/// Returns `Err` on a protocol violation, a refused session (after telling the client its
/// category), or a failed read or write; the caller logs it. A clean client `Close` (or stream
/// end) drains the remaining responses before returning `Ok(())`.
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
    let device_count = service.device_count();
    let init = match admit(&handshake, authorize, device_count) {
        Ok(init) => init,
        Err(Refused { refusal, reason }) => {
            refuse(&mut sink, refusal).await;
            return Err(reason);
        }
    };

    // Reply with the selected device's settings + this server's identity, so the client can fill in
    // `RemoteDevice::defaults`/`enumerate` without an extra round-trip.
    let info = TaskResponse {
        id: 0,
        content: TaskResponseContent::Init(SessionInfo {
            version: PROTOCOL_VERSION,
            settings: service.device_settings(init.device_index),
            device_count,
            peer_id: server_peer_id,
        }),
    };
    let info = rmp_serde::to_vec(&info)
        .map_err(|err| format!("Failed to encode session handshake response: {err}"))?;

    let SessionChannels {
        tasks: task_sender,
        mut responses,
    } = service.bind(init.session_id, init.device_index).await?;

    // The writer sends the handshake reply before any task responses.
    let (writer_done, mut writer_result) = tokio::sync::oneshot::channel();
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

    // Either half ending triggers teardown: a failed write need not close the incoming half.
    // Save a completed writer result so we don't poll the oneshot receiver twice.
    let (read_result, completed_writer) = tokio::select! {
        result = forward_tasks(source, &task_sender, init.session_id) => (result, None),
        result = &mut writer_result => (Ok(()), Some(result)),
    };

    // Drop our task sender and close the session so its worker drains and starts closing, which
    // closes the response queue and ends the writer; then await the writer so we don't tear the
    // runtime down mid-send.
    drop(task_sender);
    service.close(init.session_id).await;
    let write_result = match completed_writer {
        Some(result) => result,
        None => writer_result.await,
    }
    .unwrap_or_else(|_| Err("Session response writer stopped before finishing".into()));
    read_result.and(write_result)
}

/// An `Init` the server will not serve: the category the client is told, and the reason the
/// server logs.
struct Refused {
    refusal: SessionRefusal,
    reason: String,
}

/// Check an `Init` before any session state exists: that it can be read, then the authorizer,
/// then the device. An unauthorized client must not learn how many devices the server hosts.
fn admit(
    handshake: &[u8],
    authorize: impl FnOnce(&SessionInit) -> Result<(), String>,
    device_count: u32,
) -> Result<SessionInit, Refused> {
    let init = parse_init_handshake(handshake).map_err(|reason| Refused {
        refusal: SessionRefusal::IncompatibleProtocol,
        reason,
    })?;
    authorize(&init).map_err(|reason| Refused {
        refusal: SessionRefusal::Unauthorized,
        reason,
    })?;
    if init.device_index >= device_count {
        return Err(Refused {
            refusal: SessionRefusal::NoSuchDevice { device_count },
            reason: format!(
                "Session {} asked for device {}, but this server hosts {device_count} device(s)",
                init.session_id, init.device_index
            ),
        });
    }
    Ok(init)
}

/// Answer a refused `Init` with its category, then close, so the client can report more than a
/// closed stream. The session is refused whether or not the client hears it.
async fn refuse(sink: &mut impl FrameSink, refusal: SessionRefusal) {
    let reply = TaskResponse {
        id: 0,
        content: TaskResponseContent::InitRefused(refusal),
    };
    if let Ok(frame) = rmp_serde::to_vec(&reply) {
        let _ = sink.send(frame.into()).await;
    }
    let _ = sink.close().await;
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

    /// Send the handshake, then keep the incoming half open indefinitely.
    struct OpenSource(Option<Bytes>);

    impl FrameSource for OpenSource {
        async fn recv(&mut self) -> Result<Option<Bytes>, String> {
            match self.0.take() {
                Some(frame) => Ok(Some(frame)),
                None => std::future::pending().await,
            }
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

    /// Keeps every frame written to it.
    #[derive(Clone, Default)]
    struct RecordingSink(Arc<Mutex<Vec<Bytes>>>);

    impl RecordingSink {
        fn refusals(&self) -> Vec<SessionRefusal> {
            self.0
                .lock()
                .unwrap()
                .iter()
                .map(|frame| {
                    match rmp_serde::from_slice::<TaskResponse>(frame)
                        .unwrap()
                        .content
                    {
                        TaskResponseContent::InitRefused(refusal) => refusal,
                        other => panic!("expected a refusal, got {other:?}"),
                    }
                })
                .collect()
        }
    }

    impl FrameSink for RecordingSink {
        async fn send(&mut self, frame: Bytes) -> Result<(), String> {
            self.0.lock().unwrap().push(frame);
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

    /// The one device `FakeService` hosts.
    const HOSTED_DEVICE: u32 = 0;

    fn handshake(session_id: SessionId, device_index: u32) -> Bytes {
        let init = vec![RemoteMessage::Init(SessionInit::new(
            session_id,
            device_index,
            vec![],
        ))];
        rmp_serde::to_vec(&init).unwrap().into()
    }

    #[tokio::test]
    async fn a_stream_that_fails_still_closes_its_session() {
        let service = Arc::new(FakeService::default());
        let session_id = SessionId::new();
        let source = ScriptedSource(
            [
                Ok(Some(handshake(session_id, HOSTED_DEVICE))),
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
        let source = OpenSource(Some(handshake(session_id, HOSTED_DEVICE)));

        let result = tokio::time::timeout(
            std::time::Duration::from_secs(1),
            drive_session(source, FailingSink, service.clone(), None, |_| Ok(())),
        )
        .await
        .expect("a failed handshake reply must close the session even if input stays open");

        assert_eq!(result, Err("connection reset".to_string()));
        assert_eq!(*service.closed.lock().unwrap(), [session_id]);
    }

    #[tokio::test]
    async fn a_stream_that_cannot_bind_leaves_the_session_alone() {
        let service = Arc::new(FakeService {
            bound_elsewhere: true,
            ..Default::default()
        });
        let session_id = SessionId::new();
        let source = ScriptedSource([Ok(Some(handshake(session_id, HOSTED_DEVICE)))].into());

        let result = drive_session(source, DiscardingSink, service.clone(), None, |_| Ok(())).await;

        assert!(result.is_err());
        assert!(service.closed.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn a_device_the_server_does_not_host_is_refused() {
        let service = Arc::new(FakeService::default());
        let unhosted_device = service.device_count();
        let source =
            ScriptedSource([Ok(Some(handshake(SessionId::new(), unhosted_device)))].into());
        let sink = RecordingSink::default();

        let result = drive_session(source, sink.clone(), service.clone(), None, |_| Ok(())).await;

        let err = result.expect_err("the session was bound to a device the server does not host");
        assert!(
            err.contains(&format!("device {unhosted_device}")),
            "got: {err}"
        );
        assert!(service.tasks.lock().unwrap().is_none());
        assert_eq!(
            sink.refusals(),
            [SessionRefusal::NoSuchDevice { device_count: 1 }]
        );
    }

    #[tokio::test]
    async fn an_unauthorized_session_is_told_only_that_it_was_refused() {
        let service = Arc::new(FakeService::default());
        let unhosted_device = service.device_count();
        let source =
            ScriptedSource([Ok(Some(handshake(SessionId::new(), unhosted_device)))].into());
        let sink = RecordingSink::default();

        let result = drive_session(source, sink.clone(), service.clone(), None, |_| {
            Err("peer 7 is not on the allowlist".to_string())
        })
        .await;

        assert_eq!(result, Err("peer 7 is not on the allowlist".to_string()));
        assert!(service.tasks.lock().unwrap().is_none());
        assert_eq!(sink.refusals(), [SessionRefusal::Unauthorized]);
    }

    #[tokio::test]
    async fn a_client_on_another_protocol_version_is_refused_before_authorization() {
        let service = Arc::new(FakeService::default());
        let mut init = SessionInit::new(SessionId::new(), HOSTED_DEVICE, vec![]);
        init.version = PROTOCOL_VERSION + 1;
        let handshake = rmp_serde::to_vec(&vec![RemoteMessage::Init(init)]).unwrap();
        let source = ScriptedSource([Ok(Some(handshake.into()))].into());
        let sink = RecordingSink::default();

        let result = drive_session(source, sink.clone(), service.clone(), None, |_| {
            panic!("an incompatible client reached the authorizer")
        })
        .await;

        assert!(result.is_err());
        assert!(service.tasks.lock().unwrap().is_none());
        assert_eq!(sink.refusals(), [SessionRefusal::IncompatibleProtocol]);
    }
}
