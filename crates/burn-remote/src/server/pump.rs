//! The transport-agnostic session pump.
//!
//! One session is one duplex link. [`drive_session`] reads the init handshake, authorizes the peer,
//! binds the session, replies with the device settings, then concurrently drains task responses to
//! the sink (a detached writer) while forwarding submitted task batches from the source to the
//! session worker. This is the single implementation both transports (iroh, websocket) drive — the
//! per-transport modules only build the [`FrameSource`]/[`FrameSink`] halves and the authorizer.

use std::sync::Arc;

use crate::PeerId;
use crate::server::service::{SessionService, parse_init_handshake};
use crate::server::spawn::spawn_detached;
use crate::shared::{
    PROTOCOL_VERSION, RemoteMessage, SessionInfo, SessionInit, TaskResponse, TaskResponseContent,
};
use crate::transport::link::{FrameSink, FrameSource};

/// Drive one session to completion over a duplex link.
///
/// `authorize` runs once, after the init handshake is parsed and before the session is bound — it
/// is where a transport with an authenticated peer identity (iroh) enforces its policy; transports
/// without one (websocket) pass an allow-all closure. `server_peer_id` is echoed to the client in
/// the handshake response (the server's own identity, or `None` for websocket).
///
/// Returns `Err` on a protocol violation or a transport error; the caller logs it. A clean client
/// `Close` (or stream end) returns `Ok(())`.
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

    // Bind the session (creating it + its worker on demand) and claim its response receiver.
    let task_sender = service
        .session_task_sender(init.session_id, init.device_index)
        .await;
    let mut responses = service
        .take_response_receiver(init.session_id, init.device_index)
        .await?;

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
    sink.send(info.into()).await?;

    // Detached writer: drain the session's responses onto the sink until the queue closes (every
    // sender — the worker and any in-flight readback task — has dropped).
    let (writer_done, writer_result) = tokio::sync::oneshot::channel();
    spawn_detached(async move {
        let result = async {
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

    // Reader loop: forward each submitted task batch to the session worker in arrival order.
    let result = loop {
        let Some(frame) = source.recv().await? else {
            break Ok(());
        };
        let messages: Vec<RemoteMessage> = rmp_serde::from_slice(&frame)
            .map_err(|err| format!("Invalid remote task batch: {err}"))?;
        let mut close = false;
        let mut protocol_error = None;
        for message in messages {
            match message {
                RemoteMessage::Task(task) => {
                    task_sender
                        .send(task)
                        .await
                        .map_err(|_| "Session worker stopped".to_string())?;
                }
                RemoteMessage::Close(id) if id == init.session_id => {
                    close = true;
                    break;
                }
                RemoteMessage::Close(id) => {
                    protocol_error = Some(format!(
                        "Session {} attempted to close unrelated session {id}",
                        init.session_id
                    ));
                    break;
                }
                RemoteMessage::Init(_) => {
                    protocol_error = Some("A session stream cannot be initialized twice".into());
                    break;
                }
            }
        }
        if let Some(err) = protocol_error {
            break Err(err);
        }
        if close {
            break Ok(());
        }
    };

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
