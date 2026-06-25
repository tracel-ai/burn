//! The `/fetch` connection handler.

use std::future::Future;
#[cfg(feature = "websocket")]
use std::sync::Arc;

#[cfg(feature = "websocket")]
use burn_communication::{CommunicationChannel, Message};
use burn_std::DeviceSettings;
use tokio::sync::mpsc;

#[cfg(feature = "websocket")]
use super::base::parse_init_handshake;
#[cfg(feature = "websocket")]
use crate::PeerId;
#[cfg(feature = "websocket")]
use crate::shared::{PROTOCOL_VERSION, SessionInfo, TaskResponseContent};
use crate::shared::{SessionId, TaskResponse};

/// What a `/fetch` connection needs from the session layer: the device metadata returned on the
/// init handshake, and the session's result receiver to drain.
pub(crate) trait FetchService: Send + Sync + 'static {
    /// The default settings of the device at `device_index`, returned on the handshake.
    fn device_settings(&self, device_index: u32) -> DeviceSettings;

    /// The number of devices this server hosts, returned on the handshake so the client can
    /// enumerate every device behind the address.
    fn device_count(&self) -> u32;

    /// Claim the session's result receiver. Errors if a fetcher is already registered — the
    /// protocol allows only one fetch socket per session.
    fn take_response_receiver(
        &self,
        session_id: SessionId,
        device_index: u32,
    ) -> impl Future<Output = Result<mpsc::Receiver<TaskResponse>, String>> + Send;
}

/// The `/fetch` connection: answer the init handshake, then drain the session's result queue
/// onto the socket until it closes.
#[cfg(feature = "websocket")]
pub(crate) struct FetchHandler<S, C> {
    service: Arc<S>,
    socket: C,
}

#[cfg(feature = "websocket")]
impl<S: FetchService, C: CommunicationChannel> FetchHandler<S, C> {
    pub(crate) fn new(service: Arc<S>, socket: C) -> Self {
        Self { service, socket }
    }

    pub(crate) async fn run(mut self) {
        let Some((session_id, device_index)) = self.handshake().await else {
            return;
        };

        log::debug!(
            "[Fetch handler] New connection for {session_id} Device({device_index}) {:?}",
            std::thread::current().id()
        );

        // Claim the session's result receiver. The protocol allows only one fetch socket per
        // session, so a second fetcher is rejected here.
        let mut receiver = match self
            .service
            .take_response_receiver(session_id, device_index)
            .await
        {
            Ok(r) => r,
            Err(err) => {
                log::error!("{err}");
                return;
            }
        };

        log::debug!("Fetch writer running for session {session_id}");

        // Drain the per-session result queue. The queue closes when every sender is dropped: the
        // session's worker (on close/disconnect) and any in-flight readback tasks.
        while let Some(response) = receiver.recv().await {
            let bytes = match rmp_serde::to_vec(&response) {
                Ok(b) => b,
                Err(err) => {
                    log::error!(
                        "Failed to encode result for request {:?}: {err:?}",
                        response.id
                    );
                    continue;
                }
            };
            if let Err(err) = self.socket.send(Message::new(bytes.into())).await {
                log::warn!("Result send failed for session {session_id}: {err:?}; closing writer");
                return;
            }
        }

        log::debug!("Fetch writer for session {session_id} exited (queue closed)");
    }

    /// Read the init handshake and reply with the selected device's settings, returning the
    /// session this fetcher serves (or `None` if the handshake failed and the stream should be
    /// dropped).
    async fn handshake(&mut self) -> Option<(SessionId, u32)> {
        let msg = match self.socket.recv().await {
            Ok(Some(m)) => m,
            Ok(None) => {
                log::debug!("Fetch stream closed before init handshake");
                return None;
            }
            Err(err) => {
                log::warn!("Fetch stream error during init handshake: {err:?}");
                return None;
            }
        };

        let init = match parse_init_handshake(&msg.data) {
            Ok(init) => init,
            Err(err) => {
                log::error!("{err}; closing stream");
                return None;
            }
        };
        let session_id = init.session_id;
        let device_index = init.device_index;

        log::trace!("Init fetcher for session {session_id} (device {device_index})");

        // Reply with the selected device's default settings — the client uses these to fill in
        // `RemoteDevice::defaults` so it can resolve op dtypes without an extra RTT — and the
        // device count, so it can enumerate every device behind the address.
        let settings = self.service.device_settings(device_index);
        let device_count = self.service.device_count();
        let init_response = TaskResponse {
            content: TaskResponseContent::Init(SessionInfo {
                version: PROTOCOL_VERSION,
                settings,
                device_count,
                peer_id: None::<PeerId>,
            }),
            // Placeholder id for the handshake; the client reads this reply inline before the
            // fetch-demux task starts, so it never goes through the pending-callback map.
            id: 0,
        };
        let bytes = match rmp_serde::to_vec(&init_response) {
            Ok(b) => b,
            Err(err) => {
                log::error!("Failed to encode Init reply: {err:?}");
                return None;
            }
        };
        if let Err(err) = self.socket.send(Message::new(bytes.into())).await {
            log::error!("Failed to send Init reply for session {session_id}: {err:?}");
            return None;
        }

        Some((session_id, device_index))
    }
}
