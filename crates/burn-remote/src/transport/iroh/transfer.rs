//! Authenticated tensor transfer over independent Iroh streams.
//!
//! Server-to-server tensor movement rides its own bidirectional stream
//! ([`StreamKind::TensorTransfer`]), separate from the compute session: a tensor is exposed under a
//! [`TransferCapability`] for a specific target peer, and only that peer can download it.

use std::collections::HashMap;
use std::sync::Arc;

use burn_backend::TensorData;
use burn_ir::BackendIr;
use tokio::sync::{Mutex, Notify};

use super::node::{RemoteNode, StreamKind};
use crate::{
    PeerAddr, PeerId,
    server::transfer::TensorTransfer,
    shared::{Encode, Encoded, TransferCapability},
    transport::{
        link::{FrameSink, FrameSource, MAX_UNAUTHORIZED_FRAME_SIZE},
        message::{MessageSink, MessageSource},
    },
};

#[derive(Debug, serde::Serialize, serde::Deserialize)]
enum TransferMessage {
    Request(TransferCapability),
    Tensor(TensorData),
    Denied(String),
}

impl Encode for TransferMessage {}

struct ExposedTensor {
    message: Encoded,
    target: iroh::EndpointId,
    downloads: u32,
    max_downloads: u32,
}

/// Authenticated tensor transfer service carried on independent Iroh streams.
pub(crate) struct IrohTransfer<B: BackendIr> {
    node: RemoteNode,
    exposed: Arc<Mutex<HashMap<TransferCapability, ExposedTensor>>>,
    exposed_notify: Notify,
    _backend: core::marker::PhantomData<B>,
}

impl<B: BackendIr> IrohTransfer<B> {
    pub(crate) fn new(node: RemoteNode) -> Self {
        Self {
            node,
            exposed: Arc::new(Mutex::new(HashMap::new())),
            exposed_notify: Notify::new(),
            _backend: core::marker::PhantomData,
        }
    }

    pub(crate) async fn handle_stream(
        &self,
        remote: iroh::EndpointId,
        send: iroh::endpoint::SendStream,
        mut recv: iroh::endpoint::RecvStream,
    ) -> Result<(), String> {
        // One bare frame, since it is read before the capability authorizes the peer.
        let request = FrameSource::recv(&mut recv, MAX_UNAUTHORIZED_FRAME_SIZE)
            .await?
            .ok_or_else(|| "Tensor-transfer stream closed before its request".to_string())?;
        let TransferMessage::Request(capability) = rmp_serde::from_slice(&request)
            .map_err(|err| format!("Invalid tensor-transfer request: {err}"))?
        else {
            return Err("Expected a tensor-transfer request".into());
        };

        let response = match self.take(capability, remote).await {
            Ok(message) => message,
            Err(reason) => TransferMessage::Denied(reason)
                .encode()
                .map_err(|err| format!("Failed to encode tensor-transfer denial: {err}"))?,
        };
        let mut response_sink = MessageSink::new(send);
        response_sink.send(response).await?;
        response_sink.close().await
    }

    async fn take(
        &self,
        capability: TransferCapability,
        remote: iroh::EndpointId,
    ) -> Result<Encoded, String> {
        crate::time::timeout(TRANSFER_WAIT_TIMEOUT, async {
            loop {
                let notified = self.exposed_notify.notified();
                tokio::pin!(notified);
                notified.as_mut().enable();
                {
                    let mut exposed = self.exposed.lock().await;
                    if let Some(mut tensor) = exposed.remove(&capability) {
                        if tensor.target != remote {
                            exposed.insert(capability, tensor);
                            return Err(format!(
                                "Transfer capability is not authorized for peer {remote}"
                            ));
                        }
                        tensor.downloads += 1;
                        let message = tensor.message.clone();
                        if tensor.downloads < tensor.max_downloads {
                            exposed.insert(capability, tensor);
                        }
                        return Ok(message);
                    }
                }
                notified.as_mut().await;
            }
        })
        .await
        .map_err(|_| format!("Timed out waiting for tensor transfer {capability:?}"))?
    }

    async fn expose_response(
        &self,
        message: Encoded,
        max_downloads: u32,
        capability: TransferCapability,
        target: iroh::EndpointId,
    ) {
        self.exposed.lock().await.insert(
            capability,
            ExposedTensor {
                message,
                target,
                downloads: 0,
                max_downloads,
            },
        );
        self.exposed_notify.notify_waiters();

        let exposed = self.exposed.clone();
        crate::server::spawn::spawn_detached(async move {
            crate::time::sleep(TRANSFER_CAPABILITY_TTL).await;
            exposed.lock().await.remove(&capability);
        });
    }
}

const TRANSFER_WAIT_TIMEOUT: core::time::Duration = core::time::Duration::from_secs(300);
const TRANSFER_CAPABILITY_TTL: core::time::Duration = core::time::Duration::from_secs(300);

impl<B: BackendIr> TensorTransfer<B> for IrohTransfer<B> {
    async fn expose_data(
        &self,
        data: TensorData,
        max_downloads: u32,
        capability: TransferCapability,
        target: PeerId,
    ) {
        let Some(target) = target.into_iroh_id() else {
            log::error!("An Iroh tensor transfer cannot target a non-Iroh peer");
            return;
        };
        let message = match TransferMessage::Tensor(data).encode() {
            Ok(message) => message,
            Err(err) => {
                log::error!("Failed to encode tensor transfer {capability:?}: {err}");
                return;
            }
        };
        self.expose_response(message, max_downloads, capability, target)
            .await;
    }

    async fn download_tensor(
        &self,
        remote: PeerAddr,
        capability: TransferCapability,
    ) -> Option<TensorData> {
        match &remote {
            PeerAddr::Iroh(_) => {}
            #[cfg(feature = "websocket")]
            PeerAddr::WebSocket(_) => {
                log::error!("An Iroh compute node cannot download from a non-Iroh peer");
                return None;
            }
        }
        let (mut send, recv) = match self
            .node
            .open_stream(&remote, StreamKind::TensorTransfer)
            .await
        {
            Ok(streams) => streams,
            Err(err) => {
                log::error!("Cannot open a tensor-transfer stream to {remote}: {err}");
                return None;
            }
        };
        let request = match TransferMessage::Request(capability).encode() {
            Ok(request) => request,
            Err(err) => {
                log::error!("Failed to encode tensor-transfer request: {err}");
                return None;
            }
        };
        if let Err(err) = FrameSink::send(&mut send, request.into_bytes()).await {
            log::error!("{err}");
            return None;
        }
        let _ = send.finish();
        let response = match MessageSource::new(recv).recv().await {
            Ok(Some(response)) => response,
            Ok(None) => {
                log::error!("Tensor-transfer peer closed without a response");
                return None;
            }
            Err(err) => {
                log::error!("{err}");
                return None;
            }
        };
        match rmp_serde::from_slice(&response) {
            Ok(TransferMessage::Tensor(data)) => Some(data),
            Ok(TransferMessage::Denied(reason)) => {
                log::error!("Tensor transfer denied: {reason}");
                None
            }
            Ok(TransferMessage::Request(_)) => {
                log::error!("Tensor-transfer peer returned a request instead of tensor data");
                None
            }
            Err(err) => {
                log::error!("Invalid tensor-transfer response: {err}");
                None
            }
        }
    }

    async fn fail(&self, capability: TransferCapability, target: PeerId, reason: String) {
        let Some(target) = target.into_iroh_id() else {
            return;
        };
        let message = match TransferMessage::Denied(reason).encode() {
            Ok(message) => message,
            Err(err) => {
                log::error!("Failed to encode tensor-transfer failure: {err}");
                return;
            }
        };
        self.expose_response(message, 1, capability, target).await;
    }
}
