use std::future::Future;
#[cfg(any(feature = "websocket", feature = "iroh"))]
use std::sync::Arc;

use burn_backend::TensorData;
use burn_ir::BackendIr;

use crate::{PeerAddr, PeerId, shared::TransferCapability};

#[cfg(feature = "iroh")]
use std::collections::HashMap;
#[cfg(feature = "iroh")]
use tokio::sync::{Mutex, Notify};

/// Server-to-server tensor movement, independent from the compute-session transport.
pub(crate) trait TensorTransfer<B: BackendIr>: Send + Sync + 'static {
    fn expose_data(
        &self,
        data: TensorData,
        max_downloads: u32,
        capability: TransferCapability,
        target: PeerId,
    ) -> impl Future<Output = ()> + Send;

    fn download_tensor(
        &self,
        remote: PeerAddr,
        capability: TransferCapability,
    ) -> impl Future<Output = Option<TensorData>> + Send;

    fn fail(
        &self,
        capability: TransferCapability,
        target: PeerId,
        reason: String,
    ) -> impl Future<Output = ()> + Send;
}

#[cfg(feature = "websocket")]
pub(crate) struct WebSocketTransfer<B: BackendIr> {
    pub(crate) inner: Arc<
        burn_communication::external_comm::ExternalCommService<
            B,
            burn_communication::websocket::WebSocket,
        >,
    >,
}

#[cfg(feature = "websocket")]
impl<B: BackendIr> Clone for WebSocketTransfer<B> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

#[cfg(feature = "websocket")]
impl<B: BackendIr> TensorTransfer<B> for WebSocketTransfer<B> {
    async fn expose_data(
        &self,
        data: TensorData,
        max_downloads: u32,
        capability: TransferCapability,
        _target: PeerId,
    ) {
        self.inner
            .expose_data(data, max_downloads, capability_to_legacy_id(capability))
            .await;
    }

    async fn download_tensor(
        &self,
        remote: PeerAddr,
        capability: TransferCapability,
    ) -> Option<TensorData> {
        let address = match remote {
            PeerAddr::WebSocket(address) => address,
            #[cfg(feature = "iroh")]
            PeerAddr::Iroh(_) => {
                log::error!("A WebSocket compute node cannot download from a non-WebSocket peer");
                return None;
            }
        };
        self.inner
            .download_tensor(address, capability_to_legacy_id(capability))
            .await
    }

    async fn fail(&self, _capability: TransferCapability, _target: PeerId, reason: String) {
        log::error!("Legacy WebSocket tensor transfer failed before exposure: {reason}");
    }
}

#[cfg(feature = "websocket")]
fn capability_to_legacy_id(
    capability: TransferCapability,
) -> burn_communication::external_comm::TensorTransferId {
    // WebSocket is a compatibility transport without authenticated peer identity. Preserve its
    // old transfer service while deriving a collision-resistant-enough rendezvous key from the
    // capability. Iroh uses the complete capability and enforces the destination identity.
    capability.legacy_id().into()
}

#[cfg(feature = "iroh")]
#[derive(Debug, serde::Serialize, serde::Deserialize)]
enum TransferMessage {
    Request(TransferCapability),
    Tensor(TensorData),
    Denied(String),
}

#[cfg(feature = "iroh")]
struct ExposedTensor {
    bytes: bytes::Bytes,
    target: iroh::EndpointId,
    downloads: u32,
    max_downloads: u32,
}

/// Authenticated tensor transfer service carried on independent Iroh streams.
#[cfg(feature = "iroh")]
pub(crate) struct IrohTransfer<B: BackendIr> {
    node: crate::RemoteNode,
    exposed: Arc<Mutex<HashMap<TransferCapability, ExposedTensor>>>,
    exposed_notify: Notify,
    _backend: core::marker::PhantomData<B>,
}

#[cfg(feature = "iroh")]
impl<B: BackendIr> IrohTransfer<B> {
    pub(crate) fn new(node: crate::RemoteNode) -> Self {
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
        mut send: iroh::endpoint::SendStream,
        mut recv: iroh::endpoint::RecvStream,
    ) -> Result<(), String> {
        let request = crate::node::recv_frame(&mut recv)
            .await?
            .ok_or_else(|| "Tensor-transfer stream closed before its request".to_string())?;
        let TransferMessage::Request(capability) = rmp_serde::from_slice(&request)
            .map_err(|err| format!("Invalid tensor-transfer request: {err}"))?
        else {
            return Err("Expected a tensor-transfer request".into());
        };

        let response = match self.take(capability, remote).await {
            Ok(bytes) => bytes,
            Err(reason) => rmp_serde::to_vec(&TransferMessage::Denied(reason))
                .map(bytes::Bytes::from)
                .map_err(|err| format!("Failed to encode tensor-transfer denial: {err}"))?,
        };
        crate::node::send_frame(&mut send, &response).await?;
        send.finish()
            .map_err(|err| format!("Failed to finish tensor-transfer stream: {err}"))?;
        Ok(())
    }

    async fn take(
        &self,
        capability: TransferCapability,
        remote: iroh::EndpointId,
    ) -> Result<bytes::Bytes, String> {
        crate::server::time::timeout(TRANSFER_WAIT_TIMEOUT, async {
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
                        let bytes = tensor.bytes.clone();
                        if tensor.downloads < tensor.max_downloads {
                            exposed.insert(capability, tensor);
                        }
                        return Ok(bytes);
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
        bytes: bytes::Bytes,
        max_downloads: u32,
        capability: TransferCapability,
        target: iroh::EndpointId,
    ) {
        self.exposed.lock().await.insert(
            capability,
            ExposedTensor {
                bytes,
                target,
                downloads: 0,
                max_downloads,
            },
        );
        self.exposed_notify.notify_waiters();

        let exposed = self.exposed.clone();
        crate::server::spawn::spawn_detached(async move {
            crate::server::time::sleep(TRANSFER_CAPABILITY_TTL).await;
            exposed.lock().await.remove(&capability);
        });
    }
}

#[cfg(feature = "iroh")]
const TRANSFER_WAIT_TIMEOUT: core::time::Duration = core::time::Duration::from_secs(300);
#[cfg(feature = "iroh")]
const TRANSFER_CAPABILITY_TTL: core::time::Duration = core::time::Duration::from_secs(300);

#[cfg(feature = "iroh")]
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
        let bytes = match rmp_serde::to_vec(&TransferMessage::Tensor(data)) {
            Ok(bytes) => bytes::Bytes::from(bytes),
            Err(err) => {
                log::error!("Failed to encode tensor transfer {capability:?}: {err}");
                return;
            }
        };
        self.expose_response(bytes, max_downloads, capability, target)
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
        let (mut send, mut recv) = match self
            .node
            .open_stream(&remote, crate::node::StreamKind::TensorTransfer)
            .await
        {
            Ok(streams) => streams,
            Err(err) => {
                log::error!("{err}");
                return None;
            }
        };
        let request = match rmp_serde::to_vec(&TransferMessage::Request(capability)) {
            Ok(request) => request,
            Err(err) => {
                log::error!("Failed to encode tensor-transfer request: {err}");
                return None;
            }
        };
        if let Err(err) = crate::node::send_frame(&mut send, &request).await {
            log::error!("{err}");
            return None;
        }
        let _ = send.finish();
        let response = match crate::node::recv_frame(&mut recv).await {
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
        let bytes = match rmp_serde::to_vec(&TransferMessage::Denied(reason)) {
            Ok(bytes) => bytes.into(),
            Err(err) => {
                log::error!("Failed to encode tensor-transfer failure: {err}");
                return;
            }
        };
        self.expose_response(bytes, 1, capability, target).await;
    }
}
