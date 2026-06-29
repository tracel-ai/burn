//! Legacy WebSocket tensor transfer, carried over the `burn_communication` data service.

use std::sync::Arc;

use burn_backend::TensorData;
use burn_ir::BackendIr;

use crate::server::transfer::TensorTransfer;
use crate::shared::TransferCapability;
use crate::{PeerAddr, PeerId};

pub(crate) struct WebSocketTransfer<B: BackendIr> {
    pub(crate) inner: Arc<
        burn_communication::external_comm::ExternalCommService<
            B,
            burn_communication::websocket::WebSocket,
        >,
    >,
}

impl<B: BackendIr> Clone for WebSocketTransfer<B> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

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
        // Only the WebSocket arm remains when the Iroh transport is compiled out.
        #[cfg_attr(
            not(feature = "iroh"),
            allow(clippy::infallible_destructuring_match)
        )]
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

fn capability_to_legacy_id(
    capability: TransferCapability,
) -> burn_communication::external_comm::TensorTransferId {
    // WebSocket is a compatibility transport without authenticated peer identity. Preserve its old
    // transfer service while deriving a collision-resistant-enough rendezvous key from the
    // capability. Iroh uses the complete capability and enforces the destination identity.
    capability.legacy_id().into()
}
