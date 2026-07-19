//! The tensor-transfer seam between the session worker and a transport.
//!
//! Server-to-server tensor movement is independent from the compute-session transport: a tensor is
//! *exposed* on the source server and *downloaded* by the target server without routing the data
//! through the controlling client. Each transport implements this its own way —
//! [`IrohTransfer`](crate::transport::iroh::IrohTransfer) over authenticated Iroh streams,
//! [`WebSocketTransfer`](crate::transport::websocket::WebSocketTransfer) over the legacy data
//! service — and the session worker drives it through this trait, knowing nothing about the wire.

use std::future::Future;

use burn_backend::TensorData;
use burn_ir::BackendIr;

use crate::shared::TransferCapability;
use crate::{PeerAddr, PeerId};

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
