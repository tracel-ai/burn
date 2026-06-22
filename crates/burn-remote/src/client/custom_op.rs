//! Client-side helper for registering [custom operations](CustomOpIr) on the remote backend.
//!
//! A backend extension ships its op to the server as `OperationIr::Custom`, where a registered
//! [`CustomOpRegistry`](burn_router::CustomOpRegistry) handler executes it. How the op is registered
//! on the client differs by whether the `fusion` feature is enabled — the remote backend is a plain
//! [`BackendRouter`](burn_router::BackendRouter) without it and a
//! [`Fusion`](burn_fusion::Fusion)-wrapped one with it — and so does the tensor primitive it returns
//! (`RouterTensor` vs `FusionTensor`). [`CustomOpClient`] hides that difference: the same code builds
//! and registers a custom op, and gets back `FloatTensor<RemoteBackend>` either way.

use burn_backend::tensor::FloatTensor;
use burn_communication::Protocol;
use burn_ir::{CustomOpIr, OperationIr, TensorId};

use crate::client::RemoteChannel;
use crate::shared::RemoteProtocol;
use crate::{RemoteBackend, RemoteDevice};

/// The router channel used by the remote backend.
type Channel = RemoteChannel<<RemoteProtocol as Protocol>::Client>;

/// A client for registering [custom operations](CustomOpIr) on the remote backend, transparently
/// across the `fusion` feature.
///
/// Drop-in for the lower-level client a backend extension would otherwise reach for: allocate output
/// ids with [`create_empty_handle`](Self::create_empty_handle), build a [`CustomOpIr`], then
/// [`register`](Self::register) it. With `fusion` enabled the op joins the cached op-graph (via the
/// fusion client); without it the op streams through the router client. Either way the op reaches the
/// server, and the returned tensors are the matching `FloatTensor<RemoteBackend>`.
pub struct CustomOpClient {
    #[cfg(not(feature = "fusion"))]
    inner: <Channel as burn_router::RouterChannel>::Client,
    #[cfg(feature = "fusion")]
    inner: burn_fusion::client::GlobalFusionClient<burn_router::RouterFusionRuntime<Channel>>,
}

impl CustomOpClient {
    /// Create a client bound to the given remote device.
    pub fn new(device: &RemoteDevice) -> Self {
        #[cfg(not(feature = "fusion"))]
        let inner = burn_router::get_client::<Channel>(device);
        #[cfg(feature = "fusion")]
        let inner = burn_fusion::get_client::<burn_router::BackendRouter<Channel>>(device);

        Self { inner }
    }

    /// Allocate a fresh, uninitialized tensor id for a custom op output.
    ///
    /// Use it to build the output [`TensorIr`](burn_ir::TensorIr)s of the [`CustomOpIr`] passed to
    /// [`register`](Self::register). The id is allocated by the same client that registers the op, so
    /// it is consistent with how that client tracks tensors (fusion ids under `fusion`, router ids
    /// otherwise).
    pub fn create_empty_handle(&self) -> TensorId {
        #[cfg(not(feature = "fusion"))]
        {
            use burn_router::RouterClient;
            self.inner.create_empty_handle()
        }
        #[cfg(feature = "fusion")]
        {
            self.inner.create_empty_handle()
        }
    }

    /// Register a custom op and return its (uninitialized) output tensors.
    pub fn register(&self, op: CustomOpIr) -> Vec<FloatTensor<RemoteBackend>> {
        #[cfg(not(feature = "fusion"))]
        {
            use burn_router::RouterClient;
            self.inner.register(OperationIr::Custom(op))
        }
        #[cfg(feature = "fusion")]
        {
            use burn_fusion::{NoOp, stream::StreamId};
            // The op executes on the server through the `CustomOpRegistry`; the fusion layer only
            // ships the op-graph (the relativized `OperationIr`, scalars included), so the local
            // fusion operation is a genuine no-op.
            self.inner.register(
                StreamId::current(),
                OperationIr::Custom(op),
                NoOp::<burn_router::BackendRouter<Channel>>::new(),
            )
        }
    }
}
