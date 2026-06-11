//! Per-device service: one [`DeviceService`] per physical device, owning every session's
//! interpreter and running its tasks on the device's cubecl runner thread.
//!
//! This replaces the old one-OS-thread-per-session worker. Instead of spawning a thread per
//! session, every session targeting a device is a [`SessionState`] entry in a single
//! [`ServerDeviceService`], and tasks are dispatched onto the device's runner thread through the
//! optimized [`DeviceHandle`] channel (the same fire-and-forget channel the client already rides).
//! The service owns the session map by `&mut self` on the runner thread, so the map needs no
//! locking and the service is the natural home for any future cross-session scheduling/priority.
//!
//! ## Why [`DeviceServiceStage::Upstream`]
//!
//! The server *interprets* tasks (decode op IR, look up handles, hand them to the inner backend)
//! ahead of the inner backend's own kernel dispatch. Running on the Upstream stage gives the
//! server its own runner thread per device, distinct from the backend's Downstream thread, so
//! interpretation overlaps with kernel dispatch.
//!
//! ## Blocking tasks
//!
//! Some tasks block the runner thread synchronously: a collective op (all-reduce) parks inside
//! `register_op` until every device reaches the barrier, a same-host transfer's
//! [`RegisterTensorLocal`](Task::RegisterTensorLocal) blocks on the rendezvous, and a cross-server
//! [`RegisterTensorRemote`](Task::RegisterTensorRemote) blocks on the download. With one runner
//! thread per device this is correct for the common one-session-per-device configuration and for
//! all same-host transfers (source and target always live on *different* devices, hence different
//! runner threads). Multiple sessions on the *same* device sharing a runner is best-effort: a
//! blocking task on one stalls the others until it unblocks. Detached readbacks (`ReadTensor`,
//! `ExposeTensorRemote`) and fetch-channel sends are spawned onto the runtime so they never block
//! the runner.

use std::collections::HashMap;
use std::sync::Arc;

use burn_backend::tensor::Device;
use burn_backend::{DeviceId, DeviceService, DeviceServiceStage, ServerUtilitiesHandle};
use burn_communication::{Protocol, external_comm::ExternalCommService};
use burn_ir::BackendIr;
use burn_router::{RouterClient, TensorInterpreter};
use tokio::runtime::Handle;

use crate::server::local_comm::LocalCommService;
use crate::server::service::FetchSender;
use crate::shared::{SessionId, Task, TaskResponseContent};

/// Everything constant for the lifetime of one session on a device.
struct SessionState<B: BackendIr> {
    interpreter: TensorInterpreter<B>,
    fetch_sender: FetchSender,
}

/// One service per physical device. Owns the interpreters of every session pinned to that device
/// and runs their tasks on the device's runner thread.
///
/// Always constructed on a runtime thread and inserted into a [`DeviceHandle`] via
/// [`DeviceHandle::insert`](burn_backend::DeviceHandle::insert) — never via [`DeviceService::init`]
/// — because it needs the tokio [`Handle`] and the comm services, which `init` can't provide.
pub(crate) struct ServerDeviceService<B: BackendIr, P: Protocol> {
    device: Device<B>,
    /// Runtime used to drive the async parts of a task: `block_on` for the blocking transfers and
    /// `spawn` for detached readbacks and fetch-channel sends.
    runtime: Handle,
    external_comm: Arc<ExternalCommService<B, P>>,
    local_comm: Arc<LocalCommService<B>>,
    sessions: HashMap<SessionId, SessionState<B>>,
}

impl<B: BackendIr, P: Protocol> ServerDeviceService<B, P> {
    pub(crate) fn new(
        device: Device<B>,
        runtime: Handle,
        external_comm: Arc<ExternalCommService<B, P>>,
        local_comm: Arc<LocalCommService<B>>,
    ) -> Self {
        Self {
            device,
            runtime,
            external_comm,
            local_comm,
            sessions: HashMap::new(),
        }
    }

    /// Create a session's interpreter, registering its fetch channel. Runs on the runner thread,
    /// ordered before any task forwarded for the session.
    pub(crate) fn create_session(&mut self, session_id: SessionId, fetch_sender: FetchSender) {
        let runner = TensorInterpreter::new(self.device.clone());
        self.sessions.insert(
            session_id,
            SessionState {
                interpreter: runner,
                fetch_sender,
            },
        );
    }

    /// Tear a session down: flush and drop its interpreter (freeing its tensors), hand freed memory
    /// back to the allocator, and reclaim any same-host transfers it left exposed.
    ///
    /// Mirrors the old per-session worker's close path: dropping `SessionState` drops the session's
    /// [`FetchSender`], which closes the fetch writer's channel and ends it.
    pub(crate) fn remove_session(&mut self, session_id: SessionId) {
        if let Some(SessionState {
            interpreter: runner,
            ..
        }) = self.sessions.remove(&session_id)
        {
            let device = runner.device();

            // Flush outstanding backend work before dropping the runner so the session's tensors
            // aren't freed with work still queued against them.
            if let Err(err) = runner.sync() {
                log::warn!("runner.sync() at session {session_id} close failed: {err:?}");
            }
            if let Err(err) = B::sync(&device) {
                log::warn!("B::sync(device) at session {session_id} close failed: {err:?}");
            }

            // Drop the runner: frees every tensor handle back to the backend allocator. Must happen
            // before `memory_cleanup`, otherwise the memory is still live and nothing is reclaimable.
            drop(runner);
            B::memory_cleanup(&device);
        }

        // Reclaim any same-host transfers this session exposed that no target ever took.
        let local_comm = self.local_comm.clone();
        self.runtime
            .block_on(async move { local_comm.purge_session(session_id).await });
    }

    /// Execute a single [`Task`] against `session_id`'s state.
    ///
    /// Runs on the device's runner thread. Sync work is wrapped in
    /// [`StreamId::executes`](burn_std::id::StreamId::executes) so the runner's thread-local stream
    /// id matches the one the client assigned to the op. Blocking transfers are driven with
    /// `block_on`; detached readbacks and fetch-channel sends are spawned onto the runtime.
    pub(crate) fn process_task(&mut self, session_id: SessionId, task: Task) {
        // Clone the shared (cheap `Arc` / `Handle`) handles up front so the per-session `&mut`
        // borrow below doesn't conflict with the service-wide fields.
        let external_comm = self.external_comm.clone();
        let local_comm = self.local_comm.clone();
        let runtime = self.runtime.clone();

        let Some(state) = self.sessions.get_mut(&session_id) else {
            log::error!("Task for unknown session {session_id} dropped");
            return;
        };
        let runner = &state.interpreter;
        let fetch_sender = &state.fetch_sender;

        match task {
            Task::RegisterOperation(stream_id, op) => {
                stream_id.executes(|| runner.register_op(op));
            }
            Task::RegisterTensor(stream_id, id, data) => {
                stream_id.executes(|| runner.register_tensor_data_id(id, data));
            }
            Task::RegisterTensorRemote(stream_id, remote, new_id) => {
                log::trace!(
                    "Registering remote tensor (transfer {:?} from {:?})",
                    remote.transfer_id,
                    remote.address,
                );
                let data = runtime.block_on(async {
                    external_comm
                        .download_tensor(remote.address.clone(), remote.transfer_id)
                        .await
                });
                match data {
                    Some(data) => {
                        // Register on the client stream that will consume `new_id`, carried over
                        // the wire — not the runner thread's ambient stream.
                        stream_id.executes(|| runner.register_tensor_data_id(new_id, data));
                    }
                    None => log::error!(
                        "Failed to download tensor for transfer {:?} from {:?}",
                        remote.transfer_id,
                        remote.address,
                    ),
                }
            }
            Task::ExposeTensorLocal {
                stream_id,
                tensor,
                transfer_id,
            } => {
                // Source side of a same-host transfer: grab the device-resident primitive (no host
                // readback) and park it for the target session. Ordered after the op that produced
                // `tensor`, so the handle is present.
                let kind = stream_id.executes(|| runner.get_tensor(&tensor));
                runtime.block_on(async {
                    local_comm.expose(session_id, transfer_id, kind).await;
                });
            }
            Task::RegisterTensorLocal {
                stream_id,
                transfer_id,
                new_id,
            } => {
                // Target side of a same-host transfer: wait for the source (on its own device's
                // runner thread) to expose, then move the primitive onto this device and register
                // it. Blocks this device's runner until the source exposes.
                let kind = runtime.block_on(async { local_comm.take(transfer_id).await });
                stream_id.executes(|| runner.register_tensor_to_device(new_id, kind));
            }
            Task::ExposeTensorRemote {
                stream_id,
                tensor,
                count,
                transfer_id,
            } => {
                log::trace!("Exposing tensor (transfer {transfer_id:?})");
                // Like `ReadTensor`: the sync part of `read_tensor_async` runs in order to preserve
                // stream ordering, but the readback + expose are detached so a cross-server hand-off
                // doesn't stall the runner on a GPU→host copy.
                let fut = stream_id.executes(|| runner.read_tensor_async(tensor));
                runtime.spawn(async move {
                    match fut.await {
                        Ok(data) => external_comm.expose_data(data, count, transfer_id).await,
                        Err(e) => log::error!(
                            "read_tensor_async for transfer {transfer_id:?} failed: {e:?}"
                        ),
                    }
                });
            }
            Task::Seed(seed) => runner.seed(seed),
            Task::ReadTensor(request_id, stream_id, tensor) => {
                // `read_tensor_async` is sync at construction (locks the context, captures the
                // tensor's position in the command stream) and returns a future for the host
                // readback. Run the sync part in order, then detach the readback so awaiting it
                // doesn't stall the runner on the GPU→host copy. The client demuxes responses by
                // request id, so out-of-order completion is fine.
                let fut = stream_id.executes(|| runner.read_tensor_async(tensor));
                let sender = fetch_sender.clone();
                runtime.spawn(async move {
                    let data = fut.await;
                    sender
                        .send(request_id, TaskResponseContent::ReadTensor(data))
                        .await;
                });
            }
            Task::SyncBackend(request_id, stream_id) => {
                let res = stream_id.executes(|| runner.sync());
                let sender = fetch_sender.clone();
                runtime.spawn(async move {
                    sender
                        .send(request_id, TaskResponseContent::SyncBackend(res))
                        .await;
                });
            }
            Task::DTypeUsage(request_id, dtype) => {
                let res = runner.dtype_usage(dtype);
                let sender = fetch_sender.clone();
                runtime.spawn(async move {
                    sender
                        .send(request_id, TaskResponseContent::DTypeUsage(res))
                        .await;
                });
            }
        }
    }
}

impl<B: BackendIr, P: Protocol> DeviceService for ServerDeviceService<B, P> {
    fn init(_device_id: DeviceId) -> Self {
        // We always build the service with `DeviceHandle::insert` so it can carry the tokio handle
        // and comm services; `init` (which only gets a `DeviceId`) is never used.
        unreachable!("ServerDeviceService is constructed via DeviceHandle::insert, not init");
    }

    fn utilities(&self) -> ServerUtilitiesHandle {
        Arc::new(())
    }

    fn stage() -> DeviceServiceStage {
        DeviceServiceStage::Upstream
    }
}
