use burn_backend::tensor::Device;
use burn_backend::{DeviceHandle, DeviceId};
use burn_communication::{Protocol, external_comm::ExternalCommService};
use burn_ir::BackendIr;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU16, Ordering};
use tokio::runtime::Handle;
use tokio::sync::{Mutex, mpsc};

use crate::server::local_comm::LocalCommService;
use crate::server::service::{
    BlockingForward, FetchSender, FetchService, SubmitService, TaskForwarder,
};
use crate::server::service_device::ServerDeviceService;
use crate::shared::{SessionId, Task, TaskResponse};

/// Per-server base for the synthetic device ids that key each device's runner.
///
/// The cubecl runner registry is process-global and keyed by [`DeviceId`]. We give every server
/// instance its own `type_id` so two servers in one process (e.g. the test suite) never share a
/// runner thread, and start high to stay clear of the small `type_id`s real backends use: our
/// service runs on the Upstream stage, already distinct from a backend's Downstream runners, and a
/// high base also avoids sharing a thread with a backend that is itself Upstream-staged (fusion).
static SERVER_TAG: AtomicU16 = AtomicU16::new(0x8000);

/// Coordinates per-session state across the server's devices.
///
/// Each device hosts one [`ServerDeviceService`] behind a [`DeviceHandle`]; every session pinned
/// to a device is an entry in that service's session map, with its own
/// [`TensorInterpreter`](burn_router::TensorInterpreter) (and its own
/// [`HandleContainer`](burn_ir::HandleContainer)), so different sessions never share tensor
/// handles. The manager itself only tracks, per session, which device it is pinned to and the
/// fetch receiver the fetch handler drains — the actual work runs on the device's runner thread
/// via the handle's optimized channel. Cross-session tensor transfers go through `external_comm`
/// (cross-server) or `local_comm` (same-host).
pub struct SessionManager<B, P>
where
    B: BackendIr,
    P: Protocol,
{
    /// All devices this server hosts, indexed by the device index the client selects at
    /// session init. `devices[0]` is the default device.
    devices: Vec<Device<B>>,
    /// One runner handle per device, parallel to `devices`.
    handles: Vec<DeviceHandle<ServerDeviceService<B, P>>>,
    /// Same-host transfer rendezvous, shared with every device service. Held here too so the
    /// submit forwarder can await a transfer's `take` *off* the runner thread.
    local_comm: Arc<LocalCommService<B>>,
    /// Cross-server transfer service, likewise held so the forwarder can await a download off the
    /// runner thread.
    external_comm: Arc<ExternalCommService<B, P>>,
    sessions: Mutex<HashMap<SessionId, SessionNet>>,
}

/// The manager-side state for a session: which device it lives on, and the fetch receiver the
/// fetch handler claims (the interpreter and [`FetchSender`] live in the device service).
struct SessionNet {
    device_index: u32,
    receiver: Option<mpsc::Receiver<TaskResponse>>,
}

impl<B, P> SessionManager<B, P>
where
    B: BackendIr,
    P: Protocol,
{
    /// Build the manager, registering one [`ServerDeviceService`] runner per device.
    ///
    /// Must be called from within the tokio runtime: each service captures [`Handle::current`] so
    /// its runner thread can drive the async parts of a task (transfers, readbacks, response
    /// sends).
    pub fn new(devices: Vec<Device<B>>, external_comm: Arc<ExternalCommService<B, P>>) -> Self {
        assert!(
            !devices.is_empty(),
            "A remote server must host at least one device"
        );

        let local_comm = Arc::new(LocalCommService::new());
        let runtime = Handle::current();
        let server_tag = SERVER_TAG.fetch_add(1, Ordering::Relaxed);

        let handles = devices
            .iter()
            .enumerate()
            .map(|(index, device)| {
                let service =
                    ServerDeviceService::new(device.clone(), runtime.clone(), external_comm.clone());
                let device_id = DeviceId::new(server_tag, index as u16);
                DeviceHandle::insert(device_id, service).unwrap_or_else(|err| {
                    panic!("Failed to register remote device runner {device_id}: {err:?}")
                })
            })
            .collect();

        Self {
            devices,
            handles,
            local_comm,
            external_comm,
            sessions: Mutex::new(HashMap::new()),
        }
    }

    /// Resolve the device at `device_index`.
    ///
    /// The index is validated against the server's device count on the client init handshake, so
    /// an out-of-range index here is a protocol/configuration error. Fail loudly rather than
    /// silently collapsing onto device 0 — for a collective that would reduce a device against
    /// itself and silently corrupt the result instead of producing a clear failure.
    pub(crate) fn device(&self, device_index: u32) -> Device<B> {
        self.devices
            .get(device_index as usize)
            .cloned()
            .unwrap_or_else(|| {
                panic!(
                    "Requested device index {device_index} but server hosts only {} device(s)",
                    self.devices.len()
                )
            })
    }

    /// Resolve the runner handle for `device_index`, with the same out-of-range guard as
    /// [`device`](Self::device).
    fn handle(&self, device_index: u32) -> &DeviceHandle<ServerDeviceService<B, P>> {
        self.handles.get(device_index as usize).unwrap_or_else(|| {
            panic!(
                "Requested device index {device_index} but server hosts only {} device(s)",
                self.handles.len()
            )
        })
    }

    /// Create the session on its device runner (if new) and register its fetch channel.
    ///
    /// Idempotent and safe to call from either handler: the first to touch a session creates it
    /// under the lock; a later caller sees it already present. The session is pinned to the device
    /// index fixed at this first touch.
    async fn ensure_session(&self, session_id: SessionId, device_index: u32) {
        let mut sessions = self.sessions.lock().await;
        if sessions.contains_key(&session_id) {
            return;
        }

        let (sender, receiver) = FetchSender::channel();
        // Enqueue the session's creation on the device runner before any task for it can run (tasks
        // go through the same FIFO channel). It rides the batch like any fire-and-forget op and
        // runs when the session's first sync point (a read/sync, or a transfer) flushes the batch —
        // which is necessarily before anything depends on the session's state.
        self.handle(device_index)
            .submit(move |svc| svc.create_session(session_id, sender));
        sessions.insert(
            session_id,
            SessionNet {
                device_index,
                receiver: Some(receiver),
            },
        );
    }
}

impl<B, P> SubmitService for SessionManager<B, P>
where
    B: BackendIr,
    P: Protocol,
{
    /// Resolve the forwarder used to push [`Task`]s onto `session_id`'s device runner, creating
    /// the session on demand. The request loop resolves this once per connection and reuses it for
    /// every task, instead of re-locking the sessions map per task.
    async fn session_forwarder(&self, session_id: SessionId, device_index: u32) -> TaskForwarder {
        self.ensure_session(session_id, device_index).await;
        let submit_handle = self.handle(device_index).clone();
        let blocking_handle = submit_handle.clone();
        let local_comm = self.local_comm.clone();
        let external_comm = self.external_comm.clone();
        TaskForwarder::new(
            // Fire-and-forget: enqueue on the runner's batched channel and move on.
            move |task: Task| {
                submit_handle.submit(move |svc| svc.process_task(session_id, task));
            },
            // Sync point: a transfer (whose rendezvous/download is awaited here, off the runner, so
            // the runner never blocks and can't deadlock under concurrent cross-device transfers),
            // or a result-producing/expose task run on the runner now — flushing the batch ahead.
            move |task: Task| {
                let handle = blocking_handle.clone();
                let local_comm = local_comm.clone();
                let external_comm = external_comm.clone();
                Box::pin(async move {
                    match task {
                        Task::RegisterTensorLocal {
                            stream_id,
                            transfer_id,
                            new_id,
                        } => {
                            // Wait for the source session (on another device's runner) to expose
                            // the primitive, then hand it to this runner for the device move.
                            let kind = local_comm.take(transfer_id).await;
                            handle.submit(move |svc| {
                                svc.register_local(session_id, stream_id, new_id, kind)
                            });
                        }
                        Task::RegisterTensorRemote(stream_id, remote, new_id) => {
                            log::trace!(
                                "Downloading remote tensor (transfer {:?} from {:?})",
                                remote.transfer_id,
                                remote.address,
                            );
                            match external_comm
                                .download_tensor(remote.address.clone(), remote.transfer_id)
                                .await
                            {
                                Some(data) => handle.submit(move |svc| {
                                    svc.register_remote(session_id, stream_id, new_id, data)
                                }),
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
                            // Grab the device-resident primitive on the runner (the only half that
                            // touches the device), then park it in the rendezvous off the runner so
                            // the registry insert doesn't block it.
                            match handle
                                .submit_blocking(move |svc| svc.get_tensor(session_id, stream_id, tensor))
                            {
                                Ok(Some(kind)) => {
                                    local_comm.expose(session_id, transfer_id, kind).await;
                                }
                                Ok(None) => log::error!(
                                    "ExposeTensorLocal for unknown session {session_id} dropped"
                                ),
                                Err(err) => log::error!(
                                    "get_tensor dispatch for session {session_id} failed: {err:?}"
                                ),
                            }
                        }
                        // A read/sync/dtype query or the cross-server expose: run it on the runner
                        // now (this flushes the fire-and-forget ops batched before it). These never
                        // block the runner (readbacks/hand-offs are spawned), so submit_blocking
                        // returns promptly.
                        other => {
                            if let Err(err) =
                                handle.submit_blocking(move |svc| svc.process_task(session_id, other))
                            {
                                log::error!(
                                    "Blocking task dispatch for session {session_id} failed: {err:?}"
                                );
                            }
                        }
                    }
                }) as BlockingForward
            },
        )
    }

    /// Drop the session: run its teardown on the device runner (after every task already forwarded
    /// for it), reclaim any transfers it left exposed, and remove the manager-side entry. The
    /// runner's `remove_session` flushes and drops the interpreter — releasing the session's backend
    /// state — and dropping its [`FetchSender`] closes the fetch writer's channel.
    async fn close(&self, session_id: SessionId) {
        let device_index = {
            let sessions = self.sessions.lock().await;
            sessions.get(&session_id).map(|net| net.device_index)
        };
        if let Some(device_index) = device_index {
            // Run teardown on the runner now: `submit_blocking` flushes every fire-and-forget op
            // still batched for the session and then runs `remove_session`, so nothing is dropped
            // with work outstanding and the teardown can't be left unflushed at disconnect.
            if let Err(err) = self
                .handle(device_index)
                .submit_blocking(move |svc| svc.remove_session(session_id))
            {
                log::error!("Failed to tear down session {session_id}: {err:?}");
            }
            // Reclaim any same-host transfers this session exposed that no target ever took. Done
            // here, off the runner, so teardown stays a non-blocking runner task.
            self.local_comm.purge_session(session_id).await;
        }
        self.sessions.lock().await.remove(&session_id);
    }
}

impl<B, P> FetchService for SessionManager<B, P>
where
    B: BackendIr,
    P: Protocol,
{
    /// The device settings for `device_index`, used by the response-init handshake before any
    /// session-specific runner is needed.
    fn device_settings(&self, device_index: u32) -> burn_std::DeviceSettings {
        use burn_backend::backend::DeviceOps;
        self.device(device_index).defaults()
    }

    /// The total number of devices this server hosts. Sent to the client on the init handshake
    /// so it can enumerate every device behind the address (see [`RemoteDevice::enumerate`]).
    fn device_count(&self) -> u32 {
        self.devices.len() as u32
    }

    /// Take the fetch receiver for `session_id`, creating the session on demand.
    ///
    /// Returns `Err` if a fetcher has already been registered for this session — the
    /// protocol allows only one fetch socket per session.
    async fn take_fetch_receiver(
        &self,
        session_id: SessionId,
        device_index: u32,
    ) -> Result<mpsc::Receiver<TaskResponse>, String> {
        self.ensure_session(session_id, device_index).await;
        let mut sessions = self.sessions.lock().await;
        let net = sessions
            .get_mut(&session_id)
            .expect("session was just ensured");
        net.receiver
            .take()
            .ok_or_else(|| format!("Fetch receiver already taken for session {session_id}"))
    }
}
