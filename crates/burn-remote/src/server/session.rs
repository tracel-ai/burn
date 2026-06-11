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
use crate::server::service::{FetchSender, FetchService, SubmitService, TaskForwarder};
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
                let service = ServerDeviceService::new(
                    device.clone(),
                    runtime.clone(),
                    external_comm.clone(),
                    local_comm.clone(),
                );
                let device_id = DeviceId::new(server_tag, index as u16);
                DeviceHandle::insert(device_id, service).unwrap_or_else(|err| {
                    panic!("Failed to register remote device runner {device_id}: {err:?}")
                })
            })
            .collect();

        Self {
            devices,
            handles,
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
        // Enqueue the session's creation on the device runner before any task for it can run
        // (tasks go through the same FIFO channel).
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
        let handle = self.handle(device_index).clone();
        Box::new(move |task: Task| {
            handle.submit(move |svc| svc.process_task(session_id, task));
        })
    }

    /// Drop the session: enqueue its teardown on the device runner (after every task already
    /// forwarded for it) and remove the manager-side entry. The runner's `remove_session` flushes
    /// and drops the interpreter — releasing the session's backend state — and dropping its
    /// [`FetchSender`] closes the fetch writer's channel.
    async fn close(&self, session_id: SessionId) {
        let device_index = {
            let sessions = self.sessions.lock().await;
            sessions.get(&session_id).map(|net| net.device_index)
        };
        if let Some(device_index) = device_index {
            self.handle(device_index)
                .submit(move |svc| svc.remove_session(session_id));
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
