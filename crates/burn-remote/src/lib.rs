//! Peer-to-peer remote tensor execution for Burn.
//!
//! Iroh is the primary transport. Applications own an Iroh [`Endpoint`] and build remote devices
//! from it; a server hosts compute on its own endpoint. Compute sessions use bidirectional QUIC
//! streams, while cross-peer tensor movement uses independent authenticated streams without
//! routing payloads through the controlling client.
//!
//! The optional `websocket` feature retains the legacy address-and-port transport.

#[cfg(feature = "client")]
mod client;

#[cfg(feature = "server")]
pub mod server;

pub(crate) mod shared;
pub mod telemetry;
#[cfg(any(feature = "client", all(feature = "server", feature = "iroh")))]
pub(crate) mod time;
mod transport;

pub use burn_ir as ir;
pub use burn_router::RouterClient;

/// Network-traffic savings metric for op-graph caching, shared by the client device service and the
/// server session worker.
#[cfg(any(feature = "client", feature = "server"))]
pub(crate) mod metrics;

#[cfg(feature = "iroh")]
pub use iroh::{Endpoint, EndpointAddr, EndpointId};
#[cfg(feature = "iroh")]
pub use transport::iroh::RemoteSecret;
#[cfg(feature = "iroh")]
pub use transport::iroh::node::BURN_REMOTE_ALPN;
pub use transport::{PeerAddr, PeerId};

#[cfg(feature = "client")]
mod __client {
    use super::*;

    use burn_router::BackendRouter;

    /// The remote backend allows you to run computation on a remote device.
    ///
    /// Iroh is the primary transport. Applications own an Iroh [`Endpoint`], resolve a compute peer
    /// through their own discovery/control plane, and construct devices from the endpoint and the
    /// peer's address with [`RemoteDevice::iroh`] (or the `Device::remote_iroh` facade).
    ///
    /// ```rust, ignore
    /// let endpoint = Endpoint::builder(presets::N0).bind().await?;
    /// let remote = RemoteDevice::iroh(&endpoint, compute_peer, 0);
    /// ```
    #[cfg(not(feature = "fusion"))]
    pub type RemoteBackend = BackendRouter<RemoteChannel>;

    /// With the `fusion` feature enabled, the remote backend is wrapped in
    /// [`Fusion`](burn_fusion::Fusion) — exactly like the CubeCL backends — so recurring groups of
    /// operations are cached on the server and invoked by id, sending a repeated computation
    /// (e.g. a model block per step) over the network once instead of every step.
    #[cfg(feature = "fusion")]
    pub type RemoteBackend = burn_fusion::Fusion<BackendRouter<RemoteChannel>>;

    pub use client::{CustomOpClient, RemoteChannel, RemoteDevice};
}
#[cfg(feature = "client")]
pub use __client::*;

// No lib test may name burn_tensor: its burn-remote copy's device ids collide with this one's.
#[cfg(all(test, feature = "client", feature = "server"))]
mod tests {
    use crate::{
        RemoteBackend, RemoteDevice,
        shared::{RemoteMessage, SessionId, Task},
    };
    use burn_backend::{Scalar, TensorData, ops::FloatTensorOps};
    use burn_communication::{CommunicationChannel, Message, ProtocolClient};
    use burn_flex::Flex;
    use std::str::FromStr;

    /// Serve `server` over WebSocket on a port the OS picks, returning the address to dial.
    ///
    /// The listener is bound before this returns, so a client can connect at once.
    pub(crate) fn serve(
        rt: &tokio::runtime::Runtime,
        server: crate::server::RemoteServerBuilder<Flex>,
    ) -> String {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let address = format!("ws://{}", listener.local_addr().unwrap());
        rt.spawn(server.start_async_on(listener));
        address
    }

    /// End-to-end backend extension over the wire: the client ships a custom op as
    /// `OperationIr::Custom`, and the server executes it through a handler registered on the
    /// builder. Mirrors how a backend extension hosts its ops — the user hand-writes the client
    /// side (here, building the `CustomOpIr`) and registers the server handler.
    ///
    /// Only runs without `fusion`, since it drives the router client (`RemoteBackend`) directly.
    #[test]
    #[cfg(not(feature = "fusion"))]
    pub fn test_custom_op_over_websocket() {
        use burn_backend::TensorMetadata;
        use burn_ir::{CustomOpIr, OperationIr, ScalarIr, TensorIr};
        use burn_router::RouterClient;

        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_io()
            .build()
            .unwrap();

        // Host a "scale" custom op: multiply the input float tensor by a scalar argument.
        let address =
            serve(
                &rt,
                crate::server::RemoteServerBuilder::<Flex>::new(vec![Default::default()])
                    .custom_op("scale", |handles, ir, _device| {
                        let input = handles.get_float_tensor::<Flex>(&ir.inputs[0]);
                        let factor: Scalar = ir.scalars[0].into();
                        let output = Flex::float_mul_scalar(input, factor);
                        handles.register_float_tensor::<Flex>(&ir.outputs[0].id, output);
                    }),
            );

        // Drive the remote backend directly (no autodiff/dispatch glue). A real backend extension
        // would wrap this in a hand-written `impl MyExt for RemoteBackend`.
        let device = RemoteDevice::websocket(&address, 0);
        let input = <RemoteBackend as FloatTensorOps<RemoteBackend>>::float_from_data(
            TensorData::from([2.0f32, 4.0, 6.0]),
            &device,
        );

        // Client side: build the custom op (input tensor + the scale factor as a scalar) and ship
        // it through the remote client as `OperationIr::Custom`.
        let client = input.client.clone();
        let shape = input.shape();
        let dtype = input.dtype();
        let out_ir = TensorIr::uninit(client.create_empty_handle(), shape, dtype);
        let desc = CustomOpIr::with_scalars(
            "scale",
            &[input.into_ir()],
            &[out_ir],
            vec![ScalarIr::Float(3.0)],
        );
        let out = client.register(OperationIr::Custom(desc)).remove(0);

        let data = rt
            .block_on(<RemoteBackend as FloatTensorOps<RemoteBackend>>::float_into_data(out))
            .unwrap();
        let values: Vec<f32> = data.try_to_vec().unwrap();
        assert_eq!(values, vec![6.0, 12.0, 18.0]);

        rt.shutdown_background();
    }

    /// The Iroh counterpart of [`test_custom_op_over_websocket`]: the server hosts a "scale" handler
    /// registered on the protocol's custom-op registry, and the client ships the op as
    /// `OperationIr::Custom`. Confirms custom ops travel the merged session path over Iroh just as
    /// they do over WebSocket.
    ///
    /// Only runs without `fusion`, since it drives the router client (`RemoteBackend`) directly.
    #[test]
    #[cfg(not(feature = "fusion"))]
    pub fn test_custom_op_over_iroh() {
        use crate::{
            BURN_REMOTE_ALPN,
            server::{AllowAll, CustomOpRegistry, IrohRemoteProtocol},
            telemetry::TelemetryProbe,
        };
        use burn_backend::TensorMetadata;
        use burn_ir::{CustomOpIr, OperationIr, ScalarIr, TensorIr};
        use burn_router::RouterClient;
        use iroh::{Endpoint, RelayMode, endpoint::presets, protocol::Router};

        async fn local_endpoint() -> Endpoint {
            Endpoint::builder(presets::Minimal)
                .relay_mode(RelayMode::Disabled)
                .clear_ip_transports()
                .bind_addr("127.0.0.1:0")
                .unwrap()
                .bind()
                .await
                .unwrap()
        }

        // Server on its own runtime, hosting a "scale" custom op (multiply the input by a scalar).
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap();

        let server = rt.block_on(local_endpoint());
        let server_addr = server.addr();

        let mut custom_ops = CustomOpRegistry::<Flex>::default();
        custom_ops.register("scale", |handles, ir, _device| {
            let input = handles.get_float_tensor::<Flex>(&ir.inputs[0]);
            let factor: Scalar = ir.scalars[0].into();
            let output = Flex::float_mul_scalar(input, factor);
            handles.register_float_tensor::<Flex>(&ir.outputs[0].id, output);
        });

        let protocol = IrohRemoteProtocol::<Flex>::new(
            server.clone(),
            vec![Default::default()],
            std::sync::Arc::new(AllowAll),
            TelemetryProbe::disabled(),
            custom_ops,
        );

        // spawn() must run in a runtime context so iroh can schedule its tasks.
        let router = {
            let _guard = rt.enter();
            Router::builder(server)
                .accept(BURN_REMOTE_ALPN, protocol)
                .spawn()
        };

        let client = rt.block_on(local_endpoint());
        let device = RemoteDevice::iroh(&client, server_addr, 0);

        // Client side: build the custom op and ship it as `OperationIr::Custom`, exactly as the
        // WebSocket test does -- only the transport differs.
        let input = <RemoteBackend as FloatTensorOps<RemoteBackend>>::float_from_data(
            TensorData::from([2.0f32, 4.0, 6.0]),
            &device,
        );
        let remote_client = input.client.clone();
        let shape = input.shape();
        let dtype = input.dtype();
        let out_ir = TensorIr::uninit(remote_client.create_empty_handle(), shape, dtype);
        let desc = CustomOpIr::with_scalars(
            "scale",
            &[input.into_ir()],
            &[out_ir],
            vec![ScalarIr::Float(3.0)],
        );
        let out = remote_client.register(OperationIr::Custom(desc)).remove(0);

        let data = rt
            .block_on(<RemoteBackend as FloatTensorOps<RemoteBackend>>::float_into_data(out))
            .unwrap();
        let values: Vec<f32> = data.try_to_vec().unwrap();
        assert_eq!(values, vec![6.0, 12.0, 18.0]);

        rt.block_on(router.shutdown()).unwrap();
    }

    /// Run `body` on a worker thread and report whether it finished within `timeout`.
    ///
    /// A panic inside `body` counts as finished: a failure that surfaces is what these tests want,
    /// only a hang fails them.
    fn finishes_within(timeout: std::time::Duration, body: impl FnOnce() + Send + 'static) -> bool {
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(body));
            let _ = tx.send(());
        });
        rx.recv_timeout(timeout).is_ok()
    }

    /// A client that disconnects abruptly mid-session (socket dropped, no `Close`) must not wedge
    /// the server: it should clean the session up and keep serving everyone else. We drive a raw
    /// connection here because a `Device`'s client is process-cached and never dropped mid-test.
    #[test]
    fn test_client_disconnect_handled_cleanly_by_server() {
        type Client = burn_communication::websocket::WsClient;

        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_io()
            .build()
            .unwrap();

        let address = serve(
            &rt,
            crate::server::RemoteServerBuilder::<Flex>::new(vec![Default::default()]),
        );

        // Raw client: connect a submit stream, init a session and send one task so the server
        // spawns the session worker, then drop the socket without a `Close` to mimic a crash.
        {
            let rtc = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            let server = burn_communication::Address::from_str(&address).unwrap();
            let session_id = SessionId::new();

            rtc.block_on(async {
                let mut submit = Client::connect(server, "session")
                    .await
                    .expect("raw session connect");

                let frame = |msgs: Vec<RemoteMessage>| -> Message {
                    Message::new(rmp_serde::to_vec(&msgs).unwrap().into())
                };

                submit
                    .send(frame(vec![RemoteMessage::Init(
                        crate::shared::SessionInit::new(session_id, 0, vec![]),
                    )]))
                    .await
                    .expect("send init");
                submit
                    .send(frame(vec![RemoteMessage::Task(Task::Seed(0))]))
                    .await
                    .expect("send task");
                // Drop `submit` here (end of block): the server sees the stream end without a
                // `Close` and must run the cleanup path.
            });
        }

        // Give the server a moment to tear the abandoned session down.
        std::thread::sleep(std::time::Duration::from_millis(500));

        // The server must have survived: a fresh, normal client on the same server still works.
        let finished = finishes_within(std::time::Duration::from_secs(10), move || {
            let device = RemoteDevice::websocket(&address, 0);
            let input =
                RemoteBackend::float_from_data(TensorData::from([[10.0f32, 20.0]]), &device);
            let output = RemoteBackend::float_mul_scalar(input, Scalar::from(3.0f32));
            let data = burn_std::reader::try_read_sync(RemoteBackend::float_into_data(output))
                .expect("remote read should resolve synchronously")
                .expect("read should succeed");
            assert_eq!(data.try_to_vec::<f32>().unwrap(), vec![30.0, 60.0]);
        });
        assert!(
            finished,
            "server stopped serving after a client disconnected abruptly"
        );

        rt.shutdown_timeout(std::time::Duration::from_millis(100));
    }
}

#[cfg(all(test, feature = "fusion", feature = "server"))]
mod fusion_tests {
    use crate::{RemoteBackend, RemoteDevice, client::RemoteChannel, tests::serve};
    use burn_backend::{Backend, Shape, TensorData};
    use burn_router::BackendRouter;

    // `RemoteBackend` is `Fusion<PlainRemote>` under the `fusion` feature; `PlainRemote` is the
    // unwrapped router backend, used as the reference to compare against.
    type PlainRemote = BackendRouter<RemoteChannel>;

    fn input() -> TensorData {
        TensorData::from([[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]])
    }

    /// Run the same small multi-op graph `iters` times on backend `B`, reading the result each
    /// iteration. Every iteration has an identical op structure, so a fusion backend registers the
    /// optimization once and replays it by id on the later iterations.
    ///
    /// The graph deliberately includes a reshape, so one of the intermediate tensors (`c`) has a
    /// *different* shape than the inputs/outputs — exercising the server's reconstruction of
    /// intermediate shapes from the shape-dim map (rather than them being sent per replay).
    fn run<B: Backend<Device = RemoteDevice>>(
        device: &RemoteDevice,
        iters: usize,
    ) -> Vec<Vec<f32>> {
        let mut out = Vec::new();
        for _ in 0..iters {
            let a = B::float_from_data(input(), device); // [2, 3]
            let b = B::float_exp(a); // [2, 3]
            let c = B::float_reshape(b, Shape::from([3, 2])); // [3, 2] intermediate (distinct shape)
            let d = B::float_log(c); // [3, 2]
            let data = burn_std::reader::try_read_sync(B::float_into_data(d))
                .expect("remote read should resolve synchronously")
                .expect("read should succeed");
            out.push(data.try_to_vec::<f32>().unwrap());
        }
        out
    }

    /// The fusion-enabled remote backend must produce exactly the same results as the plain remote
    /// backend across a repeated computation that exercises register-once + replay-by-id.
    #[test]
    fn fusion_matches_plain_remote() {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_io()
            .build()
            .unwrap();

        let plain_address = serve(
            &rt,
            crate::server::RemoteServerBuilder::<burn_flex::Flex>::new(vec![Default::default()]),
        );
        let fused_address = serve(
            &rt,
            crate::server::RemoteServerBuilder::<burn_flex::Flex>::new(vec![Default::default()]),
        );

        let plain_device = RemoteDevice::websocket(&plain_address, 0);
        let fused_device = RemoteDevice::websocket(&fused_address, 0);

        let iters = 5;
        let expected = run::<PlainRemote>(&plain_device, iters);
        let actual = run::<RemoteBackend>(&fused_device, iters);

        assert_eq!(actual.len(), iters);
        assert_eq!(expected.len(), iters);
        for (a, e) in actual.iter().zip(expected.iter()) {
            assert_eq!(a.len(), e.len());
            for (av, ev) in a.iter().zip(e.iter()) {
                assert!(
                    (av - ev).abs() < 1e-5,
                    "fusion result {av} differs from plain remote {ev}"
                );
            }
        }

        rt.shutdown_background();
    }

    /// A *source* custom op (no tensor inputs — it builds a tensor from scalars on the server) whose
    /// output is then consumed by a follow-up op, read back, repeated to exercise register-once +
    /// replay. This mirrors the server-side data-loader extension pattern and isolates it from the
    /// training stack — if the fusion graph mishandles a source custom op's outputs, it surfaces here
    /// as a "Should have handle for tensor ..." panic on the server.
    #[test]
    fn fusion_custom_source_op_then_followup() {
        use crate::client::CustomOpClient;
        use burn_backend::DType;
        use burn_backend::ops::FloatTensorOps;
        use burn_flex::Flex;
        use burn_ir::{CustomOpIr, OperationOutput, ScalarIr, TensorIr};

        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_io()
            .build()
            .unwrap();

        let address =
            serve(
                &rt,
                crate::server::RemoteServerBuilder::<Flex>::new(vec![Default::default()])
                    .custom_op("make_floats", |handles, ir, device| {
                        // Build a 1-D float tensor from the op's scalars: a pure source, with no inputs.
                        let values: Vec<f32> = ir.scalars.iter().map(|s| s.elem::<f32>()).collect();
                        let n = values.len();
                        let tensor = Flex::float_from_data(TensorData::new(values, [n]), device);
                        handles.register_float_tensor::<Flex>(&ir.outputs[0].id, tensor);
                    }),
            );

        let device = RemoteDevice::websocket(&address, 0);

        for i in 0..5 {
            let client = CustomOpClient::new(&device);
            let out_ir =
                TensorIr::uninit(client.create_empty_handle(), Shape::from([3]), DType::F32);
            let made = client
                .register(CustomOpIr::with_scalars(
                    "make_floats",
                    &[],
                    &[out_ir],
                    vec![
                        ScalarIr::Float(1.0),
                        ScalarIr::Float(2.0),
                        ScalarIr::Float(3.0),
                    ],
                ))
                .output();

            // Follow-up op consuming the source output — forces a graph that references the custom
            // op's output as an input, the scenario that breaks during training.
            let doubled = <RemoteBackend as FloatTensorOps<RemoteBackend>>::float_exp(made);
            let data = burn_std::reader::try_read_sync(<RemoteBackend as FloatTensorOps<
                RemoteBackend,
            >>::float_into_data(doubled))
            .expect("remote read should resolve synchronously")
            .expect("read should succeed");

            let values = data.try_to_vec::<f32>().unwrap();
            let expected = [1.0f32.exp(), 2.0f32.exp(), 3.0f32.exp()];
            for (a, e) in values.iter().zip(expected.iter()) {
                assert!((a - e).abs() < 1e-4, "iter {i}: {a} vs {e}");
            }
        }

        rt.shutdown_background();
    }

    /// Read a source custom op's output *directly* (it is the boundary output, with no follow-up op
    /// consuming it). This is what the data loader does — `batch.tokens.to_data()` — and the case the
    /// other two tests don't cover (they always feed the output into another op first).
    #[test]
    fn fusion_read_source_output_directly() {
        use crate::client::CustomOpClient;
        use burn_backend::DType;
        use burn_backend::ops::FloatTensorOps;
        use burn_flex::Flex;
        use burn_ir::{CustomOpIr, OperationOutput, ScalarIr, TensorIr};

        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_io()
            .build()
            .unwrap();

        let address =
            serve(
                &rt,
                crate::server::RemoteServerBuilder::<Flex>::new(vec![Default::default()])
                    .custom_op("make_floats", |handles, ir, device| {
                        let values: Vec<f32> = ir.scalars.iter().map(|s| s.elem::<f32>()).collect();
                        let n = values.len();
                        let tensor = Flex::float_from_data(TensorData::new(values, [n]), device);
                        handles.register_float_tensor::<Flex>(&ir.outputs[0].id, tensor);
                    }),
            );

        let device = RemoteDevice::websocket(&address, 0);

        for i in 0..5 {
            let client = CustomOpClient::new(&device);
            let out_ir =
                TensorIr::uninit(client.create_empty_handle(), Shape::from([3]), DType::F32);
            let made = client
                .register(CustomOpIr::with_scalars(
                    "make_floats",
                    &[],
                    &[out_ir],
                    vec![
                        ScalarIr::Float(1.0),
                        ScalarIr::Float(2.0),
                        ScalarIr::Float(3.0),
                    ],
                ))
                .output();

            // Read the source output directly — no follow-up op.
            let data = burn_std::reader::try_read_sync(<RemoteBackend as FloatTensorOps<
                RemoteBackend,
            >>::float_into_data(made))
            .expect("remote read should resolve synchronously")
            .expect("read should succeed");
            let values = data.try_to_vec::<f32>().unwrap();
            assert_eq!(values, vec![1.0, 2.0, 3.0], "iter {i}");
        }

        rt.shutdown_background();
    }

    /// Closer to the data-loader: a source custom op with *three* outputs that are consumed at
    /// *different depths* of the following graph (one immediately, one mid-graph, one only at the
    /// end — like `tokens`/`mask`/`labels`). Exercises a source op's outputs surviving across many
    /// ops before being bound as inputs, under register-once + replay.
    #[test]
    fn fusion_custom_source_multi_output_long_lived() {
        use crate::client::CustomOpClient;
        use burn_backend::DType;
        use burn_backend::ops::FloatTensorOps;
        use burn_flex::Flex;
        use burn_ir::{CustomOpIr, OperationOutput, ScalarIr, TensorIr};

        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_io()
            .build()
            .unwrap();

        let address =
            serve(
                &rt,
                crate::server::RemoteServerBuilder::<Flex>::new(vec![Default::default()])
                    .custom_op("make3", |handles, ir, device| {
                        // 9 scalars → three [3] outputs (chunks of 3).
                        let values: Vec<f32> = ir.scalars.iter().map(|s| s.elem::<f32>()).collect();
                        for (i, out) in ir.outputs.iter().enumerate() {
                            let chunk = values[i * 3..(i + 1) * 3].to_vec();
                            let tensor = Flex::float_from_data(TensorData::new(chunk, [3]), device);
                            handles.register_float_tensor::<Flex>(&out.id, tensor);
                        }
                    }),
            );

        let device = RemoteDevice::websocket(&address, 0);

        for i in 0..5 {
            let client = CustomOpClient::new(&device);
            let mk = |client: &CustomOpClient| {
                TensorIr::uninit(client.create_empty_handle(), Shape::from([3]), DType::F32)
            };
            let [a, b, c] = client
                .register(CustomOpIr::with_scalars(
                    "make3",
                    &[],
                    &[mk(&client), mk(&client), mk(&client)],
                    (1..=9).map(|v| ScalarIr::Float(v as f64)).collect(),
                ))
                .outputs::<3>();

            // a consumed immediately, b mid-graph, c only at the end — so b and c are source-op
            // outputs that survive across several ops before being bound as inputs.
            type B = RemoteBackend;
            let t = <B as FloatTensorOps<B>>::float_exp(a);
            let t = <B as FloatTensorOps<B>>::float_add(t, b);
            let t = <B as FloatTensorOps<B>>::float_log(t);
            let t = <B as FloatTensorOps<B>>::float_add(t, c);
            let data =
                burn_std::reader::try_read_sync(<B as FloatTensorOps<B>>::float_into_data(t))
                    .expect("remote read should resolve synchronously")
                    .expect("read should succeed");

            let values = data.try_to_vec::<f32>().unwrap();
            // a=[1,2,3], b=[4,5,6], c=[7,8,9]; t = log(exp(a)+b) + c
            let expected: Vec<f32> = (0..3)
                .map(|k| {
                    let a = (k + 1) as f32;
                    let b = (k + 4) as f32;
                    let c = (k + 7) as f32;
                    (a.exp() + b).ln() + c
                })
                .collect();
            for (g, e) in values.iter().zip(expected.iter()) {
                assert!((g - e).abs() < 1e-3, "iter {i}: {g} vs {e}");
            }
        }

        rt.shutdown_background();
    }

    /// Regression guard for the `free_handle` drop-suppression override
    /// (`RouterFusionRuntime::free_handle`): a *second live reference* to a source op's output is
    /// held across the graph drain that consumes the first one. The override removes the drained
    /// block's container entry and bumps the handle refcount so `RouterTensor::drop` doesn't
    /// re-register a redundant server `Drop`; if that bookkeeping mishandles a surviving clone, the
    /// retained tensor's id is freed too early and the *next* graph that uses it panics on the
    /// server with "Should have handle for tensor ..." (or reads back garbage). Looping exercises
    /// register-once + replay so the bug would surface on a later iteration even if the first slips
    /// through.
    #[test]
    fn fusion_custom_source_output_clone_survives_drain() {
        use crate::client::CustomOpClient;
        use burn_backend::DType;
        use burn_backend::ops::FloatTensorOps;
        use burn_flex::Flex;
        use burn_ir::{CustomOpIr, OperationOutput, ScalarIr, TensorIr};

        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_io()
            .build()
            .unwrap();

        let address =
            serve(
                &rt,
                crate::server::RemoteServerBuilder::<Flex>::new(vec![Default::default()])
                    .custom_op("make_floats", |handles, ir, device| {
                        let values: Vec<f32> = ir.scalars.iter().map(|s| s.elem::<f32>()).collect();
                        let n = values.len();
                        let tensor = Flex::float_from_data(TensorData::new(values, [n]), device);
                        handles.register_float_tensor::<Flex>(&ir.outputs[0].id, tensor);
                    }),
            );

        let device = RemoteDevice::websocket(&address, 0);

        type B = RemoteBackend;
        for i in 0..5 {
            let client = CustomOpClient::new(&device);
            let out_ir =
                TensorIr::uninit(client.create_empty_handle(), Shape::from([3]), DType::F32);
            let made = client
                .register(CustomOpIr::with_scalars(
                    "make_floats",
                    &[],
                    &[out_ir],
                    vec![
                        ScalarIr::Float(1.0),
                        ScalarIr::Float(2.0),
                        ScalarIr::Float(3.0),
                    ],
                ))
                .output();

            // A second live reference to the same source output. Kept alive across the drain that
            // consumes `made` below — this is the scenario the override's refcount bump must not
            // free.
            let kept = made.clone();

            // Consume `made` in a graph and read it back, forcing a drain that frees the block's
            // handles while `kept` still references the source output's id.
            let exp = <B as FloatTensorOps<B>>::float_exp(made);
            let exp_data =
                burn_std::reader::try_read_sync(<B as FloatTensorOps<B>>::float_into_data(exp))
                    .expect("remote read should resolve synchronously")
                    .expect("read should succeed");
            let exp_values = exp_data.try_to_vec::<f32>().unwrap();
            let exp_expected = [1.0f32.exp(), 2.0f32.exp(), 3.0f32.exp()];
            for (g, e) in exp_values.iter().zip(exp_expected.iter()) {
                assert!((g - e).abs() < 1e-3, "iter {i} (exp): {g} vs {e}");
            }

            // Now use the retained clone in a *new* graph. If the drain above freed the id out from
            // under it, this read fails server-side or returns garbage.
            let log = <B as FloatTensorOps<B>>::float_log(kept);
            let log_data =
                burn_std::reader::try_read_sync(<B as FloatTensorOps<B>>::float_into_data(log))
                    .expect("remote read should resolve synchronously")
                    .expect("read should succeed");
            let log_values = log_data.try_to_vec::<f32>().unwrap();
            let log_expected = [1.0f32.ln(), 2.0f32.ln(), 3.0f32.ln()];
            for (g, e) in log_values.iter().zip(log_expected.iter()) {
                assert!((g - e).abs() < 1e-3, "iter {i} (log): {g} vs {e}");
            }
        }

        rt.shutdown_background();
    }
}
