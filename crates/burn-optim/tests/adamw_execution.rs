#![cfg(all(feature = "adamw-execution-tests", not(target_family = "wasm")))]

use burn::backend::ir::{IrVisitorMut, TensorId, TensorIr};
use burn::module::{Module, Param};
use burn::tensor::{
    CaptureTensor, CapturedGraph, DType, Device, GradientCheckpointingStrategy, Sharding, Shards,
    Tensor, TensorData, Tolerance,
    remote::RemoteHost,
    server::{RemoteServer, WebSocketTransport},
};
use burn_core as burn;
use burn_cubecl::cubecl::logging::{LaunchObservation, LaunchObserver};
use burn_optim::{
    AdamWConfig, AdamWExecutionConfig, AdamWState, GradientsParams, ModuleOptimizer, Optimizer,
    RecordState, StateSink,
};
use std::{
    collections::BTreeMap,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering::Relaxed},
    },
};

const LR: f64 = 0.01;
const PARAMETERS: [[f32; 4]; 2] = [[1.0, -2.0, 3.0, -4.0], [5.0, -6.0, 7.0, -8.0]];
const GRADIENTS: [[f32; 4]; 2] = [[0.5, -1.0, 2.0, -0.25], [-0.75, 0.125, -1.5, 3.0]];
static TEST_LOCK: Mutex<()> = Mutex::new(());

#[derive(Default)]
struct Launches {
    adamw: AtomicUsize,
    total: AtomicUsize,
}

impl LaunchObserver for Launches {
    fn launched(&self, name: &'static str) {
        // Include server/member threads: a local-thread observer could miss an invalid dispatch.
        self.total.fetch_add(1, Relaxed);
        if name.contains("optim::adamw::") {
            self.adamw.fetch_add(1, Relaxed);
        }
    }
}

fn observe_fallback() -> (Arc<Launches>, LaunchObservation) {
    assert!(
        !burn_cubecl::FUSED,
        "run this suite without default/fusion features"
    );
    let counts = Arc::new(Launches::default());
    let observation = LaunchObservation::new(counts.clone());
    let cpu = Device::cpu();
    let output = config(AdamWExecutionConfig::Fused).build().step(
        LR,
        tensor(PARAMETERS[0], &cpu),
        gradient(0, 0, &cpu),
        None,
    );
    cpu.sync().unwrap();
    assert_eq!(counts.adamw.load(Relaxed), 1, "observer positive control");
    drop(output);
    counts.adamw.store(0, Relaxed);
    counts.total.store(0, Relaxed);
    (counts, observation)
}

fn config(execution: AdamWExecutionConfig) -> AdamWConfig {
    AdamWConfig::new()
        .with_beta_1(0.5)
        .with_beta_2(0.5)
        .with_weight_decay(0.1)
        .with_amsgrad(true)
        .with_execution(Some(execution))
}

fn tensor(values: [f32; 4], device: &Device) -> Tensor<2> {
    if let Some(members) = device.shard_members() {
        let shards = members
            .iter()
            .enumerate()
            .map(|(i, member)| {
                Tensor::from_data(
                    TensorData::new(values[i * 2..i * 2 + 2].to_vec(), [1, 2]),
                    member,
                )
            })
            .collect();
        Tensor::from_shards(
            Shards {
                shards,
                sharding: Sharding::Split { dim: 0 },
            },
            device,
        )
        .unwrap()
    } else {
        Tensor::from_data(TensorData::new(values.to_vec(), [2, 2]), device)
    }
}

fn gradient(index: usize, step: usize, device: &Device) -> Tensor<2> {
    tensor(
        GRADIENTS[index].map(|g| g * if step == 0 { 1.0 } else { -0.5 }),
        device,
    )
}

fn assert_data(expected: &TensorData, actual: &TensorData) {
    assert_eq!(expected.dtype(), DType::F32);
    assert_eq!(actual.dtype(), expected.dtype());
    assert_eq!(actual.shape(), expected.shape());
    expected.assert_approx_eq::<f32>(actual, Tolerance::rel_abs(1e-5, 1e-6));
}

fn moments(state: &AdamWState<2>) -> [&Tensor<2>; 3] {
    [
        &state.momentum.moment_1,
        &state.momentum.moment_2,
        state.momentum.max_moment_2.as_ref().unwrap(),
    ]
}

#[derive(Module, Debug)]
struct Pair {
    first: Param<Tensor<2>>,
    second: Param<Tensor<2>>,
}

impl Pair {
    fn new(device: &Device) -> Self {
        Self {
            first: Param::from_tensor(tensor(PARAMETERS[0], device)),
            second: Param::from_tensor(tensor(PARAMETERS[1], device)),
        }
    }

    fn params(&self) -> [&Param<Tensor<2>>; 2] {
        [&self.first, &self.second]
    }

    fn grads(&self, step: usize, device: &Device) -> GradientsParams {
        let mut grads = GradientsParams::new();
        for (i, param) in self.params().into_iter().enumerate() {
            grads.register(param.id, gradient(i, step, device));
        }
        grads
    }
}

type Step = (Tensor<2>, Option<AdamWState<2>>);

fn assert_record(optimizer: &ModuleOptimizer, model: &Pair, reference: &[Step], time: usize) {
    let reader = burn_pack::Reader::from_bytes(optimizer.into_bytes().unwrap()).unwrap();
    let mut expected = BTreeMap::new();
    for ((param, (_, state)), path) in model
        .params()
        .into_iter()
        .zip(reference)
        .zip(["first", "second"])
    {
        let prefix = param.id.val().to_string();
        assert_eq!(reader.metadata()[&prefix], path);
        assert_eq!(
            usize::try_from(reader.scalars()[&format!("{prefix}.__rank")]).unwrap(),
            2
        );
        assert_eq!(
            usize::try_from(reader.scalars()[&format!("{prefix}.momentum.time")]).unwrap(),
            time
        );
        let mut sink = StateSink::default();
        state.as_ref().unwrap().state_flatten(&prefix, &mut sink);
        for (name, scalar) in sink.scalars {
            assert_eq!(reader.scalars().get(&name), Some(&scalar));
        }
        for (name, data) in sink.tensors {
            expected.insert(name, (param.id.val(), data));
        }
    }
    let tensors = reader.into_tensors().unwrap();
    assert_eq!(tensors.len(), 6);
    for item in tensors {
        let (name, dtype, shape, id, bytes) = item.into_parts().unwrap();
        let (expected_id, data) = expected.remove(&name).unwrap();
        assert_eq!(id, Some(expected_id));
        assert_data(&data, &TensorData::from_bytes(bytes, shape, dtype));
    }
    assert!(expected.is_empty());
}

fn check_updates(device: Device) {
    let cpu = Device::cpu();
    let tracked = device.clone().autodiff().gradient_checkpointing();
    let strategy = tracked.gradient_checkpointing_strategy().unwrap();
    assert_ne!(strategy, GradientCheckpointingStrategy::Disabled);
    let mut model = Pair::new(&tracked);
    let ids = model.params().map(|param| param.id);
    if device.shard_members().is_some() {
        assert_eq!(
            model.first.val().sharding(),
            Some(Sharding::Split { dim: 0 })
        );
    }
    let mut optimizer = config(AdamWExecutionConfig::Fused).init();
    let reference = config(AdamWExecutionConfig::Reference).build();
    let fused = config(AdamWExecutionConfig::Fused).build();
    let mut expected: Vec<Step> = PARAMETERS
        .into_iter()
        .map(|p| (tensor(p, &cpu), None))
        .collect();
    let mut actual: Vec<Step> = PARAMETERS
        .into_iter()
        .map(|p| (tensor(p, &device), None))
        .collect();
    let (launches, _observation) = observe_fallback();
    for step in 0..2 {
        let grads = model.grads(step, &device);
        model = optimizer.step(LR, model, grads);
        expected = expected
            .into_iter()
            .enumerate()
            .map(|(i, (p, s))| reference.step(LR, p, gradient(i, step, &cpu), s))
            .collect();
        actual = actual
            .into_iter()
            .enumerate()
            .map(|(i, (p, s))| fused.step(LR, p, gradient(i, step, &device), s))
            .collect();
        for (i, param) in model.params().into_iter().enumerate() {
            let value = param.val();
            assert_eq!(param.id, ids[i]);
            assert_eq!(value.device(), tracked);
            assert!(value.is_require_grad());
            assert_eq!(value.gradient_checkpointing_strategy(), Some(strategy));
            assert_data(&expected[i].0.to_data(), &value.to_data());
            assert_eq!(actual[i].0.device(), device);
            assert_data(&expected[i].0.to_data(), &actual[i].0.to_data());
            let state = actual[i].1.as_ref().unwrap();
            assert_eq!(state.momentum.time, step + 1);
            let g = GRADIENTS[i];
            // The reversed smaller second gradient cancels m1 and retains an older AMSGrad max.
            let oracle = [
                g.map(|v| if step == 0 { v * 0.5 } else { 0.0 }),
                g.map(|v| v * v * if step == 0 { 0.5 } else { 0.375 }),
                g.map(|v| v * v * 0.5),
            ];
            for ((a, b), values) in moments(expected[i].1.as_ref().unwrap())
                .into_iter()
                .zip(moments(state))
                .zip(oracle)
            {
                assert_eq!(b.device(), device);
                assert!(!b.is_autodiff());
                assert_data(&a.to_data(), &b.to_data());
                assert_data(&TensorData::new(values.to_vec(), [2, 2]), &b.to_data());
            }
        }
        assert_record(&optimizer, &model, &expected, step + 1);
        device.sync().unwrap();
        assert_eq!(launches.adamw.load(Relaxed), 0);
        if step == 0 {
            optimizer = config(AdamWExecutionConfig::Fused)
                .init()
                .from_bytes(optimizer.into_bytes().unwrap())
                .unwrap();
        }
    }
    assert!(
        launches.total.load(Relaxed) > 0,
        "reference operations must execute"
    );
}

#[test]
fn fused_split_shards_preserve_updates_state_and_parameter_context() {
    let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let cpu = Device::cpu();
    let device = Device::shard_options(&[cpu.clone(), cpu]).init().unwrap();
    assert!(matches!(
        device.as_dispatch(),
        burn::backend::DispatchDevice::ShardedCompute(_)
    ));
    check_updates(device);
}

#[derive(Default)]
struct CanonicalIds(BTreeMap<TensorId, TensorId>);

impl CanonicalIds {
    fn id(&mut self, original: TensorId) -> TensorId {
        let next = TensorId::new(self.0.len() as u64);
        *self.0.entry(original).or_insert(next)
    }
}

impl IrVisitorMut for CanonicalIds {
    fn visit_tensor_mut(&mut self, tensor: &mut TensorIr) {
        tensor.id = self.id(tensor.id);
    }
}

fn capture_steps(execution: AdamWExecutionConfig) -> CapturedGraph {
    let device = Device::capture();
    let optimizer = config(execution).build();
    let mut captured = device
        .capture_scope(|scope| {
            let mut pairs: Vec<Step> = PARAMETERS
                .into_iter()
                .map(|p| (tensor(p, &device), None))
                .collect();
            for step in 0..2 {
                pairs = pairs
                    .into_iter()
                    .enumerate()
                    .map(|(i, (p, s))| optimizer.step(LR, p, gradient(i, step, &device), s))
                    .collect();
                for (_, state) in &pairs {
                    assert_eq!(state.as_ref().unwrap().momentum.time, step + 1);
                }
            }
            let outputs = pairs
                .iter()
                .flat_map(|(p, state)| {
                    [p].into_iter()
                        .chain(moments(state.as_ref().unwrap()))
                        .map(CaptureTensor::capture_id)
                })
                .collect::<Vec<_>>();
            scope.complete([], outputs)
        })
        .unwrap();
    // Normalize only globally allocated IDs, preserving operations, values, shapes and statuses.
    let mut ids = CanonicalIds::default();
    for op in &mut captured.graph.operations {
        op.visit_mut(&mut ids);
    }
    captured.graph.inputs = captured
        .graph
        .inputs
        .into_iter()
        .map(|id| ids.id(id))
        .collect();
    captured.graph.outputs = captured
        .graph
        .outputs
        .into_iter()
        .map(|id| ids.id(id))
        .collect();
    captured.values = captured
        .values
        .into_iter()
        .map(|(id, value)| (ids.id(id), value))
        .collect();
    captured
}

#[test]
fn fused_capture_preserves_reference_ir_and_all_state_outputs_without_launching() {
    let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let (launches, _observation) = observe_fallback();
    let reference = capture_steps(AdamWExecutionConfig::Reference);
    let actual = capture_steps(AdamWExecutionConfig::Fused);
    assert!(!reference.graph.operations.is_empty());
    assert!(reference.graph.inputs.is_empty());
    assert_eq!(reference.graph.outputs.len(), 8);
    assert_eq!(reference.graph, actual.graph);
    assert!(!reference.values.is_empty());
    assert_eq!(reference.values.len(), actual.values.len());
    for (id, expected) in reference.values {
        let actual = &actual.values[&id];
        assert_eq!(expected.shape(), actual.shape());
        assert_eq!(expected.dtype(), actual.dtype());
        assert_eq!(expected.as_bytes(), actual.as_bytes());
    }
    assert_eq!(launches.total.load(Relaxed), 0);
}

#[test]
fn fused_websocket_remote_preserves_updates_state_and_parameter_context() {
    let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let (done, completion) = std::sync::mpsc::channel();
    let worker = std::thread::spawn(move || {
        // Dropping a Tokio runtime can wait forever on blocking tasks after a failed assertion.
        struct Server(Option<tokio::runtime::Runtime>);
        impl Drop for Server {
            fn drop(&mut self) {
                self.0.take().unwrap().shutdown_background();
            }
        }
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .unwrap();
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let host = RemoteHost::websocket(&format!("ws://{}", listener.local_addr().unwrap()));
        runtime.spawn(
            RemoteServer::new([Device::cpu()])
                .serve_async(WebSocketTransport::from_listener(listener)),
        );
        let _server = Server(Some(runtime));
        let device = Device::remote_options(&host).init().unwrap();
        assert!(matches!(
            device.as_dispatch(),
            burn::backend::DispatchDevice::Remote(_)
        ));
        check_updates(device);
        let _ = done.send(());
    });
    match completion.recv_timeout(std::time::Duration::from_secs(30)) {
        Err(std::sync::mpsc::RecvTimeoutError::Timeout) => panic!("Remote AdamW test timed out"),
        Ok(()) | Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
            worker
                .join()
                .unwrap_or_else(|panic| std::panic::resume_unwind(panic));
        }
    }
}
