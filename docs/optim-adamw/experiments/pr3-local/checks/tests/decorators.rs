use burn::module::{Module, Param};
use burn::tensor::{
    CaptureTensor, CapturedGraph, DType, Device, GradientCheckpointingStrategy, Tensor, TensorData,
    Tolerance,
};
use burn_cubecl::cubecl::logging::{LaunchObservation, LaunchObserver};
use burn_ir::{IrVisitorMut, TensorId, TensorIr};
use burn_optim::{
    AdamWConfig, AdamWExecutionConfig, AdamWState, GradientsParams, ModuleOptimizer, Optimizer,
    RecordState, StateSink,
};
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

const LR: f64 = 0.01;
const PARAMETERS: [[f32; 4]; 2] = [[1.0, -2.0, 3.0, -4.0], [5.0, -6.0, 7.0, -8.0]];
const GRADIENTS: [[f32; 4]; 2] = [[0.5, -1.0, 2.0, -0.25], [-0.75, 0.125, -1.5, 3.0]];
static TEST_LOCK: Mutex<()> = Mutex::new(());

// Observe all threads: Group can dispatch work through its members. These tests run serially.
#[derive(Default)]
struct Launches(Mutex<Vec<&'static str>>);

impl LaunchObserver for Launches {
    fn launched(&self, name: &'static str) {
        self.0.lock().unwrap().push(name);
    }
}

impl Launches {
    fn start() -> (Arc<Self>, LaunchObservation) {
        let launches = Arc::new(Self::default());
        let observation = LaunchObservation::new(launches.clone());
        (launches, observation)
    }

    fn counts(&self) -> (usize, usize, usize) {
        let names = self.0.lock().unwrap();
        (
            names
                .iter()
                .filter(|n| n.contains("adamw_foreach::"))
                .count(),
            names
                .iter()
                .filter(|n| n.contains("optim::adamw::eager::adamw::"))
                .count(),
            names.len(),
        )
    }

    fn assert_fallback(&self) {
        let (batch, single, _) = self.counts();
        assert_eq!((batch, single), (0, 0), "unexpected AdamW raw launch");
    }
}

fn config(execution: AdamWExecutionConfig) -> AdamWConfig {
    AdamWConfig::new()
        .with_beta_1(0.5)
        .with_beta_2(0.5)
        .with_epsilon(1e-5)
        .with_weight_decay(0.1)
        .with_amsgrad(true)
        .with_execution(Some(execution))
}

fn tensor(values: [f32; 4], device: &Device) -> Tensor<2> {
    Tensor::from_data(TensorData::new(values.to_vec(), [2, 2]), device)
}

fn gradient(index: usize, step: usize, device: &Device) -> Tensor<2> {
    let scale = if step == 0 { 1.0 } else { -0.5 };
    tensor(GRADIENTS[index].map(|g| g * scale), device)
}

fn assert_data(expected: &TensorData, actual: &TensorData) {
    assert_eq!(actual.dtype(), DType::F32);
    assert_eq!(expected.dtype(), actual.dtype());
    assert_eq!(expected.shape(), actual.shape());
    expected.assert_approx_eq::<f32>(actual, Tolerance::rel_abs(1e-5, 1e-6));
}

fn assert_tensor(expected: &Tensor<2>, actual: &Tensor<2>, device: &Device) {
    assert_eq!(actual.device(), *device);
    assert_eq!(expected.device(), *device);
    assert_data(&expected.to_data(), &actual.to_data());
}

fn state_tensors(state: &AdamWState<2>) -> [&Tensor<2>; 3] {
    [
        &state.momentum.moment_1,
        &state.momentum.moment_2,
        state
            .momentum
            .max_moment_2
            .as_ref()
            .expect("AMSGrad maximum"),
    ]
}

fn assert_state(expected: &AdamWState<2>, actual: &AdamWState<2>, device: &Device, time: usize) {
    assert_eq!(expected.momentum.time, time);
    assert_eq!(actual.momentum.time, time);
    for (expected, actual) in state_tensors(expected)
        .into_iter()
        .zip(state_tensors(actual))
    {
        assert_tensor(expected, actual, device);
        assert!(
            !actual.is_autodiff(),
            "optimizer state must remain on the inner device"
        );
    }
}

fn calibrate_observer() {
    assert!(
        !burn_cubecl::FUSED,
        "this harness requires raw CPU execution"
    );
    let cpu = Device::cpu();
    cpu.sync().unwrap();
    let (launches, _observation) = Launches::start();
    let output = config(AdamWExecutionConfig::Foreach).build().step(
        LR,
        tensor(PARAMETERS[0], &cpu),
        gradient(0, 0, &cpu),
        None,
    );
    cpu.sync().unwrap();
    let (batch, single, total) = launches.counts();
    assert_eq!((batch, single), (0, 1), "observer positive control failed");
    println!(
        "observer control: raw CPU Foreach direct step, batch={batch}, single={single}, total={total}"
    );
    drop(output);
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

    fn grads(&self, device: &Device, step: usize) -> GradientsParams {
        let mut grads = GradientsParams::new();
        for (i, param) in self.params().into_iter().enumerate() {
            grads.register(param.id, gradient(i, step, device));
        }
        grads
    }
}

fn check_module_record(
    optimizer: &ModuleOptimizer,
    model: &Pair,
    direct: &[(Tensor<2>, Option<AdamWState<2>>)],
    time: usize,
) {
    // OptimizerRecord's fields are private; read its public burnpack representation.
    let reader = burn_pack::Reader::from_bytes(optimizer.into_bytes().unwrap()).unwrap();
    let mut expected_tensors = BTreeMap::new();
    for ((param, (_, state)), path) in model
        .params()
        .into_iter()
        .zip(direct)
        .zip(["first", "second"])
    {
        let prefix = param.id.val().to_string();
        assert_eq!(reader.metadata().get(&prefix).unwrap(), path);
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
            expected_tensors.insert(name, (param.id.val(), data));
        }
    }
    let tensors = reader.into_tensors().unwrap();
    assert_eq!(tensors.len(), 6);
    for item in tensors {
        let (name, dtype, shape, param_id, bytes) = item.into_parts().unwrap();
        let (expected_id, expected_data) = expected_tensors.remove(&name).unwrap();
        assert_eq!(param_id, Some(expected_id));
        assert_data(&expected_data, &TensorData::from_bytes(bytes, shape, dtype));
    }
    assert!(expected_tensors.is_empty());
}

#[test]
fn group_module_and_direct_steps_preserve_state_and_use_reference_fallback() {
    let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    calibrate_observer();
    let cpu = Device::cpu();
    let group = Device::group_options(&[cpu.clone(), cpu.clone()])
        .init()
        .unwrap();
    assert!(matches!(
        group.as_dispatch(),
        burn::backend::DispatchDevice::Group(_)
    ));
    let tracked = group.clone().autodiff().gradient_checkpointing();
    let initial_strategy = tracked
        .gradient_checkpointing_strategy()
        .expect("the tracked fixture must enable autodiff");
    assert_ne!(
        initial_strategy,
        GradientCheckpointingStrategy::Disabled,
        "the fixture must enable checkpointing so losing it is observable"
    );
    println!("Group checkpointing oracle: initial tracked strategy={initial_strategy:?}");
    let mut expected_model = Pair::new(&tracked);
    let mut actual_model = expected_model.clone();
    let mut expected_optimizer = config(AdamWExecutionConfig::Reference).init();
    let mut actual_optimizer = config(AdamWExecutionConfig::Foreach).init();
    let reference = config(AdamWExecutionConfig::Reference).build();
    let foreach = config(AdamWExecutionConfig::Foreach).build();
    let make_direct = || {
        PARAMETERS
            .into_iter()
            .map(|p| (tensor(p, &group), None))
            .collect::<Vec<_>>()
    };
    let mut expected_direct = make_direct();
    let mut actual_direct = make_direct();

    group.sync().unwrap();
    let (launches, _observation) = Launches::start();
    for step in 0..2 {
        let expected_grads = expected_model.grads(&group, step);
        let actual_grads = actual_model.grads(&group, step);
        expected_model = expected_optimizer.step(LR, expected_model, expected_grads);
        actual_model = actual_optimizer.step(LR, actual_model, actual_grads);
        expected_direct = expected_direct
            .into_iter()
            .enumerate()
            .map(|(i, (p, s))| reference.step(LR, p, gradient(i, step, &group), s))
            .collect();
        actual_direct = actual_direct
            .into_iter()
            .enumerate()
            .map(|(i, (p, s))| foreach.step(LR, p, gradient(i, step, &group), s))
            .collect();

        for (i, (a, b)) in expected_model
            .params()
            .into_iter()
            .zip(actual_model.params())
            .enumerate()
        {
            assert_eq!(a.id, b.id);
            assert_tensor(&a.val(), &b.val(), &tracked);
            assert!(b.val().is_require_grad());
            assert_eq!(
                a.val().gradient_checkpointing_strategy(),
                Some(initial_strategy),
                "Reference must preserve the initial checkpointing strategy"
            );
            assert_eq!(
                b.val().gradient_checkpointing_strategy(),
                Some(initial_strategy),
                "Foreach must preserve the initial checkpointing strategy"
            );
            let (expected_p, expected_s) = &expected_direct[i];
            let (actual_p, actual_s) = &actual_direct[i];
            assert_tensor(expected_p, actual_p, &group);
            assert_tensor(actual_p, &b.val().inner(), &group);
            assert_state(
                expected_s.as_ref().unwrap(),
                actual_s.as_ref().unwrap(),
                &group,
                step + 1,
            );
            // The second gradient is smaller and reversed: stored moments must be consumed.
            let expected_m1 = GRADIENTS[i].map(|g| if step == 0 { 0.5 * g } else { 0.0 });
            let expected_m2 = GRADIENTS[i].map(|g| {
                if step == 0 {
                    0.5 * g * g
                } else {
                    0.375 * g * g
                }
            });
            let expected_max = GRADIENTS[i].map(|g| 0.5 * g * g);
            for (values, state_tensor) in [expected_m1, expected_m2, expected_max]
                .into_iter()
                .zip(state_tensors(actual_s.as_ref().unwrap()))
            {
                assert_data(
                    &TensorData::new(values.to_vec(), [2, 2]),
                    &state_tensor.to_data(),
                );
            }
        }
        check_module_record(
            &expected_optimizer,
            &expected_model,
            &expected_direct,
            step + 1,
        );
        check_module_record(&actual_optimizer, &actual_model, &actual_direct, step + 1);
        group.sync().unwrap();
        launches.assert_fallback();
        println!(
            "Group step {}: two params, six state tensors, time={}, F32 [2,2], module/direct device and values match, Reference/Foreach checkpointing={initial_strategy:?}, batch=0 single=0",
            step + 1,
            step + 1
        );

        if step == 0 {
            actual_optimizer = config(AdamWExecutionConfig::Foreach)
                .init()
                .from_bytes(actual_optimizer.into_bytes().unwrap())
                .unwrap();
        }
    }
    let (batch, single, total) = launches.counts();
    assert!(
        total > 0,
        "Group must actually execute its reference tensor operations"
    );
    println!(
        "Group observed launches: batch={batch}, single={single}, reference operations={total}; checkpoint restored between steps"
    );
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

fn canonicalize(mut captured: CapturedGraph) -> CapturedGraph {
    // Change only globally allocated IDs. Preserve operations, scalars, shapes and statuses.
    let mut ids = CanonicalIds::default();
    for operation in &mut captured.graph.operations {
        operation.visit_mut(&mut ids);
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

fn capture_steps(execution: AdamWExecutionConfig) -> (CapturedGraph, Vec<[usize; 2]>) {
    let device = Device::capture();
    assert!(matches!(
        device.as_dispatch(),
        burn::backend::DispatchDevice::Capture(_)
    ));
    let optimizer = config(execution).build();
    let mut times = Vec::new();
    let captured = device
        .capture_scope(|scope| {
            let mut pairs: Vec<(Tensor<2>, Option<AdamWState<2>>)> = PARAMETERS
                .into_iter()
                .map(|p| (tensor(p, &device), None))
                .collect();
            for step in 0..2 {
                pairs = pairs
                    .into_iter()
                    .enumerate()
                    .map(|(i, (p, s))| optimizer.step(LR, p, gradient(i, step, &device), s))
                    .collect();
                let step_times =
                    std::array::from_fn(|i| pairs[i].1.as_ref().unwrap().momentum.time);
                assert_eq!(step_times, [step + 1; 2]);
                times.push(step_times);
                for (p, state) in &pairs {
                    for output in [p]
                        .into_iter()
                        .chain(state_tensors(state.as_ref().unwrap()))
                    {
                        assert_eq!(output.device(), device);
                        assert_eq!(output.dims(), [2, 2]);
                        assert_eq!(output.dtype(), DType::F32);
                        assert!(!output.is_autodiff());
                    }
                }
            }
            // Keep all final handles alive until complete closes the capture session.
            let outputs = pairs
                .iter()
                .flat_map(|(p, state)| {
                    [p].into_iter()
                        .chain(state_tensors(state.as_ref().unwrap()))
                        .map(CaptureTensor::capture_id)
                })
                .collect::<Vec<_>>();
            assert_eq!(outputs.len(), 8);
            scope.complete([], outputs)
        })
        .expect("Device::capture() must record a complete AdamW update");
    (canonicalize(captured), times)
}

#[test]
fn capture_records_identical_reference_ir_with_all_state_outputs_and_no_launches() {
    let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    calibrate_observer();
    let (launches, _observation) = Launches::start();
    let (reference, reference_times) = capture_steps(AdamWExecutionConfig::Reference);
    let (foreach, foreach_times) = capture_steps(AdamWExecutionConfig::Foreach);
    assert_eq!(reference_times, [[1, 1], [2, 2]]);
    assert_eq!(foreach_times, reference_times);
    assert!(!reference.graph.operations.is_empty());
    assert_eq!(reference.graph.outputs.len(), 8);
    assert!(reference.graph.inputs.is_empty());
    assert_eq!(reference.graph, foreach.graph, "only tensor IDs may differ");
    assert!(!reference.values.is_empty());
    assert_eq!(reference.values.len(), foreach.values.len());
    for (id, data) in &reference.values {
        let actual = &foreach.values[id];
        assert_eq!(data.shape(), actual.shape());
        assert_eq!(data.dtype(), actual.dtype());
        assert_eq!(data.as_bytes(), actual.as_bytes());
    }
    for id in &foreach.graph.outputs {
        let output = foreach
            .graph
            .operations
            .iter()
            .flat_map(|op| op.outputs())
            .find(|tensor| tensor.id == *id)
            .expect("every declared output must be computed");
        assert_eq!(output.shape.dims(), [2, 2]);
        assert_eq!(output.dtype, DType::F32);
    }
    launches.assert_fallback();
    assert_eq!(
        launches.counts().2,
        0,
        "capture must not execute any raw kernel"
    );

    let labels = [
        "first.param",
        "first.m1",
        "first.m2",
        "first.max",
        "second.param",
        "second.m1",
        "second.m2",
        "second.max",
    ];
    let logs = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("logs");
    std::fs::create_dir_all(&logs).unwrap();
    for (name, captured, times) in [
        ("reference", &reference, &reference_times),
        ("foreach", &foreach, &foreach_times),
    ] {
        let document = serde_json::json!({
            "graph": captured.graph,
            "output_labels": labels,
            "time_after_each_step": times,
            "initial_values": captured.values.iter().map(|(id, data)| serde_json::json!({
                "id": id,
                "shape": data.shape(),
                "dtype": format!("{:?}", data.dtype()),
                "values": data.as_slice::<f32>().unwrap(),
            })).collect::<Vec<_>>(),
        });
        std::fs::write(
            logs.join(format!("capture-{name}-ir.json")),
            serde_json::to_vec_pretty(&document).unwrap(),
        )
        .unwrap();
    }
    println!(
        "Capture: {} identical normalized operations, eight outputs (param/m1/m2/max for each parameter), time=[[1,1],[2,2]], batch=0 single=0 total_raw=0",
        reference.graph.operations.len()
    );
}

#[cfg(feature = "remote")]
#[test]
fn remote_module_and_direct_steps_keep_state_and_use_reference_fallback() {
    let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    // Like burn-remote's own tests, keep a regression in a blocking read/sync from hanging CI.
    // A timed-out worker is detached and disappears when the failed test process exits.
    let limit = std::time::Duration::from_secs(30);
    let (done, completion) = std::sync::mpsc::channel();
    let worker = std::thread::spawn(move || {
        check_remote_updates();
        let _ = done.send(());
    });
    match completion.recv_timeout(limit) {
        Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
            panic!("Remote optimizer test did not finish within {limit:?}");
        }
        Ok(()) | Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
            worker
                .join()
                .unwrap_or_else(|panic| std::panic::resume_unwind(panic));
        }
    }
}

#[cfg(feature = "remote")]
fn check_remote_updates() {
    use burn::tensor::{
        remote::RemoteHost,
        server::{RemoteServer, WebSocketTransport},
    };

    calibrate_observer();
    let cpu = Device::cpu();
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let host = RemoteHost::websocket(&format!("ws://{}", listener.local_addr().unwrap()));
    // Shut down on assertion failure as well; otherwise a server task could outlive the test.
    struct Server(Option<tokio::runtime::Runtime>);
    impl Drop for Server {
        fn drop(&mut self) {
            self.0.take().unwrap().shutdown_background();
        }
    }
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.spawn(
        RemoteServer::new([cpu.clone()]).serve_async(WebSocketTransport::from_listener(listener)),
    );
    let _server = Server(Some(runtime));
    let remote = Device::remote_options(&host).init().unwrap();
    assert!(matches!(
        remote.as_dispatch(),
        burn::backend::DispatchDevice::Remote(_)
    ));
    let tracked = remote.clone().autodiff().gradient_checkpointing();
    let strategy = tracked.gradient_checkpointing_strategy().unwrap();
    assert_ne!(strategy, GradientCheckpointingStrategy::Disabled);
    let mut expected_model = Pair::new(&tracked);
    let mut actual_model = expected_model.clone();
    let mut expected_optimizer = config(AdamWExecutionConfig::Reference).init();
    let mut actual_optimizer = config(AdamWExecutionConfig::Foreach).init();
    let reference = config(AdamWExecutionConfig::Reference).build();
    let foreach = config(AdamWExecutionConfig::Foreach).build();
    let mut local_direct: Vec<_> = PARAMETERS
        .into_iter()
        .map(|p| (tensor(p, &cpu), None))
        .collect();
    let mut remote_direct: Vec<_> = PARAMETERS
        .into_iter()
        .map(|p| (tensor(p, &remote), None))
        .collect();
    cpu.sync().unwrap();
    remote.sync().unwrap();
    let (launches, _observe) = Launches::start();
    for step in 0..2 {
        let ga = expected_model.grads(&remote, step);
        let gb = actual_model.grads(&remote, step);
        expected_model = expected_optimizer.step(LR, expected_model, ga);
        actual_model = actual_optimizer.step(LR, actual_model, gb);
        local_direct = local_direct
            .into_iter()
            .enumerate()
            .map(|(i, (p, s))| reference.step(LR, p, gradient(i, step, &cpu), s))
            .collect();
        remote_direct = remote_direct
            .into_iter()
            .enumerate()
            .map(|(i, (p, s))| foreach.step(LR, p, gradient(i, step, &remote), s))
            .collect();
        for (i, (a, b)) in expected_model
            .params()
            .into_iter()
            .zip(actual_model.params())
            .enumerate()
        {
            assert_eq!(a.id, b.id);
            assert_tensor(&a.val(), &b.val(), &tracked);
            for p in [a, b] {
                assert!(p.val().is_require_grad());
                assert_eq!(p.val().gradient_checkpointing_strategy(), Some(strategy));
            }
            let (local_p, local_s) = &local_direct[i];
            let (remote_p, remote_s) = &remote_direct[i];
            assert_eq!(remote_p.device(), remote);
            assert_data(&local_p.to_data(), &remote_p.to_data());
            assert_data(&local_p.to_data(), &b.val().to_data());
            let local_s = local_s.as_ref().unwrap();
            let remote_s = remote_s.as_ref().unwrap();
            assert_eq!(remote_s.momentum.time, step + 1);
            for (expected, actual) in state_tensors(local_s)
                .into_iter()
                .zip(state_tensors(remote_s))
            {
                assert_eq!(actual.device(), remote);
                assert!(!actual.is_autodiff());
                assert_data(&expected.to_data(), &actual.to_data());
            }
            let m1 = GRADIENTS[i].map(|g| if step == 0 { 0.5 * g } else { 0.0 });
            let m2 = GRADIENTS[i].map(|g| {
                if step == 0 {
                    0.5 * g * g
                } else {
                    0.375 * g * g
                }
            });
            let maximum = GRADIENTS[i].map(|g| 0.5 * g * g);
            for (values, actual) in [m1, m2, maximum].into_iter().zip(state_tensors(remote_s)) {
                assert_data(&TensorData::new(values.to_vec(), [2, 2]), &actual.to_data());
            }
        }
        check_module_record(
            &expected_optimizer,
            &expected_model,
            &local_direct,
            step + 1,
        );
        check_module_record(&actual_optimizer, &actual_model, &local_direct, step + 1);
        remote.sync().unwrap();
        cpu.sync().unwrap();
        launches.assert_fallback();
        println!(
            "Remote step {}: loopback WebSocket to CPU, params/state agree with local Reference, time={}, context={strategy:?}, batch=0 single=0",
            step + 1,
            step + 1
        );
        if step == 0 {
            actual_optimizer = config(AdamWExecutionConfig::Foreach)
                .init()
                .from_bytes(actual_optimizer.into_bytes().unwrap())
                .unwrap();
        }
    }
    let (batch, single, total) = launches.counts();
    assert!(total > 0, "remote/local reference operations must execute");
    println!(
        "Remote totals: batch={batch}, single={single}, reference_raw_operations={total}; checkpoint restored between steps"
    );
}
