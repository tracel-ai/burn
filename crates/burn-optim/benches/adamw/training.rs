//! Complete MLP training steps, using separately built raw and Fusion binaries.
//!
//! `--training-samples <reference|fused> <tiny_mlp|large_mlp>` measures unobserved intervals.
//! `--observe-training` accepts the same arguments and counts launches without timing them.
//! Both accept `--sync-every 1|8|32`, `--warmup N` and `--samples N` (defaults: 1, 2, 3).
//! `--layers N`, `--width N` and `--batch N` override the named model's dimensions. Width must
//! be a power of two so each parameter uses one AdamW launch; microbenchmarks cover scalar tails.
//! `--verify-training-path <workload>` compares updates and serialized state, including resume.
//! Optional `--snapshot-prefix PATH` writes final Reference/Fused burnpacks for cross-build
//! comparison with `--compare-training-snapshots A B`; all floating tensors use value tolerances.

use burn::module::{Module, Param, ParamId};
use burn::store::ModuleRecord;
use burn::tensor::{Bytes, DType, Device, Tensor, TensorData, Tolerance};
use burn_core as burn;
use burn_cubecl::cubecl::logging::{LaunchObservation, LaunchObserver};
use burn_nn::Linear;
use burn_optim::{AdamWConfig, AdamWExecutionConfig, GradientsParams, ModuleOptimizer};
use burn_pack::{Reader, Writer};
use std::{
    collections::BTreeMap,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering::Relaxed},
    },
    time::Instant,
};

const LR: f64 = 0.001;
const VERIFY_STEPS: usize = 4;

#[derive(Clone, Copy)]
struct Workload {
    name: &'static str,
    layers: usize,
    width: usize,
    batch: usize,
}

impl Workload {
    fn parse(name: &str, args: &[String]) -> Self {
        let mut workload = match name {
            "tiny_mlp" => Self {
                name: "tiny_mlp",
                layers: 12,
                width: 64,
                batch: 16,
            },
            "large_mlp" => Self {
                name: "large_mlp",
                layers: 3,
                width: 1024,
                batch: 32,
            },
            _ => panic!("expected tiny_mlp or large_mlp"),
        };
        workload.layers = number(args, "--layers", workload.layers);
        workload.width = number(args, "--width", workload.width);
        workload.batch = number(args, "--batch", workload.batch);
        assert!(workload.layers > 0 && workload.width > 0 && workload.batch > 0);
        assert!(
            workload.width.is_power_of_two(),
            "training width must be a power of two for one-launch-per-parameter checks"
        );
        workload
    }
}

#[derive(Module, Debug)]
struct Mlp {
    layers: Vec<Linear>,
}

// Integer arithmetic makes the initial weights and data identical across devices and binaries.
fn values(len: usize, seed: u32, scale: f32) -> Vec<f32> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((state >> 8) as f32 / 8_388_608.0 - 1.0) * scale
        })
        .collect()
}

fn parameter<const D: usize>(
    data: Vec<f32>,
    shape: [usize; D],
    id: u64,
    device: &Device,
) -> Param<Tensor<D>> {
    let mut param = Param::from_tensor(Tensor::from_data(TensorData::new(data, shape), device));
    param.id = ParamId::from(id);
    param
}

impl Mlp {
    fn new(workload: Workload, device: &Device) -> Self {
        let width = workload.width;
        let layers = (0..workload.layers)
            .map(|index| Linear {
                weight: parameter(
                    values(
                        width * width,
                        17 + index as u32,
                        0.5 / (width as f32).sqrt(),
                    ),
                    [width, width],
                    (2 * index + 1) as u64,
                    device,
                ),
                bias: Some(parameter(
                    values(width, 7919 + index as u32, 0.01),
                    [width],
                    (2 * index + 2) as u64,
                    device,
                )),
            })
            .collect();
        Self { layers }
    }

    fn forward(&self, mut input: Tensor<2>) -> Tensor<2> {
        for layer in &self.layers {
            // A residual preserves useful gradients through the many-small-parameter workload.
            input = input.clone() + layer.forward(input).tanh() * 0.1;
        }
        input
    }
}

fn optimizer(fused: bool, amsgrad: bool) -> ModuleOptimizer {
    AdamWConfig::new()
        .with_weight_decay(0.0001)
        .with_amsgrad(amsgrad)
        .with_execution(Some(if fused {
            AdamWExecutionConfig::Fused
        } else {
            AdamWExecutionConfig::Reference
        }))
        .init()
}

struct Training {
    model: Option<Mlp>,
    optimizer: ModuleOptimizer,
    input: Tensor<2>,
    target: Tensor<2>,
}

impl Training {
    fn new(workload: Workload, fused: bool, amsgrad: bool, device: &Device) -> Self {
        let tracked = device.clone().autodiff();
        let shape = [workload.batch, workload.width];
        let len = shape[0] * shape[1];
        Self {
            model: Some(Mlp::new(workload, &tracked)),
            optimizer: optimizer(fused, amsgrad),
            input: Tensor::from_data(TensorData::new(values(len, 123, 0.5), shape), &tracked),
            target: Tensor::from_data(TensorData::new(values(len, 456, 0.25), shape), &tracked),
        }
    }

    fn step(&mut self) -> Tensor<1> {
        let model = self.model.take().unwrap();
        let loss = (model.forward(self.input.clone()) - self.target.clone())
            .square()
            .mean();
        let grads = GradientsParams::from_grads(loss.clone().backward(), &model);
        self.model = Some(self.optimizer.step(LR, model, grads));
        loss
    }

    fn resume(&mut self, workload: Workload, fused: bool, amsgrad: bool, device: &Device) {
        let model = self
            .model
            .take()
            .unwrap()
            .into_record()
            .into_bytes()
            .unwrap();
        let state = self.optimizer.into_bytes().unwrap();
        self.model = Some(
            Mlp::new(workload, &device.clone().autodiff())
                .load_record(ModuleRecord::from_bytes(model).unwrap()),
        );
        self.optimizer = optimizer(fused, amsgrad).from_bytes(state).unwrap();
    }

    fn snapshot(&self, losses: &[f32], steps: usize, amsgrad: bool) -> Bytes {
        let model = Reader::from_bytes(
            self.model
                .as_ref()
                .unwrap()
                .clone()
                .into_record()
                .into_bytes()
                .unwrap(),
        )
        .unwrap();
        let state = Reader::from_bytes(self.optimizer.into_bytes().unwrap()).unwrap();
        let params = self.model.as_ref().unwrap().layers.len() * 2;
        assert_eq!(
            state.metadata().len(),
            params,
            "every parameter must have optimizer state"
        );
        for id in 1..=params {
            assert_eq!(
                usize::try_from(state.scalars()[&format!("{id}.momentum.time")]).unwrap(),
                steps
            );
        }
        let scalars = state.scalars().clone();
        let paths = state.metadata().clone();
        let mut tensors = model.into_tensors().unwrap();
        assert_eq!(tensors.len(), params);
        let moments = state.into_tensors().unwrap();
        assert_eq!(moments.len(), params * if amsgrad { 3 } else { 2 });
        tensors.extend(moments);
        let data = TensorData::new(losses.to_vec(), [losses.len()]);
        let (bytes, shape, dtype) = data.into_parts();
        tensors.push(burn_pack::Tensor::new(
            "losses".into(),
            dtype,
            shape,
            None,
            bytes,
        ));
        let mut writer = Writer::new(tensors);
        for (key, scalar) in scalars {
            writer = writer.with_scalar(&key, scalar);
        }
        for (key, path) in paths {
            writer = writer.with_metadata(&key, &path);
        }
        writer.into_bytes().unwrap()
    }
}

fn compare(expected: Reader, actual: Reader) {
    assert_eq!(
        expected.scalars(),
        actual.scalars(),
        "optimizer counters/ranks"
    );
    assert_eq!(expected.metadata(), actual.metadata(), "parameter paths");
    let collect = |reader: Reader| {
        reader
            .into_tensors()
            .unwrap()
            .into_iter()
            .map(|tensor| {
                let (name, dtype, shape, id, bytes) = tensor.into_parts().unwrap();
                (name, (id, TensorData::from_bytes(bytes, shape, dtype)))
            })
            .collect::<BTreeMap<_, _>>()
    };
    let expected = collect(expected);
    let actual = collect(actual);
    assert_eq!(
        expected.keys().collect::<Vec<_>>(),
        actual.keys().collect::<Vec<_>>()
    );
    for (name, (id, value)) in expected {
        let (actual_id, actual) = &actual[&name];
        assert_eq!(&id, actual_id, "{name}: parameter identity");
        assert_eq!(value.dtype(), DType::F32, "{name}: expected F32");
        assert_eq!(actual.dtype(), value.dtype());
        assert_eq!(actual.shape(), value.shape());
        assert!(
            value
                .as_slice::<f32>()
                .unwrap()
                .iter()
                .chain(actual.as_slice::<f32>().unwrap())
                .all(|x| x.is_finite()),
            "{name}: nonfinite training values"
        );
        value.assert_approx_eq::<f32>(actual, Tolerance::rel_abs(2e-4, 2e-6));
    }
}

#[derive(Default)]
struct Launches {
    total: AtomicUsize,
    adamw: AtomicUsize,
}

impl LaunchObserver for Launches {
    fn launched(&self, name: &'static str) {
        self.total.fetch_add(1, Relaxed);
        if name.contains("optim::adamw::eager::adamw::") {
            self.adamw.fetch_add(1, Relaxed);
        }
    }
}

fn mode(fused: bool) -> &'static str {
    match (burn_cubecl::FUSED, fused) {
        (false, false) => "raw_reference",
        (false, true) => "raw_fused",
        (true, false) => "fusion_reference",
        (true, true) => "fusion_fused",
    }
}

fn device() -> Device {
    let device = Device::default();
    assert!(
        matches!(device.as_dispatch(), burn::backend::DispatchDevice::Cube(_)),
        "training comparison requires a CubeCL runtime; set BURN_DEVICE=cpu or cuda"
    );
    device
}

fn verify(workload: Workload, snapshot_prefix: Option<&str>) {
    let device = device();
    for amsgrad in [false, true] {
        let mut reference = Training::new(workload, false, amsgrad, &device);
        let mut fused = Training::new(workload, true, amsgrad, &device);
        let mut reference_losses = Vec::new();
        let mut fused_losses = Vec::new();
        for step in 1..=VERIFY_STEPS {
            let reference_counts = Arc::new(Launches::default());
            let reference_observation = LaunchObservation::new(reference_counts.clone());
            let expected = reference.step();
            // Finish each mode's lazy work inside its own observation window.
            reference_losses.push(expected.into_scalar());
            device.sync().unwrap();
            drop(reference_observation);
            assert_eq!(
                reference_counts.adamw.load(Relaxed),
                0,
                "Reference dispatched AdamW"
            );
            let counts = Arc::new(Launches::default());
            let observation = LaunchObservation::new(counts.clone());
            fused_losses.push(fused.step().into_scalar());
            device.sync().unwrap();
            drop(observation);
            assert_eq!(
                counts.adamw.load(Relaxed),
                workload.layers * 2,
                "{} must dispatch AdamW for every parameter",
                mode(true)
            );
            let expected = reference.snapshot(&reference_losses, step, amsgrad);
            let actual = fused.snapshot(&fused_losses, step, amsgrad);
            compare(
                Reader::from_bytes(expected.clone()).unwrap(),
                Reader::from_bytes(actual.clone()).unwrap(),
            );
            if step == 2 {
                // Only the candidate resumes; the reference continues without a checkpoint.
                fused.resume(workload, true, amsgrad, &device);
                compare(
                    Reader::from_bytes(actual).unwrap(),
                    Reader::from_bytes(fused.snapshot(&fused_losses, step, amsgrad)).unwrap(),
                );
            } else if step == VERIFY_STEPS
                && let Some(prefix) = snapshot_prefix
            {
                for (fused, bytes) in [(false, expected), (true, actual)] {
                    std::fs::write(
                        format!("{prefix}-{}-amsgrad{amsgrad}.bpk", mode(fused)),
                        &bytes[..],
                    )
                    .unwrap();
                }
            }
        }
        println!(
            "{{\"kind\":\"training_verified\",\"workload\":\"{}\",\"mode\":\"{}\",\"steps\":{VERIFY_STEPS},\"amsgrad\":{amsgrad},\"reference_losses\":{reference_losses:?},\"fused_losses\":{fused_losses:?}}}",
            workload.name,
            mode(true)
        );
    }
}

fn memory(device: &Device) {
    match device.measure_memory() {
        Some(report) => {
            let peak_upper_bound: u64 = report
                .pools
                .iter()
                .map(|pool| pool.page_size * pool.pages_peak)
                .sum();
            println!(
                "{{\"kind\":\"training_memory\",\"allocations\":{},\"in_use_bytes\":{},\"reserved_bytes\":{},\"pool_peak_upper_bound_bytes\":{peak_upper_bound}}}",
                report.usage.number_allocs, report.usage.bytes_in_use, report.usage.bytes_reserved
            );
        }
        None => println!("{{\"kind\":\"training_memory\",\"available\":false}}"),
    }
}

fn measure(
    workload: Workload,
    fused: bool,
    sync_every: usize,
    warmup: usize,
    samples: usize,
    observe: bool,
) {
    let device = device();
    let mut training = Training::new(workload, fused, false, &device);
    for _ in 0..warmup {
        for _ in 0..sync_every {
            drop(training.step());
        }
        device.sync().unwrap();
    }
    device.sync().unwrap();
    println!(
        "{{\"kind\":\"training_config\",\"workload\":\"{}\",\"mode\":\"{}\",\"layers\":{},\"width\":{},\"batch\":{},\"parameters\":{},\"sync_every\":{sync_every},\"warmup_blocks\":{warmup},\"sample_blocks\":{samples},\"warmup_steps\":{},\"measured_steps\":{},\"observed\":{observe},\"jit_excluded\":true}}",
        workload.name,
        mode(fused),
        workload.layers,
        workload.width,
        workload.batch,
        workload.layers * 2,
        warmup * sync_every,
        samples * sync_every
    );
    let counts = Arc::new(Launches::default());
    let observation = observe.then(|| LaunchObservation::new(counts.clone()));
    for sample in 0..samples {
        let start = Instant::now();
        for _ in 0..sync_every {
            drop(training.step());
        }
        let enqueue = start.elapsed();
        device.sync().unwrap();
        let total = start.elapsed();
        if !observe {
            println!(
                "{{\"kind\":\"training_sample\",\"sample\":{sample},\"steps\":{sync_every},\"enqueue_ms\":{:.6},\"total_ms\":{:.6},\"per_step_ms\":{:.6}}}",
                enqueue.as_secs_f64() * 1000.0,
                total.as_secs_f64() * 1000.0,
                total.as_secs_f64() * 1000.0 / sync_every as f64
            );
        }
    }
    drop(observation);
    if observe {
        let adamw = counts.adamw.load(Relaxed);
        if fused {
            assert_eq!(
                adamw,
                workload.layers * 2 * samples * sync_every,
                "each parameter update must reach specialized AdamW"
            );
        } else {
            assert_eq!(adamw, 0, "Reference submitted specialized AdamW");
        }
        println!(
            "{{\"kind\":\"training_launches\",\"total\":{},\"adamw\":{adamw},\"steps\":{}}}",
            counts.total.load(Relaxed),
            samples * sync_every
        );
    }
    memory(&device);
}

fn option<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.iter()
        .position(|arg| arg == name)
        .map(|index| args.get(index + 1).expect("option needs a value").as_str())
}

fn number(args: &[String], name: &str, default: usize) -> usize {
    option(args, name)
        .map(|value| value.parse().expect("expected positive integer"))
        .unwrap_or(default)
}

pub(super) fn run(args: &[String]) -> bool {
    if let Some(index) = args
        .iter()
        .position(|arg| arg == "--compare-training-snapshots")
    {
        compare(
            Reader::from_file(&args[index + 1]).unwrap(),
            Reader::from_file(&args[index + 2]).unwrap(),
        );
        println!("training snapshots match");
        return true;
    }
    if let Some(name) = option(args, "--verify-training-path") {
        verify(
            Workload::parse(name, args),
            option(args, "--snapshot-prefix"),
        );
        return true;
    }
    let Some((index, flag)) = args
        .iter()
        .enumerate()
        .find(|(_, arg)| matches!(arg.as_str(), "--training-samples" | "--observe-training"))
    else {
        return false;
    };
    let fused = match args
        .get(index + 1)
        .expect("expected execution mode")
        .as_str()
    {
        "reference" => false,
        "fused" => true,
        _ => panic!("expected reference or fused"),
    };
    let workload = Workload::parse(args.get(index + 2).expect("expected workload"), args);
    let sync_every = number(args, "--sync-every", 1);
    let warmup = number(args, "--warmup", 2);
    let samples = number(args, "--samples", 3);
    assert!(
        [1, 8, 32].contains(&sync_every),
        "sync-every must be 1, 8 or 32"
    );
    assert!(
        warmup > 0 && samples > 0,
        "warmup and samples must be positive"
    );
    assert!(
        warmup * sync_every >= 2,
        "warm up initial and stateful updates before timing"
    );
    measure(
        workload,
        fused,
        sync_every,
        warmup,
        samples,
        flag == "--observe-training",
    );
    true
}
