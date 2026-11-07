use burn::module::{Param, ParamGroup};
use burn::tensor::{Device, Tensor};
use burn_core as burn;
use burn_cubecl::cubecl::logging::{LaunchObservation, LaunchObserver};
use burn_optim::lr_scheduler::module_lr_scheduler::{ModuleLearningRate, ModuleLrScheduler};
use burn_optim::{AdamWConfig, AdamWExecutionConfig, GradientsParams};
use std::{
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::Instant,
};

#[derive(burn_derive::Module, Debug)]
struct Model {
    vectors: Vec<Param<Tensor<1>>>,
    matrices: Vec<Param<Tensor<2>>>,
    conv: Vec<Param<Tensor<4>>>,
}

impl Model {
    fn new(workload: &str, device: &Device) -> Self {
        let (small, large, conv) = match workload {
            "tiny" => (30, 0, 0),
            "mixed_rank" => (0, 1, 1),
            "large_1" => (0, 1, 0),
            "near_limit" => (8, 0, 0),
            "large_4" => (0, 4, 0),
            "large_16" => (0, 16, 0),
            "long_short" => (30, 2, 0),
            "over_limit" | "fragmented_lr" => (80, 0, 0),
            _ => panic!("unknown module workload"),
        };
        let dim = if workload == "mixed_rank" { 64 } else { 1024 };
        let small_shape = if workload == "near_limit" {
            [512, 256]
        } else {
            [64, 64]
        };
        Self {
            vectors: (0..usize::from(workload == "mixed_rank"))
                .map(|_| Param::from_tensor(Tensor::ones([64], device)))
                .collect(),
            matrices: (0..small)
                .map(|_| Param::from_tensor(Tensor::ones(small_shape, device)))
                .chain((0..large).map(|_| Param::from_tensor(Tensor::ones([dim, dim], device))))
                .collect(),
            conv: (0..conv)
                .map(|_| Param::from_tensor(Tensor::ones([16, 8, 3, 3], device)))
                .collect(),
        }
    }

    fn learning_rates(&self, workload: &str) -> ModuleLearningRate {
        if workload != "fragmented_lr" {
            return 0.001.into();
        }
        let mut scheduler = ModuleLrScheduler::new(0.001);
        for (i, parameter) in self.matrices.iter().enumerate() {
            scheduler = scheduler.with_group(
                ParamGroup::from_ids(vec![parameter.id]),
                0.001 + i as f64 * 0.00001,
            );
        }
        scheduler.step()
    }

    fn grads(&self, device: &Device) -> GradientsParams {
        let mut grads = GradientsParams::new();
        for param in &self.vectors {
            grads.register(
                param.id,
                Tensor::<1>::full(param.val().shape(), 0.01, device),
            );
        }
        for param in &self.matrices {
            grads.register(
                param.id,
                Tensor::<2>::full(param.val().shape(), 0.01, device),
            );
        }
        for param in &self.conv {
            grads.register(
                param.id,
                Tensor::<4>::full(param.val().shape(), 0.01, device),
            );
        }
        grads
    }
}

#[derive(Default)]
struct Counts {
    batch: AtomicUsize,
    single: AtomicUsize,
}
impl LaunchObserver for Counts {
    fn launched(&self, name: &'static str) {
        if name.contains("adamw_foreach::") {
            self.batch.fetch_add(1, Ordering::Relaxed);
        }
        if name.contains("optim::adamw::eager::adamw::") {
            self.single.fetch_add(1, Ordering::Relaxed);
        }
    }
}

fn assert_path(counts: &Counts, mode: &str, workload: &str, model: &Model, raw: &Device) {
    let on_cpu = matches!(raw.as_dispatch(), burn::backend::DispatchDevice::Cube(device)
        if device.runtime() == burn_cubecl::cubecl::RuntimeId::Cpu);
    let singles = model.vectors.len() + model.matrices.len() + model.conv.len();
    let expected = if mode == "foreach" && on_cpu {
        match workload {
            "tiny" => (4, 0),
            "mixed_rank" => (1, 0),
            "long_short" => (4, 2),
            "over_limit" => (10, 0),
            "fragmented_lr" => (0, 80),
            _ => (0, singles),
        }
    } else {
        (0, singles)
    };
    assert_eq!(
        (
            counts.batch.load(Ordering::Relaxed),
            counts.single.load(Ordering::Relaxed)
        ),
        expected,
        "{mode} {workload} did not use the expected optimizer path"
    );
}

pub fn run(mode: &str, workload: &str) {
    if let Some(mode) = mode.strip_prefix("profile-") {
        profile(mode, workload);
        return;
    }
    if mode == "paired" {
        paired(workload);
        return;
    }
    let execution = match mode {
        "fused" => AdamWExecutionConfig::Fused,
        "foreach" => AdamWExecutionConfig::Foreach,
        _ => panic!("expected fused or foreach"),
    };
    let raw = Device::default();
    let device = raw.clone().autodiff();
    let mut model = Model::new(workload, &device);
    let learning_rates = model.learning_rates(workload);
    let mut optim = AdamWConfig::new().with_execution(Some(execution)).init();
    let grads = model.grads(&raw);
    raw.sync().unwrap();
    let start = Instant::now();
    model = optim.step(learning_rates.clone(), model, grads);
    raw.sync().unwrap();
    println!(
        "{{\"phase\":\"first\",\"mode\":\"{mode}\",\"workload\":\"{workload}\",\"total_ms\":{}}}",
        start.elapsed().as_secs_f64() * 1e3
    );
    for _ in 0..5 {
        let grads = model.grads(&raw);
        model = optim.step(learning_rates.clone(), model, grads);
        raw.sync().unwrap();
    }
    if let Some(report) = raw.measure_memory() {
        let peak: u64 = report
            .pools
            .iter()
            .map(|pool| pool.page_size * pool.pages_peak)
            .sum();
        println!(
            "{{\"phase\":\"memory\",\"allocations\":{},\"in_use_bytes\":{},\"reserved_bytes\":{},\"sum_pool_peaks_bytes\":{peak}}}",
            report.usage.number_allocs, report.usage.bytes_in_use, report.usage.bytes_reserved
        );
    }
    let counts = Arc::new(Counts::default());
    {
        let _observe = LaunchObservation::new(counts.clone());
        let grads = model.grads(&raw);
        model = optim.step(learning_rates.clone(), model, grads);
        raw.sync().unwrap();
    }
    println!(
        "{{\"phase\":\"path\",\"mode\":\"{mode}\",\"workload\":\"{workload}\",\"batch\":{},\"single\":{}}}",
        counts.batch.load(Ordering::Relaxed),
        counts.single.load(Ordering::Relaxed)
    );
    assert_path(&counts, mode, workload, &model, &raw);
    for sample in 0..30 {
        let grads = model.grads(&raw);
        raw.sync().unwrap();
        let start = Instant::now();
        model = optim.step(learning_rates.clone(), model, grads);
        let enqueue = start.elapsed();
        raw.sync().unwrap();
        let total = start.elapsed();
        println!(
            "{{\"phase\":\"steady\",\"mode\":\"{mode}\",\"workload\":\"{workload}\",\"sample\":{sample},\"enqueue_ms\":{},\"total_ms\":{}}}",
            enqueue.as_secs_f64() * 1e3,
            total.as_secs_f64() * 1e3
        );
    }
    let start = Instant::now();
    drop((model, optim));
    raw.sync().unwrap();
    println!(
        "{{\"phase\":\"drop\",\"total_ms\":{}}}",
        start.elapsed().as_secs_f64() * 1e3
    );
}

/// Separate instrumented windows: profile markers can perturb timing, so these samples must not
/// be mixed with the uninstrumented paired throughput comparison.
fn profile(mode: &str, workload: &str) {
    let execution = match mode {
        "fused" => AdamWExecutionConfig::Fused,
        "foreach" => AdamWExecutionConfig::Foreach,
        _ => panic!("expected profile-fused or profile-foreach"),
    };
    let raw = Device::default();
    let device = raw.clone().autodiff();
    let mut model = Model::new(workload, &device);
    let learning_rates = model.learning_rates(workload);
    let mut optim = AdamWConfig::new().with_execution(Some(execution)).init();
    for _ in 0..8 {
        let grads = model.grads(&raw);
        model = optim.step(learning_rates.clone(), model, grads);
        raw.sync().unwrap();
    }
    for sample in 0..20 {
        let grads = model.grads(&raw);
        raw.sync().unwrap();
        let start = Instant::now();
        let (updated, duration) = raw
            .profile(|| optim.step(learning_rates.clone(), model, grads))
            .unwrap();
        let method = duration.timing_method();
        let ticks = burn::tensor::read_sync(duration.resolve())
            .expect("optimizer window must contain work");
        model = updated;
        println!(
            "{{\"phase\":\"device_window\",\"mode\":\"{mode}\",\"workload\":\"{workload}\",\"sample\":{sample},\"method\":\"{method}\",\"window_ms\":{},\"host_ms\":{}}}",
            ticks.duration().as_secs_f64() * 1e3,
            start.elapsed().as_secs_f64() * 1e3
        );
    }
}

fn paired(workload: &str) {
    let raw = Device::default();
    let device = raw.clone().autodiff();
    let mut models = [
        Some(Model::new(workload, &device)),
        Some(Model::new(workload, &device)),
    ];
    let learning_rates = models
        .each_ref()
        .map(|model| model.as_ref().unwrap().learning_rates(workload));
    let mut optimizers = [
        AdamWConfig::new()
            .with_execution(Some(AdamWExecutionConfig::Fused))
            .init(),
        AdamWConfig::new()
            .with_execution(Some(AdamWExecutionConfig::Foreach))
            .init(),
    ];
    for _ in 0..8 {
        for i in 0..2 {
            let model = models[i].take().unwrap();
            let grads = model.grads(&raw);
            models[i] = Some(optimizers[i].step(learning_rates[i].clone(), model, grads));
            raw.sync().unwrap();
        }
    }
    // Verify this exact process uses the measured paths before timing. A paired run must
    // fail if batching silently falls back, even when a separate verification run passed.
    for (i, mode) in ["fused", "foreach"].into_iter().enumerate() {
        let counts = Arc::new(Counts::default());
        let _observe = LaunchObservation::new(counts.clone());
        let model = models[i].take().unwrap();
        let grads = model.grads(&raw);
        let model = optimizers[i].step(learning_rates[i].clone(), model, grads);
        raw.sync().unwrap();
        assert_path(&counts, mode, workload, &model, &raw);
        models[i] = Some(model);
    }
    for pair in 0..40 {
        for i in if pair % 2 == 0 { [0, 1] } else { [1, 0] } {
            let model = models[i].take().unwrap();
            let grads = model.grads(&raw);
            raw.sync().unwrap();
            let start = Instant::now();
            models[i] = Some(optimizers[i].step(learning_rates[i].clone(), model, grads));
            let enqueue = start.elapsed();
            raw.sync().unwrap();
            let total = start.elapsed();
            let mode = if i == 0 { "fused" } else { "foreach" };
            println!(
                "{{\"phase\":\"paired\",\"mode\":\"{mode}\",\"workload\":\"{workload}\",\"pair\":{pair},\"enqueue_ms\":{},\"total_ms\":{}}}",
                enqueue.as_secs_f64() * 1e3,
                total.as_secs_f64() * 1e3
            );
        }
    }
}
