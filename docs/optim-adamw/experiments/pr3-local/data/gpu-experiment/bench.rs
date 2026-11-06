use burn::module::{Module, ModuleMapper, Param};
use burn::tensor::{Device, Tensor};
use burn_core as burn;
use burn_cubecl::cubecl::logging::{LaunchObservation, LaunchObserver};
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
            "large_4" => (0, 4, 0),
            "large_16" => (0, 16, 0),
            "long_short" => (30, 2, 0),
            "over_limit" => (80, 0, 0),
            _ => panic!("unknown module workload"),
        };
        let dim = if workload == "mixed_rank" { 64 } else { 1024 };
        Self {
            vectors: (0..usize::from(workload == "mixed_rank"))
                .map(|_| Param::from_tensor(Tensor::ones([64], device)))
                .collect(),
            matrices: (0..small)
                .map(|_| Param::from_tensor(Tensor::ones([64, 64], device)))
                .chain((0..large).map(|_| Param::from_tensor(Tensor::ones([dim, dim], device))))
                .collect(),
            conv: (0..conv)
                .map(|_| Param::from_tensor(Tensor::ones([16, 8, 3, 3], device)))
                .collect(),
        }
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

struct Identity;
impl ModuleMapper for Identity {}

pub fn run(mode: &str, workload: &str) {
    let execution = match mode {
        "fused" => AdamWExecutionConfig::Fused,
        "foreach" => AdamWExecutionConfig::Foreach,
        _ => panic!("expected fused or foreach"),
    };
    let raw = Device::default();
    let device = raw.clone().autodiff();
    let mut model = Model::new(workload, &device);
    let mut optim = AdamWConfig::new().with_execution(Some(execution)).init();
    let grads = model.grads(&raw);
    raw.sync().unwrap();
    let start = Instant::now();
    model = optim.step(0.001, model, grads);
    raw.sync().unwrap();
    println!(
        "{{\"phase\":\"first\",\"mode\":\"{mode}\",\"workload\":\"{workload}\",\"total_ms\":{}}}",
        start.elapsed().as_secs_f64() * 1e3
    );
    for _ in 0..5 {
        let grads = model.grads(&raw);
        model = optim.step(0.001, model, grads);
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
        model = optim.step(0.001, model, grads);
        raw.sync().unwrap();
    }
    println!(
        "{{\"phase\":\"path\",\"mode\":\"{mode}\",\"workload\":\"{workload}\",\"batch\":{},\"single\":{}}}",
        counts.batch.load(Ordering::Relaxed),
        counts.single.load(Ordering::Relaxed)
    );
    assert!(counts.batch.load(Ordering::Relaxed) + counts.single.load(Ordering::Relaxed) > 0);
    for sample in 0..30 {
        let grads = model.grads(&raw);
        raw.sync().unwrap();
        let start = Instant::now();
        model = optim.step(0.001, model, grads);
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
    for _ in 0..100 {
        model = model.map(&mut Identity).map(&mut Identity);
    }
    let maps = start.elapsed();
    println!(
        "{{\"phase\":\"identity_maps\",\"total_ms\":{},\"iterations\":100}}",
        maps.as_secs_f64() * 1e3
    );
    let start = Instant::now();
    drop((model, optim));
    raw.sync().unwrap();
    println!(
        "{{\"phase\":\"drop\",\"total_ms\":{}}}",
        start.elapsed().as_secs_f64() * 1e3
    );
}
