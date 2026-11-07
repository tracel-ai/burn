//! AdamW per-parameter optimizer-step benchmark.
//!
//! Measures the reference and fused execution on the same raw CubeCL runtime:
//!
//! ```sh
//! cargo bench -p burn-optim --bench adamw --no-default-features \
//!     --features std,cpu
//! ```
//!
//! Kernel fusion makes every fused step decline and run the reference implementation, so a
//! fusion build would report reference numbers under the fused name; [`main`] refuses it.
//!
//! `--verify-path` runs every workload once in both execution modes and checks the fused result
//! against the reference before any measurement.
//! The `unaligned` case offsets only the fused gradient; its reference uses identical values in
//! an aligned allocation because generic reference kernels require aligned buffer bindings.
//!
//! `--measure-memory <reference|fused> <workload>` runs the workload and prints the device
//! allocator's live and peak usage. Peak pages are per process, so run each mode separately.

#[cfg(cubecl_backend)]
#[path = "../src/optim/adamw/test_support.rs"]
mod test_support;

#[cfg(cubecl_backend)]
#[path = "adamw/foreach.rs"]
mod foreach;

mod bench {
    use std::fmt;

    #[cfg(cubecl_backend)]
    use super::test_support::KernelCounter;
    use burn_core::tensor::{Device, Distribution, Tensor};
    #[cfg(cubecl_backend)]
    use burn_core::tensor::{TensorData, Tolerance};
    use burn_optim::{AdamW, AdamWConfig, AdamWExecutionConfig, AdamWState, Optimizer};

    const LR: f64 = 1e-3;
    const STEPS: usize = 3;

    /// One parameter distribution of a workload.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum Workload {
        /// 30 parameters of `[64, 64]`: host and launch bound.
        Tiny,
        /// A bias `[64]`, a linear `[64, 64]` and a convolution `[16, 8, 3, 3]`.
        MixedRank,
        /// One `[1024, 1024]` parameter: scalar-kernel throughput.
        Large1,
        /// Four `[1024, 1024]` parameters.
        Large4,
        /// Sixteen `[1024, 1024]` parameters.
        Large16,
        /// Thirty small matrices and two large ones.
        LongShort,
        /// One large vector with one element below a multiple of the vector width.
        TailMinus,
        /// One large vector with one element above a multiple of the vector width.
        TailPlus,
        /// One large vector whose fused gradient has an offset binding.
        Unaligned,
    }

    impl Workload {
        /// The `[1024, 1024]`-shaped parameters of a workload.
        fn large(self) -> usize {
            match self {
                Workload::Large1 => 1,
                Workload::Large4 => 4,
                Workload::Large16 => 16,
                Workload::LongShort => 2,
                Workload::Tiny
                | Workload::MixedRank
                | Workload::TailMinus
                | Workload::TailPlus
                | Workload::Unaligned => 0,
            }
        }

        /// The `[64, 64]`-shaped parameters of a workload.
        fn small(self) -> usize {
            match self {
                Workload::Tiny | Workload::LongShort => 30,
                _ => 0,
            }
        }

        /// The workload named on the command line.
        pub fn from_name(name: &str) -> Option<Self> {
            WORKLOADS.iter().copied().find(|w| w.to_string() == name)
        }
    }

    impl fmt::Display for Workload {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            let name = match self {
                Workload::Tiny => "tiny",
                Workload::MixedRank => "mixed_rank",
                Workload::Large1 => "large_1",
                Workload::Large4 => "large_4",
                Workload::Large16 => "large_16",
                Workload::LongShort => "long_short",
                Workload::TailMinus => "tail_minus",
                Workload::TailPlus => "tail_plus",
                Workload::Unaligned => "unaligned",
            };
            f.write_str(name)
        }
    }

    pub const WORKLOADS: [Workload; 9] = [
        Workload::Tiny,
        Workload::MixedRank,
        Workload::Large1,
        Workload::Large4,
        Workload::Large16,
        Workload::LongShort,
        Workload::TailMinus,
        Workload::TailPlus,
        Workload::Unaligned,
    ];

    fn optimizer(fused: bool) -> AdamW {
        AdamWConfig::new()
            .with_weight_decay(1e-4)
            .with_execution(fused.then_some(AdamWExecutionConfig::Fused))
            .build()
    }

    fn param<const D: usize>(shape: [usize; D], device: &Device, varied: bool) -> Tensor<D> {
        if varied {
            Tensor::random(shape, Distribution::Default, device)
        } else {
            Tensor::ones(shape, device)
        }
    }

    fn grad<const D: usize>(shape: [usize; D], device: &Device, varied: bool) -> Tensor<D> {
        if varied {
            Tensor::random(shape, Distribution::Uniform(-1.0, 1.0), device)
        } else {
            Tensor::full(shape, 0.01, device)
        }
    }

    /// A `[dim, dim]` parameter and its state, stepped in place.
    #[derive(Clone)]
    struct Square {
        param: Option<Tensor<2>>,
        grad: Tensor<2>,
        state: Option<AdamWState<2>>,
    }

    fn step_slot<const D: usize>(
        param: &mut Option<Tensor<D>>,
        grad: &Tensor<D>,
        state: &mut Option<AdamWState<D>>,
        optimizer: &AdamW,
    ) {
        let (updated, next) = optimizer.step(LR, param.take().unwrap(), grad.clone(), state.take());
        *param = Some(updated);
        *state = next;
    }

    impl Square {
        fn new(dim: usize, device: &Device, varied: bool) -> Self {
            Self::from_shape([dim, dim], device, varied)
        }

        fn from_shape(shape: [usize; 2], device: &Device, varied: bool) -> Self {
            Self {
                param: Some(param(shape, device, varied)),
                grad: grad(shape, device, varied),
                state: None,
            }
        }

        fn step(&mut self, optimizer: &AdamW) {
            step_slot(&mut self.param, &self.grad, &mut self.state, optimizer);
        }
    }

    /// A mixed-rank parameter and its state, stepped in place.
    #[derive(Clone)]
    enum Mixed {
        Bias(Option<Tensor<1>>, Tensor<1>, Option<AdamWState<1>>),
        Linear(Option<Tensor<2>>, Tensor<2>, Option<AdamWState<2>>),
        Conv(Option<Tensor<4>>, Tensor<4>, Option<AdamWState<4>>),
    }

    impl Mixed {
        fn all(device: &Device, varied: bool) -> [Self; 3] {
            [
                Self::Bias(
                    Some(param([64], device, varied)),
                    grad([64], device, varied),
                    None,
                ),
                Self::Linear(
                    Some(param([64, 64], device, varied)),
                    grad([64, 64], device, varied),
                    None,
                ),
                Self::Conv(
                    Some(param([16, 8, 3, 3], device, varied)),
                    grad([16, 8, 3, 3], device, varied),
                    None,
                ),
            ]
        }

        fn step(&mut self, optimizer: &AdamW) {
            match self {
                Mixed::Bias(param, grad, state) => step_slot(param, grad, state, optimizer),
                Mixed::Linear(param, grad, state) => step_slot(param, grad, state, optimizer),
                Mixed::Conv(param, grad, state) => step_slot(param, grad, state, optimizer),
            }
        }

        #[cfg(cubecl_backend)]
        fn tensors(&self) -> Vec<TensorData> {
            match self {
                Mixed::Bias(param, _, state) => state_tensors(param, state),
                Mixed::Linear(param, _, state) => state_tensors(param, state),
                Mixed::Conv(param, _, state) => state_tensors(param, state),
            }
        }
    }

    /// A workload's live parameters and optimizer state.
    #[derive(Clone)]
    enum Case {
        Squares(Vec<Square>),
        Mixed(Box<[Mixed; 3]>),
    }

    /// A parameter's own value followed by its state, in a stable order.
    #[cfg(cubecl_backend)]
    fn state_tensors<const D: usize>(
        param: &Option<Tensor<D>>,
        state: &Option<AdamWState<D>>,
    ) -> Vec<TensorData> {
        let mut tensors = vec![param.as_ref().unwrap().to_data()];
        if let Some(state) = state {
            tensors.push(state.momentum.moment_1.to_data());
            tensors.push(state.momentum.moment_2.to_data());
            if let Some(max) = &state.momentum.max_moment_2 {
                tensors.push(max.to_data());
            }
        }
        tensors
    }

    #[cfg(cubecl_backend)]
    fn allocation_ids<const D: usize>(
        param: &Option<Tensor<D>>,
        state: &Option<AdamWState<D>>,
    ) -> Vec<usize> {
        let id = |tensor: &Tensor<D>| {
            let primitive = tensor
                .clone()
                .try_into_primitive::<burn_cubecl::Cube>()
                .unwrap();
            let primitive = (&primitive as &dyn std::any::Any)
                .downcast_ref::<burn_cubecl::tensor::CubeTensor>()
                .expect("raw CubeCL");
            primitive.handle.memory.descriptor().id.value
        };
        let mut ids = vec![id(param.as_ref().unwrap())];
        if let Some(state) = state {
            ids.push(id(&state.momentum.moment_1));
            ids.push(id(&state.momentum.moment_2));
            if let Some(max) = &state.momentum.max_moment_2 {
                ids.push(id(max));
            }
        }
        ids
    }

    impl Case {
        fn new(workload: Workload, device: &Device, varied: bool, fused: bool) -> Self {
            if workload == Workload::MixedRank {
                return Self::Mixed(Box::new(Mixed::all(device, varied)));
            }
            if workload == Workload::Unaligned {
                let case = Self::Squares(vec![Square::from_shape([1, 1_048_576], device, varied)]);
                #[cfg(cubecl_backend)]
                if fused {
                    let mut case = case;
                    case.offset_gradient(device);
                    return case;
                }
                #[cfg(not(cubecl_backend))]
                let _ = fused;
                return case;
            }
            if matches!(workload, Workload::TailMinus | Workload::TailPlus) {
                let len = if workload == Workload::TailMinus {
                    1_048_575
                } else {
                    1_048_577
                };
                return Self::Squares(vec![Square::from_shape([1, len], device, varied)]);
            }
            let mut squares = Vec::new();
            for _ in 0..workload.small() {
                squares.push(Square::new(64, device, varied));
            }
            for _ in 0..workload.large() {
                squares.push(Square::new(1024, device, varied));
            }
            Self::Squares(squares)
        }

        /// Keep the reference aligned: its generic vector kernels assume aligned bindings.
        /// The fused gradient retains this offset through warmup and every measured step.
        #[cfg(cubecl_backend)]
        fn offset_gradient(&mut self, device: &Device) {
            let Self::Squares(params) = self else {
                unreachable!("the offset workload has one vector")
            };
            assert_eq!(params.len(), 1);
            let grad = &mut params[0].grad;
            let len = grad.shape().num_elements();
            let offset = match device.as_dispatch() {
                burn_core::backend::DispatchDevice::Cube(cube)
                    if cube.runtime() == burn_cubecl::cubecl::RuntimeId::Wgpu =>
                {
                    cube.client().properties().memory.alignment.div_ceil(4) as usize
                }
                _ => 1,
            };
            let mut values = vec![0.0; offset];
            values.extend_from_slice(grad.to_data().as_slice::<f32>().unwrap());
            let mut primitive =
                Tensor::<2>::from_data(TensorData::new(values, [1, len + offset]), device)
                    .try_into_primitive::<burn_cubecl::Cube>()
                    .unwrap();
            let raw = (&mut primitive as &mut dyn std::any::Any)
                .downcast_mut::<burn_cubecl::tensor::CubeTensor>()
                .expect("unaligned benchmark requires raw CubeCL");
            raw.handle.offset_start = Some((offset * 4) as u64);
            raw.meta.remove(1);
            raw.meta.insert(1, len, 1);
            raw.meta.remove(0);
            raw.meta.insert(0, 1, len);
            *grad = Tensor::from_primitive::<burn_cubecl::Cube>(primitive);
        }

        /// Number of parameter slots stepped by this case.
        #[cfg(cubecl_backend)]
        fn len(&self) -> usize {
            match self {
                Case::Squares(params) => params.len(),
                Case::Mixed(_) => 3,
            }
        }

        /// Prove the production bridge reaches the kernel for both the initial and the
        /// stateful step, rather than silently falling back to the reference implementation.
        #[cfg(cubecl_backend)]
        fn assert_fused_runs(&self, optimizer: &AdamW) {
            let mut stepped = self.clone();
            let (launches, _observation) = KernelCounter::start();
            for kind in ["initial", "stateful"] {
                let before = launches.get();
                stepped.step(optimizer);
                let count = launches.get() - before;
                let tail_case = matches!(self, Self::Squares(params) if params.len() == 1
                    && params[0].param.as_ref().unwrap().dims()[0] == 1
                    && params[0].param.as_ref().unwrap().dims()[1] % 2 == 1);
                assert!(
                    count == self.len() || (tail_case && count == 2),
                    "the {kind} fused step did not reach the expected kernels: {count}"
                );
            }
        }

        #[cfg(cubecl_backend)]
        fn tensors(&self) -> Vec<TensorData> {
            match self {
                Case::Squares(params) => params
                    .iter()
                    .flat_map(|param| state_tensors(&param.param, &param.state))
                    .collect(),
                Case::Mixed(params) => params.iter().flat_map(Mixed::tensors).collect(),
            }
        }

        #[cfg(cubecl_backend)]
        fn allocation_ids(&self) -> Vec<usize> {
            match self {
                Self::Squares(params) => params
                    .iter()
                    .flat_map(|p| allocation_ids(&p.param, &p.state))
                    .collect(),
                Self::Mixed(params) => params
                    .iter()
                    .flat_map(|p| match p {
                        Mixed::Bias(p, _, s) => allocation_ids(p, s),
                        Mixed::Linear(p, _, s) => allocation_ids(p, s),
                        Mixed::Conv(p, _, s) => allocation_ids(p, s),
                    })
                    .collect(),
            }
        }

        fn step(&mut self, optimizer: &AdamW) {
            match self {
                Case::Squares(params) => {
                    for param in params.iter_mut() {
                        param.step(optimizer);
                    }
                }
                Case::Mixed(params) => {
                    for param in params.iter_mut() {
                        param.step(optimizer);
                    }
                }
            }
        }
    }

    pub fn run(bencher: divan::Bencher, workload: Workload, fused: bool, sync: bool) {
        let device = Device::default();
        let optimizer = optimizer(fused);
        let mut case = Case::new(workload, &device, false, fused);
        #[cfg(cubecl_backend)]
        if fused {
            case.assert_fused_runs(&optimizer);
        }
        for _ in 0..5 {
            case.step(&optimizer);
        }
        device.sync().unwrap();
        bencher.bench_local(|| {
            case.step(&optimizer);
            if sync {
                device.sync().unwrap();
            }
        });
        device.sync().unwrap();
    }

    /// The benchmark's defaults must pick a raw CubeCL device, not Flex or a fusion handle.
    #[cfg(cubecl_backend)]
    pub fn assert_raw_cube_device() {
        let device = Device::default();
        let probe = Tensor::<1>::zeros([1], &device);
        assert!(
            probe.try_into_primitive::<burn_cubecl::Cube>().is_ok(),
            "the benchmark device is not raw CubeCL; select a CubeCL runtime without fusion"
        );
    }

    /// The fused and reference results must agree for every workload.
    #[cfg(cubecl_backend)]
    pub fn verify() {
        let tolerance = Tolerance::rel_abs(8.0 * f32::EPSILON, 4.0 * f32::EPSILON);
        let device = Device::default();

        for workload in WORKLOADS {
            let reference = optimizer(false);
            let fused = optimizer(true);
            // Varied inputs so decay, state and bias correction cannot hide behind constants.
            let mut reference_case = Case::new(workload, &device, true, false);
            // The fused case must start from the reference's exact values.
            let mut fused_case = reference_case.clone();
            if workload == Workload::Unaligned {
                fused_case.offset_gradient(&device);
            }
            #[cfg(cubecl_backend)]
            fused_case.assert_fused_runs(&fused);

            for _ in 0..STEPS {
                reference_case.step(&reference);
                fused_case.step(&fused);
            }

            let reference_tensors = reference_case.tensors();
            let fused_tensors = fused_case.tensors();
            assert_eq!(
                reference_tensors.len(),
                fused_tensors.len(),
                "workload {workload}: state tensors diverged"
            );
            for (reference, fused) in reference_tensors.iter().zip(fused_tensors.iter()) {
                reference.assert_approx_eq::<f32>(fused, tolerance);
            }

            println!("verified: {workload} ({} tensors)", fused_tensors.len());
        }
    }

    /// Print what the device allocator holds after a workload's steady state.
    ///
    /// Peak pages are per process, so each mode is measured by its own invocation.
    pub fn measure_memory(workload: Workload, fused: bool) {
        let device = Device::default();
        let optimizer = optimizer(fused);
        let mut case = Case::new(workload, &device, false, fused);
        // Prove the path in a separate --verify-path process: cloning and stepping here
        // would inflate the process-wide allocator peaks before the measurement.
        for _ in 0..STEPS {
            case.step(&optimizer);
        }
        device.sync().unwrap();

        match device.measure_memory() {
            Some(report) => {
                // Per-pool peaks can occur at different times; their sum is an upper bound.
                let peak: u64 = report
                    .pools
                    .iter()
                    .map(|pool| pool.page_size * pool.pages_peak)
                    .sum();
                println!(
                    "memory workload={workload} mode={} allocs={} in_use={} reserved={} peak={peak}",
                    if fused { "fused" } else { "reference" },
                    report.usage.number_allocs,
                    report.usage.bytes_in_use,
                    report.usage.bytes_reserved,
                );
            }
            None => println!(
                "memory workload={workload} mode={}: this device does not report allocator usage",
                if fused { "fused" } else { "reference" },
            ),
        }
    }

    /// Count retained output allocation IDs separately from timed samples.
    #[cfg(cubecl_backend)]
    pub fn measure_reuse(workload: Workload, fused: bool, shared: bool) {
        let device = Device::default();
        let optimizer = optimizer(fused);
        let mut case = Case::new(workload, &device, false, fused);
        case.step(&optimizer);
        device.sync().unwrap();
        let retained = shared.then(|| case.clone());
        let before = case.allocation_ids();
        case.step(&optimizer);
        device.sync().unwrap();
        let after = case.allocation_ids();
        let reused = before.iter().zip(&after).filter(|(a, b)| a == b).count();
        println!(
            "reuse workload={workload} shared={shared} outputs={} reused={reused} new_output_ids={}",
            after.len(),
            after.len() - reused
        );
        drop(retained);
    }

    /// Time the first optimizer step, which also compiles the kernels it submits.
    pub fn measure_jit(workload: Workload, fused: bool) {
        let device = Device::default();
        let optimizer = optimizer(fused);
        let mut case = Case::new(workload, &device, false, fused);
        // Setup compiles its own kernels; only the step is measured. The path is proven by
        // `--verify-path`; running the proof here would JIT the kernel outside the timer.
        device.sync().unwrap();
        let start = std::time::Instant::now();
        case.step(&optimizer);
        device.sync().unwrap();
        println!(
            "jit workload={workload} mode={} first_step_ms={:.3}",
            if fused { "fused" } else { "reference" },
            start.elapsed().as_secs_f64() * 1e3,
        );
    }

    /// Emit individual steady-state samples with a drained queue at each boundary.
    /// With `shared`, retain the previous parameters and state to exercise COW refusal.
    pub fn samples(workload: Workload, fused: bool, shared: bool) {
        let device = Device::default();
        let optimizer = optimizer(fused);
        let mut case = Case::new(workload, &device, false, fused);
        for _ in 0..5 {
            let retained = shared.then(|| case.clone());
            case.step(&optimizer);
            device.sync().unwrap();
            drop(retained);
        }
        for sample in 0..30 {
            let retained = shared.then(|| case.clone());
            let start = std::time::Instant::now();
            case.step(&optimizer);
            let enqueue = start.elapsed();
            device.sync().unwrap();
            let total = start.elapsed();
            let drop_start = std::time::Instant::now();
            drop(retained);
            device.sync().unwrap();
            let drop_sync = drop_start.elapsed();
            println!(
                "{{\"workload\":\"{workload}\",\"fused\":{fused},\"shared\":{shared},\"sample\":{sample},\"enqueue_ms\":{:.6},\"total_ms\":{:.6},\"retained_drop_sync_ms\":{:.6}}}",
                enqueue.as_secs_f64() * 1e3,
                total.as_secs_f64() * 1e3,
                drop_sync.as_secs_f64() * 1e3,
            );
        }
        let start = std::time::Instant::now();
        drop(case);
        device.sync().unwrap();
        println!(
            "{{\"final_drop_sync_ms\":{:.6}}}",
            start.elapsed().as_secs_f64() * 1e3
        );
    }

    #[divan::bench(args = WORKLOADS, consts = [false, true], sample_count = 10)]
    pub fn reference<const SYNC: bool>(bencher: divan::Bencher, workload: Workload) {
        run(bencher, workload, false, SYNC);
    }

    #[cfg(cubecl_backend)]
    #[divan::bench(args = WORKLOADS, consts = [false, true], sample_count = 10)]
    pub fn fused<const SYNC: bool>(bencher: divan::Bencher, workload: Workload) {
        run(bencher, workload, true, SYNC);
    }
}

fn mode_workload(args: &[String], flag: &str) -> Option<(String, bench::Workload)> {
    let index = args.iter().position(|arg| arg == flag)?;
    let mode = args.get(index + 1)?.clone();
    let workload = args
        .get(index + 2)
        .and_then(|name| bench::Workload::from_name(name))
        .unwrap_or_else(|| panic!("usage: {flag} <reference|fused> <workload>"));
    Some((mode, workload))
}

fn main() {
    #[cfg(cubecl_backend)]
    bench::assert_raw_cube_device();
    #[cfg(cubecl_backend)]
    if burn_cubecl::FUSED {
        panic!(
            "kernel fusion is enabled: every fused step would fall back to the reference \
             implementation. Rebuild with --no-default-features --features std,cpu (or another \
             raw CubeCL runtime) for meaningful measurements."
        );
    }

    let args: Vec<String> = std::env::args().collect();
    #[cfg(cubecl_backend)]
    if let Some(index) = args.iter().position(|arg| arg == "--module-samples") {
        foreach::run(&args[index + 1], &args[index + 2]);
        return;
    }

    if args.iter().any(|arg| arg == "--verify-path") {
        #[cfg(cubecl_backend)]
        {
            bench::verify();
            return;
        }
        #[cfg(not(cubecl_backend))]
        panic!(
            "--verify-path needs the fused path: enable a burn-optim runtime feature such as cpu"
        );
    }
    if let Some((mode, workload)) = mode_workload(&args, "--measure-memory") {
        match mode.as_str() {
            "reference" => bench::measure_memory(workload, false),
            #[cfg(cubecl_backend)]
            "fused" => bench::measure_memory(workload, true),
            #[cfg(not(cubecl_backend))]
            "fused" => panic!(
                "the fused path is not compiled in this build: enable a burn-optim runtime feature"
            ),
            other => panic!("unknown mode {other}: expected reference or fused"),
        }
        return;
    }
    if let Some((mode, workload)) = mode_workload(&args, "--measure-jit") {
        match mode.as_str() {
            "reference" => bench::measure_jit(workload, false),
            #[cfg(cubecl_backend)]
            "fused" => bench::measure_jit(workload, true),
            #[cfg(not(cubecl_backend))]
            "fused" => panic!(
                "the fused path is not compiled in this build: enable a burn-optim runtime feature"
            ),
            other => panic!("unknown mode {other}: expected reference or fused"),
        }
        return;
    }

    if let Some((mode, workload)) = mode_workload(&args, "--samples-json") {
        let fused = match mode.as_str() {
            "reference" => false,
            #[cfg(cubecl_backend)]
            "fused" => true,
            other => panic!("unsupported sample mode {other}"),
        };
        bench::samples(workload, fused, args.iter().any(|arg| arg == "--shared"));
        return;
    }

    #[cfg(cubecl_backend)]
    if let Some((mode, workload)) = mode_workload(&args, "--measure-reuse") {
        let fused = match mode.as_str() {
            "fused" => true,
            "reference" => false,
            other => panic!("unknown mode {other}"),
        };
        bench::measure_reuse(workload, fused, args.iter().any(|a| a == "--shared"));
        return;
    }

    #[cfg(not(cubecl_backend))]
    if std::env::var_os("ADAMW_REFERENCE_ONLY").is_none() {
        eprintln!(
            "skipping the AdamW benchmark: this target measures the fused path. Build with \
             --features std,cpu, or enable a backend feature and set ADAMW_REFERENCE_ONLY=1 to \
             measure the reference implementation."
        );
        return;
    }

    divan::main();
}
