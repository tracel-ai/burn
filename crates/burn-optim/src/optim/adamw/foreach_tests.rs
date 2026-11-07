use super::*;
use crate::{GradientsParams, OptimizerRecord};
use burn::module::Param;
use burn::tensor::{TensorData, Tolerance};
use burn_cubecl::cubecl::logging::{LaunchObservation, LaunchObserver};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

struct Counts {
    batch: AtomicUsize,
    single: AtomicUsize,
    thread: std::thread::ThreadId,
    streams: std::sync::Mutex<Vec<burn::tensor::StreamId>>,
}

impl LaunchObserver for Counts {
    fn launched(&self, name: &'static str) {
        if self.thread != std::thread::current().id() {
            return;
        }
        if name.contains("adamw_foreach::") {
            self.batch.fetch_add(1, Ordering::Relaxed);
            self.streams
                .lock()
                .unwrap()
                .push(burn::tensor::StreamId::current());
        }
        if name.contains("optim::adamw::eager::adamw::") {
            self.single.fetch_add(1, Ordering::Relaxed);
        }
    }
}

impl Counts {
    fn start() -> (Arc<Self>, LaunchObservation) {
        let counts = Arc::new(Self {
            batch: AtomicUsize::new(0),
            single: AtomicUsize::new(0),
            thread: std::thread::current().id(),
            streams: std::sync::Mutex::new(Vec::new()),
        });
        let observation = LaunchObservation::new(counts.clone());
        (counts, observation)
    }
    fn batch(&self) -> usize {
        self.batch.load(Ordering::Relaxed)
    }
}

#[derive(burn_derive::Module, Debug)]
struct Mixed {
    vectors: Vec<Param<Tensor<1>>>,
    matrix: Param<Tensor<2>>,
    conv: Param<Tensor<4>>,
}

impl Mixed {
    fn new(device: &Device, count: usize) -> Self {
        Self {
            vectors: (0..count)
                .map(|i| Param::from_tensor(Tensor::full([19 + i], 0.5 + i as f32, device)))
                .collect(),
            matrix: Param::from_tensor(
                Tensor::from_data(
                    TensorData::new((0..15).map(|i| i as f32 * 0.1).collect(), [3, 5]),
                    device,
                )
                .transpose(),
            ),
            conv: Param::from_tensor(Tensor::full([1, 2, 2, 3], -0.7, device)),
        }
    }
    fn grads(&self, device: &Device, step: usize, missing: bool) -> GradientsParams {
        let mut grads = GradientsParams::new();
        for (i, p) in self.vectors.iter().enumerate() {
            if missing && i == 0 {
                continue;
            }
            grads.register(
                p.id,
                Tensor::<1>::full(p.val().shape(), (i + step + 1) as f32 * 0.01, device),
            );
        }
        grads.register(self.matrix.id, Tensor::<2>::full([5, 3], -0.1, device));
        grads.register(self.conv.id, Tensor::<4>::full([1, 2, 2, 3], 0.3, device));
        grads
    }
    fn close(&self, other: &Self) {
        for (a, b) in self.vectors.iter().zip(&other.vectors) {
            close(&a.val(), &b.val());
        }
        close(&self.matrix.val(), &other.matrix.val());
        close(&self.conv.val(), &other.conv.val());
    }
}

fn close<const D: usize>(a: &Tensor<D>, b: &Tensor<D>) {
    assert_eq!(a.is_require_grad(), b.is_require_grad());
    assert_eq!(
        a.gradient_checkpointing_strategy(),
        b.gradient_checkpointing_strategy()
    );
    a.to_data()
        .assert_approx_eq::<f32>(&b.to_data(), Tolerance::rel_abs(1e-5, 1e-6));
}

fn record_close(a: OptimizerRecord, b: OptimizerRecord) {
    assert_eq!(a.scalars, b.scalars);
    assert_eq!(a.paths, b.paths);
    assert_eq!(a.tensors.len(), b.tensors.len());
    for a in a.tensors {
        let b = b.tensors.iter().find(|b| b.name == a.name).unwrap();
        assert_eq!(a.param_id, b.param_id);
        a.data
            .assert_approx_eq::<f32>(&b.data, Tolerance::rel_abs(1e-5, 1e-6));
    }
}

#[test]
fn foreach_direct_steps_use_the_single_parameter_path() {
    let _guard = super::step_guard();
    let device = Device::default();
    let config = AdamWConfig::new().with_amsgrad(true);
    let reference = config.clone().build();
    let candidate = config
        .with_execution(Some(AdamWExecutionConfig::Foreach))
        .build();
    let make = || Tensor::<1>::from_data([0.5, -1.0, 2.0], &device);
    let mut expected = (make(), None);
    let mut actual = (make(), None);
    let (counts, _observe) = Counts::start();
    for step in 1..=2 {
        let grad = Tensor::<1>::from_data([0.1 * step as f32, -0.3, 0.2], &device);
        expected = reference.step(0.01, expected.0, grad.clone(), expected.1);
        actual = candidate.step(0.01, actual.0, grad, actual.1);
        close(&actual.0, &expected.0);
        let a = &actual.1.as_ref().unwrap().momentum;
        let b = &expected.1.as_ref().unwrap().momentum;
        assert_eq!(a.time, step);
        close(&a.moment_1, &b.moment_1);
        close(&a.moment_2, &b.moment_2);
        close(
            a.max_moment_2.as_ref().unwrap(),
            b.max_moment_2.as_ref().unwrap(),
        );
    }
    assert_eq!(counts.batch(), 0);
    let raw_cube = !burn_cubecl::FUSED
        && matches!(device.as_dispatch(), burn::backend::DispatchDevice::Cube(_));
    let expected_launches = if raw_cube {
        let bindings = raw(make()).client.properties().hardware.max_bindings;
        usize::from(bindings >= 7) + usize::from(bindings >= 10)
    } else {
        0
    };
    assert_eq!(counts.single.load(Ordering::Relaxed), expected_launches);
}

#[test]
fn foreach_module_matches_reference_and_restores_checkpoint() {
    let _guard = super::step_guard();
    let raw = Device::default();
    let device = raw.clone().autodiff().gradient_checkpointing();
    for amsgrad in [false, true] {
        for cautious in [false, true] {
            let config = AdamWConfig::new()
                .with_amsgrad(amsgrad)
                .with_cautious_weight_decay(cautious)
                .with_weight_decay(0.3)
                .with_grad_clipping(Some(GradientClippingConfig::Value(0.08)));
            let mut reference = config.clone().init();
            let mut candidate = config
                .clone()
                .with_execution(Some(AdamWExecutionConfig::Foreach))
                .init();
            let mut a = Mixed::new(&device, 3);
            let mut b = a.clone();
            let (counts, _observe) = Counts::start();
            for step in 0..5 {
                let missing = step == 1;
                let ga = a.grads(&raw, step, missing);
                let gb = b.grads(&raw, step, missing);
                a = reference.step(0.01, a, ga);
                b = candidate.step(0.01, b, gb);
                a.close(&b);
                record_close(reference.to_record(), candidate.to_record());
                if step == 2 {
                    // Cross both directions while retaining ids/ranks/time and group paths.
                    let ref_bytes = reference.into_bytes().unwrap();
                    let foreach_bytes = candidate.into_bytes().unwrap();
                    reference = config.clone().init().from_bytes(foreach_bytes).unwrap();
                    candidate = config
                        .clone()
                        .with_execution(Some(AdamWExecutionConfig::Foreach))
                        .init()
                        .from_bytes(ref_bytes)
                        .unwrap();
                }
            }
            if binding_budget() < if amsgrad { 13 } else { 11 } {
                assert_eq!(counts.batch(), 0);
                if !burn_cubecl::FUSED
                    && matches!(raw.as_dispatch(), burn::backend::DispatchDevice::Cube(_))
                {
                    assert!(counts.single.load(Ordering::Relaxed) > 0);
                }
            } else {
                assert!(counts.batch() > 0, "the module must reach the batch kernel");
            }
        }
    }
}

#[test]
fn foreach_splits_parameter_budget_and_handles_empty_and_singleton() {
    let _guard = super::step_guard();
    let device = Device::default();
    let ad = device.clone().autodiff();
    let config = AdamWConfig::new();
    let mut a = Mixed::new(&ad, 21);
    let mut b = a.clone();
    let mut reference = config.clone().init();
    let mut candidate = config
        .with_execution(Some(AdamWExecutionConfig::Foreach))
        .init();
    let (counts, _observe) = Counts::start();
    let ga = a.grads(&device, 0, false);
    let gb = b.grads(&device, 0, false);
    a = reference.step(0.01, a, ga);
    b = candidate.step(0.01, b, gb);
    a.close(&b);
    if binding_budget() >= 11 {
        assert!(counts.batch() >= 3);
    }
    let before = counts.batch();
    b = candidate.step(0.01, b, GradientsParams::new());
    a.close(&b);
    assert_eq!(counts.batch(), before);
    let mut ga = GradientsParams::new();
    let mut gb = GradientsParams::new();
    ga.register(a.matrix.id, Tensor::<2>::ones([5, 3], &device));
    gb.register(b.matrix.id, Tensor::<2>::ones([5, 3], &device));
    a = reference.step(0.01, a, ga);
    b = candidate.step(0.01, b, gb);
    a.close(&b);
    assert_eq!(counts.batch(), before);
    record_close(reference.to_record(), candidate.to_record());
}

#[test]
fn foreach_restores_collected_effective_stream_and_preserves_pins() {
    let _guard = super::step_guard();
    if binding_budget() == 0 {
        return;
    }
    use burn::tensor::StreamId;
    use burn_cubecl::kernel::optim::{
        AdamWHostScalars, AdamWInputs, execute_adamw_batch, prepare_adamw,
    };
    let device = Device::default();
    let outer = StreamId::allocate();
    let other = StreamId::allocate();
    let singleton = StreamId::allocate();
    let unpinned = StreamId::allocate();
    let make = |pin: Option<StreamId>| {
        let mut param = raw(Tensor::<1>::full([19], 1.0, &device));
        if let Some(pin) = pin {
            unsafe { param.client.set_stream(pin) };
        }
        let mut grad = raw(Tensor::<1>::full([19], 0.1, &device));
        grad.client = param.client.clone();
        (param, grad)
    };
    let mut pending = Vec::new();
    outer.executes(|| {
        for pin in [None, None, Some(other), Some(other), Some(singleton)] {
            let (param, grad) = make(pin);
            let inputs = AdamWInputs::Initial { param, grad };
            pending.push(
                prepare_adamw(
                    AdamWHostScalars::new(0.01, 0.9, 0.999, 1e-5, 0.0, 1),
                    false,
                    false,
                    inputs,
                )
                .ok()
                .unwrap(),
            );
        }
    });
    pending.extend(core::iter::once_with(|| {
        unpinned.executes(|| {
            let (param, grad) = make(None);
            prepare_adamw(
                AdamWHostScalars::new(0.01, 0.9, 0.999, 1e-5, 0.0, 1),
                false,
                false,
                AdamWInputs::Initial { param, grad },
            )
            .ok()
            .unwrap()
        })
    }));
    let (counts, _observe) = Counts::start();
    let ambient = StreamId::allocate();
    let results: Vec<
        burn_cubecl::kernel::optim::AdamWBatchResult<burn_cubecl::tensor::CubeTensor>,
    > = ambient.executes(|| {
        let output = execute_adamw_batch(pending);
        assert_eq!(StreamId::current(), ambient);
        output
    });
    assert_eq!(counts.single.load(Ordering::Relaxed), 0);
    assert!(matches!(
        results.last().unwrap(),
        burn_cubecl::kernel::optim::AdamWBatchResult::Deferred(_)
    ));
    let resume = StreamId::allocate();
    let outputs = resume.executes(|| {
        let outputs = results
            .into_iter()
            .map(|result| result.finish())
            .collect::<Vec<_>>();
        assert_eq!(StreamId::current(), resume);
        outputs
    });
    if binding_budget() >= 11 {
        assert_eq!(counts.batch(), 2);
        assert_eq!(counts.single.load(Ordering::Relaxed), 2);
        assert_eq!(*counts.streams.lock().unwrap(), [outer, other]);
    } else {
        assert_eq!(counts.batch(), 0);
        assert_eq!(counts.single.load(Ordering::Relaxed), 6);
    }
    for (i, output) in outputs.into_iter().enumerate() {
        let expected = match i {
            0..2 => outer,
            2..4 => other,
            4 => singleton,
            _ => unpinned,
        };
        assert_eq!(output.moment_1.handle.stream, expected);
        assert_eq!(output.moment_2.handle.stream, expected);
        let p = from_raw::<1>(output.param);
        close(&p, &Tensor::full([19], 0.990001, &device));
    }
}

#[test]
fn foreach_chunk_tails_and_owned_state_reuse_match_reference() {
    let _guard = super::step_guard();
    if binding_budget() < 19 {
        return;
    }
    use burn_cubecl::kernel::optim::execute_adamw_batch;
    let device = Device::default();
    let config = AdamWConfig::new()
        .with_amsgrad(true)
        .with_cautious_weight_decay(true)
        .with_weight_decay(0.4);
    let reference = config.clone().build();
    let candidate = config
        .with_execution(Some(AdamWExecutionConfig::Foreach))
        .build();
    let mut expected = Vec::new();
    let mut actual = Vec::new();
    for (i, len) in [8195, 33, 4097].into_iter().enumerate() {
        let make = || {
            let p = varying(len, i as f32 + 0.3, &device);
            let state = AdamWState {
                momentum: AdaptiveMomentumState {
                    time: 5,
                    moment_1: varying(len, (i + 1) as f32 * 0.013, &device),
                    moment_2: varying(len, (i + 1) as f32 * 0.002, &device),
                    max_moment_2: Some(varying(len, (i + 1) as f32 * 0.005, &device)),
                },
            };
            (p, Some(state))
        };
        expected.push(make());
        actual.push(make());
    }
    let id = |t: &Tensor<1>| raw(t.clone()).handle.memory.descriptor().id;
    for step in 0..3 {
        let mut ids = Vec::new();
        let (times, pending): (Vec<_>, Vec<_>) = core::mem::take(&mut actual)
            .into_iter()
            .zip(core::mem::take(&mut expected))
            .enumerate()
            .map(|(i, ((p, s), (rp, rs)))| {
                let grad = varying(
                    p.shape().num_elements(),
                    (i + step + 1) as f32 * 0.07,
                    &device,
                );
                let state = s.as_ref().unwrap();
                ids.push([
                    id(&p),
                    id(&state.momentum.moment_1),
                    id(&state.momentum.moment_2),
                    id(state.momentum.max_moment_2.as_ref().unwrap()),
                ]);
                expected.push(reference.step(0.01, rp, grad.clone(), rs));
                candidate.prepare(0.01, p, grad, s).ok().unwrap()
            })
            .unzip();
        let (counts, _observe) = Counts::start();
        let outputs = execute_adamw_batch(pending)
            .into_iter()
            .map(|result| result.finish())
            .collect::<Vec<_>>();
        assert!(counts.batch() > 0);
        for ((output, time), ids) in outputs.into_iter().zip(times).zip(ids) {
            let (p, s) = AdamW::restore::<1>(time, output);
            assert_eq!(
                [
                    id(&p),
                    id(&s.momentum.moment_1),
                    id(&s.momentum.moment_2),
                    id(s.momentum.max_moment_2.as_ref().unwrap())
                ],
                ids
            );
            actual.push((p, Some(s)));
        }
        for ((a, sa), (b, sb)) in actual.iter().zip(&expected) {
            close(a, b);
            let a = &sa.as_ref().unwrap().momentum;
            let b = &sb.as_ref().unwrap().momentum;
            assert_eq!(a.time, b.time);
            close(&a.moment_1, &b.moment_1);
            close(&a.moment_2, &b.moment_2);
            close(
                a.max_moment_2.as_ref().unwrap(),
                b.max_moment_2.as_ref().unwrap(),
            );
        }
    }
}

fn varying(len: usize, base: f32, device: &Device) -> Tensor<1> {
    Tensor::from_data(
        TensorData::new(
            (0..len)
                .map(|i| base + i as f32 * 0.000003)
                .collect::<Vec<_>>(),
            [len],
        ),
        device,
    )
}

fn raw<const D: usize>(tensor: Tensor<D>) -> burn_cubecl::tensor::CubeTensor {
    let primitive = tensor.try_into_primitive::<burn_cubecl::Cube>().unwrap();
    *(Box::new(primitive) as Box<dyn core::any::Any>)
        .downcast::<burn_cubecl::tensor::CubeTensor>()
        .ok()
        .unwrap()
}

fn from_raw<const D: usize>(tensor: burn_cubecl::tensor::CubeTensor) -> Tensor<D> {
    let primitive = *(Box::new(tensor) as Box<dyn core::any::Any>)
        .downcast::<CubePrimitive>()
        .ok()
        .unwrap();
    Tensor::from_primitive::<burn_cubecl::Cube>(primitive)
}

fn binding_budget() -> u32 {
    if burn_cubecl::FUSED {
        return 0;
    }
    let device = Device::default();
    if !matches!(device.as_dispatch(), burn::backend::DispatchDevice::Cube(cube)
        if cube.runtime() == burn_cubecl::cubecl::RuntimeId::Cpu)
    {
        return 0;
    }
    let tensor = raw(Tensor::<1>::ones([1], &device));
    tensor.client.properties().hardware.max_bindings
}

#[test]
fn foreach_keeps_mixed_optimizers_last_group_lr_and_missing_gradient_time() {
    let _guard = super::step_guard();
    use crate::lr_scheduler::{constant::ConstantLr, module_lr_scheduler::ModuleLrScheduler};
    use burn::module::ParamGroup;
    let raw = Device::default();
    let device = raw.clone().autodiff();
    let mut a = Mixed::new(&device, 5);
    let mut b = a.clone();
    let group = ParamGroup::from_ids(vec![a.vectors[0].id, a.vectors[1].id]);
    let sgd_id = a.vectors[4].id;
    let sgd_group = ParamGroup::from_ids(vec![sgd_id]);
    let lr_group = ParamGroup::from_ids(vec![a.vectors[1].id, a.vectors[2].id]);
    let base = AdamWConfig::new().with_weight_decay(0.1);
    let make = |execution: Option<AdamWExecutionConfig>| {
        base.clone()
            .with_execution(execution.clone())
            .init()
            .with_group(group.clone(), crate::SgdConfig::new().build(), None)
            .with_group(
                group.clone(),
                base.clone()
                    .with_beta_1(0.7)
                    .with_amsgrad(true)
                    .with_execution(execution)
                    .build(),
                Some(GradientClippingConfig::Value(0.03).init()),
            )
            .with_group(
                sgd_group.clone(),
                crate::SgdConfig::new()
                    .with_momentum(Some(
                        crate::momentum::MomentumConfig::new()
                            .with_momentum(0.5)
                            .with_dampening(0.0),
                    ))
                    .build(),
                None,
            )
    };
    let mut reference = make(None);
    let mut candidate = make(Some(AdamWExecutionConfig::Foreach));
    let mut scheduler =
        ModuleLrScheduler::new(ConstantLr::new(0.01)).with_group(lr_group, ConstantLr::new(0.2));
    let (counts, _observe) = Counts::start();
    for step in 0..5 {
        let lr = scheduler.step();
        let ga = a.grads(&raw, step, step % 2 == 0);
        let gb = b.grads(&raw, step, step % 2 == 0);
        a = reference.step(lr.clone(), a, ga);
        b = candidate.step(lr, b, gb);
        a.close(&b);
        let record = candidate.to_record();
        assert!(record.tensors.iter().any(|tensor| {
            tensor.param_id == Some(sgd_id.val())
                && tensor.name == format!("{}.momentum.velocity", sgd_id.val())
        }));
        if step == 0 {
            b.vectors[4].to_data().assert_approx_eq::<f32>(
                &TensorData::new(vec![4.4995; 23], [23]),
                Tolerance::absolute(1e-6),
            );
        }
        record_close(reference.to_record(), record);
    }
    if binding_budget() >= 11 {
        assert!(
            counts.batch() > 0,
            "AdamW must batch alongside the active SGD group"
        );
    }
}

#[test]
fn foreach_multi_gradients_restore_reparameterized_parameters() {
    let _guard = super::step_guard();
    use burn::module::{Module, Reparameterization, Reparameterizer};
    #[derive(burn_derive::Module, Debug)]
    struct Scale {
        scale: Param<Tensor<1>>,
    }
    impl Reparameterization for Scale {
        const NAME: &'static str = "scale";
        fn apply<const D: usize>(&self, base: Tensor<D>) -> Tensor<D> {
            base * self.scale.val().reshape([1; D])
        }
    }
    struct Scaler;
    impl Reparameterizer for Scaler {
        type Reparam = Scale;
        fn reparameterize<const D: usize>(
            &mut self,
            _path: &str,
            param: Param<Tensor<D>>,
        ) -> (Param<Tensor<D>>, Option<Scale>) {
            let scale = (D == 2).then(|| Scale {
                scale: Param::from_tensor(Tensor::full([1], 2.0, &param.lazy_device())),
            });
            (param, scale)
        }
    }
    #[derive(burn_derive::Module, Debug)]
    struct Scaled {
        weight: Param<Tensor<2>>,
        bias: Param<Tensor<1>>,
    }
    let raw = Device::default();
    let device = raw.clone().autodiff().gradient_checkpointing();
    let mut a = Scaled {
        weight: Param::from_tensor(Tensor::full([2, 2], 0.4, &device)),
        bias: Param::from_tensor(Tensor::full([2], 0.2, &device)),
    }
    .apply_reparameterization(Scaler);
    let mut b = a.clone();
    let grads = |m: &Scaled, factor: f64| {
        let value = (m.weight.val().sum() + m.bias.val().sum()) * factor;
        GradientsParams::from_grads(value.backward(), m)
    };
    let config = AdamWConfig::new().with_amsgrad(true);
    let mut reference = config.clone().init();
    let mut candidate = config
        .with_execution(Some(AdamWExecutionConfig::Foreach))
        .init();
    for _ in 0..3 {
        let ga = grads(&a, 1.0);
        let gb = crate::MultiGradientsParams {
            grads: vec![(grads(&b, 0.5), raw.clone()), (grads(&b, 0.5), raw.clone())],
        };
        a = reference.step(0.01, a, ga);
        b = candidate.step_multi(0.01, b, gb);
        close(&a.weight.base(), &b.weight.base());
        close(&a.weight.val(), &b.weight.val());
        close(&a.bias.val(), &b.bias.val());
        let sa = a.weight.reparameterization::<Scale>().unwrap();
        let sb = b.weight.reparameterization::<Scale>().unwrap();
        close(&sa.scale.val(), &sb.scale.val());
        record_close(reference.to_record(), candidate.to_record());
    }
}

#[test]
fn foreach_rank_above_dynamic_limit_keeps_original_failure() {
    let _guard = super::step_guard();
    #[derive(burn_derive::Module, Debug)]
    struct HighRank {
        weight: Param<Tensor<9>>,
    }
    let raw = Device::default();
    let device = raw.clone().autodiff();
    let model = HighRank {
        weight: Param::from_tensor(Tensor::ones([1; 9], &device)),
    };
    for execution in [
        AdamWExecutionConfig::Reference,
        AdamWExecutionConfig::Foreach,
    ] {
        let mut optim = AdamWConfig::new().with_execution(Some(execution)).init();
        let mut grads = GradientsParams::new();
        grads.register(model.weight.id, Tensor::<9>::ones([1; 9], &raw));
        let (counts, _observe) = Counts::start();
        let error = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            optim.step(0.01, model.clone(), grads)
        }))
        .unwrap_err();
        let message = error
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| error.downcast_ref::<&str>().copied())
            .unwrap();
        assert!(message.contains("Unsupported tensor rank for optimizer state: 9"));
        assert_eq!(counts.batch(), 0);
        assert!(optim.to_record().is_empty());
    }
}

#[test]
fn foreach_aliases_across_items_keep_reads_and_preserved_maximum() {
    let _guard = super::step_guard();
    if binding_budget() < 15 {
        return;
    }
    let device = Device::default();
    let make = || {
        let shared = Tensor::<1>::full([16], 1.0, &device);
        let other = Tensor::<1>::full([16], 2.0, &device);
        let moment = Tensor::<1>::full([16], 0.01, &device);
        let state = |m1, m2, max| {
            Some(AdamWState {
                momentum: AdaptiveMomentumState {
                    time: 3,
                    moment_1: m1,
                    moment_2: m2,
                    max_moment_2: Some(max),
                },
            })
        };
        vec![
            (
                shared.clone(),
                Tensor::full([16], 0.3, &device),
                state(
                    moment.clone(),
                    Tensor::full([16], 0.02, &device),
                    other.clone(),
                ),
            ),
            (
                other,
                shared,
                state(
                    Tensor::full([16], 0.04, &device),
                    moment,
                    Tensor::full([16], 0.05, &device),
                ),
            ),
        ]
    };
    let config = AdamWConfig::new().with_weight_decay(0.1);
    let reference = config.clone().build();
    let expected: Vec<_> = make()
        .into_iter()
        .map(|(p, g, s)| reference.step(0.01, p, g, s))
        .collect();
    let candidate = config
        .with_execution(Some(AdamWExecutionConfig::Foreach))
        .build();
    let inputs = make();
    let old_params: Vec<_> = inputs
        .iter()
        .map(|(p, _, _)| raw(p.clone()).handle.memory.descriptor().id)
        .collect();
    let (times, pending): (Vec<_>, Vec<_>) = inputs
        .into_iter()
        .map(|(p, g, s)| candidate.prepare(0.01, p, g, s).ok().unwrap())
        .unzip();
    let outputs = burn_cubecl::kernel::optim::execute_adamw_batch(pending)
        .into_iter()
        .map(|result| result.finish())
        .collect::<Vec<_>>();
    for (i, ((output, time), (rp, rs))) in outputs.into_iter().zip(times).zip(expected).enumerate()
    {
        let (p, s) = AdamW::restore::<1>(time, output);
        assert_ne!(raw(p.clone()).handle.memory.descriptor().id, old_params[i]);
        close(&p, &rp);
        let s = &s.momentum;
        let r = &rs.unwrap().momentum;
        close(&s.moment_1, &r.moment_1);
        close(&s.moment_2, &r.moment_2);
        close(
            s.max_moment_2.as_ref().unwrap(),
            r.max_moment_2.as_ref().unwrap(),
        );
    }
}

#[test]
fn foreach_public_boundary_declines_gpu_without_consuming_inputs() {
    let _guard = super::step_guard();
    if burn_cubecl::FUSED {
        return;
    }
    let device = Device::default();
    if !matches!(device.as_dispatch(), burn::backend::DispatchDevice::Cube(_)) {
        return;
    }
    let param = raw(Tensor::<1>::ones([8], &device));
    if param.device.runtime() == burn_cubecl::cubecl::RuntimeId::Cpu {
        return;
    }
    let id = param.handle.memory.descriptor().id;
    let grad = raw(Tensor::<1>::full([8], 0.1, &device));
    use burn_cubecl::kernel::optim::{AdamWHostScalars, AdamWInputs, prepare_adamw};
    let refused = prepare_adamw(
        AdamWHostScalars::new(0.01, 0.9, 0.999, 1e-5, 0.0, 1),
        false,
        false,
        AdamWInputs::Initial { param, grad },
    )
    .err()
    .unwrap();
    let AdamWInputs::Initial { param, .. } = refused else {
        panic!("initial variant must survive rejection")
    };
    assert_eq!(param.handle.memory.descriptor().id, id);
    close(&from_raw::<1>(param), &Tensor::ones([8], &device));
}

#[test]
fn foreach_splits_chunk_budget_and_refuses_oversized_input() {
    let _guard = super::step_guard();
    if binding_budget() < 11 {
        return;
    }
    let device = Device::default();
    let candidate = AdamWConfig::new()
        .with_execution(Some(AdamWExecutionConfig::Foreach))
        .build();
    let reference = AdamWConfig::new().build();
    let mut expected = Vec::new();
    let (_, pending): (Vec<_>, Vec<_>) = (0..4)
        .map(|i| {
            let len = 10 * 4096;
            let p = varying(len, 0.5 + i as f32, &device);
            let g = varying(len, 0.01, &device);
            expected.push(reference.step(0.01, p.clone(), g.clone(), None));
            candidate.prepare(0.01, p, g, None).ok().unwrap()
        })
        .unzip();
    let (counts, _observe) = Counts::start();
    let outputs = burn_cubecl::kernel::optim::execute_adamw_batch(pending)
        .into_iter()
        .map(|result| result.finish())
        .collect::<Vec<_>>();
    assert_eq!(counts.batch(), 1);
    assert_eq!(counts.single.load(Ordering::Relaxed), 1);
    for (output, (p, s)) in outputs.into_iter().zip(expected) {
        let (actual, state) = AdamW::restore::<1>(1, output);
        close(&actual, &p);
        close(&state.momentum.moment_2, &s.unwrap().momentum.moment_2);
    }
    for len in [32 * 4096 - 1, 32 * 4096, 32 * 4096 + 1] {
        let p = varying(len, 0.4, &device);
        let id = raw(p.clone()).handle.memory.descriptor().id;
        let result = candidate
            .prepare(0.01, p, varying(len, 0.01, &device), None)
            .err()
            .unwrap();
        assert_eq!(raw(result.0.clone()).handle.memory.descriptor().id, id);
        assert!(result.2.is_none());
    }
    assert_eq!(counts.batch(), 1);
    assert_eq!(counts.single.load(Ordering::Relaxed), 1);

    // The boundary still permits a long item when one chunk remains for a shorter peer.
    let mut expected = Vec::new();
    let (_, pending): (Vec<_>, Vec<_>) = [31 * 4096, 1]
        .into_iter()
        .map(|len| {
            let p = varying(len, 0.4, &device);
            let g = varying(len, 0.01, &device);
            expected.push(reference.step(0.01, p.clone(), g.clone(), None));
            candidate.prepare(0.01, p, g, None).ok().unwrap()
        })
        .unzip();
    let before = counts.batch();
    let outputs = burn_cubecl::kernel::optim::execute_adamw_batch(pending)
        .into_iter()
        .map(|result| result.finish())
        .collect::<Vec<_>>();
    assert_eq!(counts.batch() - before, 1);
    for (output, (p, s)) in outputs.into_iter().zip(expected) {
        let (actual, state) = AdamW::restore::<1>(1, output);
        close(&actual, &p);
        close(&state.momentum.moment_2, &s.unwrap().momentum.moment_2);
    }
}

#[test]
fn foreach_module_singletons_and_full_chunk_budget_keep_parameter_reuse() {
    let _guard = super::step_guard();
    if binding_budget() < 11 {
        return;
    }
    #[derive(burn_derive::Module, Debug)]
    struct Large {
        weight: Param<Tensor<2>>,
    }
    let raw_device = Device::default();
    let device = raw_device.clone().autodiff();
    let allocation = |model: &Large| {
        raw(model.weight.val().inner())
            .handle
            .memory
            .descriptor()
            .id
    };
    // The small item is deferred until writeback; the full-budget item bypasses collection.
    for shape in [[4, 4], [512, 256]] {
        let make = || Large {
            weight: Param::from_tensor(Tensor::full(shape, 0.5, &device)),
        };
        let mut expected = make();
        let mut actual = make();
        let mut reference = AdamWConfig::new()
            .with_execution(Some(AdamWExecutionConfig::Fused))
            .init();
        let mut candidate = AdamWConfig::new()
            .with_execution(Some(AdamWExecutionConfig::Foreach))
            .init();
        for _ in 0..2 {
            let before = allocation(&actual);
            let mut ga = GradientsParams::new();
            ga.register(
                expected.weight.id,
                Tensor::<2>::full(shape, 0.1, &raw_device),
            );
            expected = reference.step(0.01, expected, ga);
            let mut gb = GradientsParams::new();
            gb.register(actual.weight.id, Tensor::<2>::full(shape, 0.1, &raw_device));
            let (counts, _observe) = Counts::start();
            actual = candidate.step(0.01, actual, gb);
            assert_eq!(counts.batch(), 0);
            assert_eq!(counts.single.load(Ordering::Relaxed), 1);
            assert_eq!(allocation(&actual), before);
            close(&actual.weight.val(), &expected.weight.val());
        }
    }
}

#[test]
fn foreach_amsgrad_switches_preserve_and_reinitialize_maximum() {
    let _guard = super::step_guard();
    if binding_budget() < 19 {
        return;
    }
    #[derive(burn_derive::Module, Debug)]
    struct MixedRank {
        vector: Param<Tensor<1>>,
        rank8: Param<Tensor<8>>,
    }
    let raw_device = Device::default();
    let device = raw_device.clone().autodiff();
    let mut expected = MixedRank {
        vector: Param::from_tensor(Tensor::full([4], 0.4, &device)),
        rank8: Param::from_tensor(Tensor::full([1, 1, 1, 1, 1, 1, 2, 2], 1.4, &device)),
    };
    let mut actual = expected.clone();
    let ids = [(actual.vector.id, 1), (actual.rank8.id, 8)];
    let mut reference_bytes = None;
    let mut candidate_bytes = None;
    for (step, (amsgrad, grad, expected_m2)) in [
        (false, 4.0, 8.0),
        (true, 2.0, 6.0),
        (false, 1.0, 3.5),
        (true, 0.5, 1.875),
    ]
    .into_iter()
    .enumerate()
    {
        let config = AdamWConfig::new()
            .with_beta_1(0.5)
            .with_beta_2(0.5)
            .with_amsgrad(amsgrad)
            .with_weight_decay(0.0);
        let mut reference = config.clone().init();
        let mut candidate = config
            .with_execution(amsgrad.then_some(AdamWExecutionConfig::Foreach))
            .init();
        if let Some(bytes) = reference_bytes.take() {
            reference = reference.from_bytes(bytes).unwrap();
        }
        if let Some(bytes) = candidate_bytes.take() {
            candidate = candidate.from_bytes(bytes).unwrap();
        }
        let make_grads = |model: &MixedRank| {
            let mut grads = GradientsParams::new();
            grads.register(model.vector.id, Tensor::<1>::full([4], grad, &raw_device));
            grads.register(
                model.rank8.id,
                Tensor::<8>::full([1, 1, 1, 1, 1, 1, 2, 2], grad, &raw_device),
            );
            grads
        };
        let ga = make_grads(&expected);
        let gb = make_grads(&actual);
        expected = reference.step(0.01, expected, ga);
        let (counts, _observe) = Counts::start();
        actual = candidate.step(0.01, actual, gb);
        assert_eq!(counts.batch(), usize::from(amsgrad));
        assert_eq!(counts.single.load(Ordering::Relaxed), 0);
        close(&actual.vector.val(), &expected.vector.val());
        close(&actual.rank8.val(), &expected.rank8.val());
        let record = candidate.to_record();
        for (id, rank) in ids {
            assert_eq!(
                usize::try_from(record.scalars[&format!("{}.__rank", id.val())]).unwrap(),
                rank
            );
            assert_eq!(
                usize::try_from(record.scalars[&format!("{}.momentum.time", id.val())]).unwrap(),
                step + 1
            );
            for field in ["moment_2", "max_moment_2"] {
                let tensor = record
                    .tensors
                    .iter()
                    .find(|t| t.name == format!("{}.momentum.{field}", id.val()));
                if field == "max_moment_2" && step == 0 {
                    assert!(tensor.is_none());
                    continue;
                }
                let tensor = tensor.unwrap();
                assert_eq!(tensor.data.dtype(), burn::tensor::DType::F32);
                assert_eq!(tensor.data.shape().num_dims(), rank);
                let value = if field == "moment_2" {
                    expected_m2
                } else {
                    6.0
                };
                assert!(
                    tensor
                        .data
                        .as_slice::<f32>()
                        .unwrap()
                        .iter()
                        .all(|x| *x == value)
                );
            }
        }
        record_close(reference.to_record(), record);
        reference_bytes = Some(reference.into_bytes().unwrap());
        candidate_bytes = Some(candidate.into_bytes().unwrap());
    }
}

#[test]
fn foreach_amsgrad_maximum_keeps_nan_direction_and_equal_zero_bits() {
    let _guard = super::step_guard();
    if binding_budget() < 19 {
        return;
    }
    let device = Device::default();
    let config = AdamWConfig::new()
        .with_beta_2(0.5)
        .with_amsgrad(true)
        .with_weight_decay(0.0);
    let reference = config.clone().build();
    let candidate = config
        .with_execution(Some(AdamWExecutionConfig::Foreach))
        .build();
    let to_tensor =
        |values: Vec<f32>| Tensor::<1>::from_data(TensorData::new(values, [3]), &device);
    let mut expected = Vec::new();
    let (_, pending): (Vec<_>, Vec<_>) = [false, true]
        .into_iter()
        .map(|reverse| {
            let order = if reverse { [2, 1, 0] } else { [0, 1, 2] };
            let tensor = |values: [f32; 3]| to_tensor(order.map(|i| values[i]).to_vec());
            let param = tensor([1.0, 2.0, 3.0]);
            let grad = tensor([2.0, 0.0, 0.0]);
            let state = AdamWState {
                momentum: AdaptiveMomentumState {
                    time: 3,
                    moment_1: tensor([0.0; 3]),
                    moment_2: tensor([0.0, f32::NAN, 0.0]),
                    max_moment_2: Some(tensor([f32::NAN, 7.0, -0.0])),
                },
            };
            expected.push(reference.step(0.01, param.clone(), grad.clone(), Some(state.clone())));
            candidate
                .prepare(0.01, param, grad, Some(state))
                .ok()
                .unwrap()
        })
        .unzip();
    let (counts, _observe) = Counts::start();
    let outputs = burn_cubecl::kernel::optim::execute_adamw_batch(pending)
        .into_iter()
        .map(|result| result.finish())
        .collect::<Vec<_>>();
    assert_eq!(counts.batch(), 1);
    assert_eq!(counts.single.load(Ordering::Relaxed), 0);
    let check = |a: &Tensor<1>, b: &Tensor<1>| {
        let a = a.to_data();
        let b = b.to_data();
        for (&a, &b) in a
            .as_slice::<f32>()
            .unwrap()
            .iter()
            .zip(b.as_slice::<f32>().unwrap())
        {
            if b.is_nan() {
                assert!(a.is_nan());
            } else if b == 0.0 || b.is_infinite() {
                assert_eq!(a.to_bits(), b.to_bits());
            } else {
                assert!((a - b).abs() <= 1e-6 + 1e-5 * b.abs());
            }
        }
    };
    for (i, (output, (rp, rs))) in outputs.into_iter().zip(expected).enumerate() {
        let (p, s) = AdamW::restore::<1>(4, output);
        let r = rs.unwrap();
        check(&p, &rp);
        check(&s.momentum.moment_1, &r.momentum.moment_1);
        check(&s.momentum.moment_2, &r.momentum.moment_2);
        let maximum = s.momentum.max_moment_2.unwrap();
        check(&maximum, r.momentum.max_moment_2.as_ref().unwrap());
        let data = maximum.to_data();
        let values = data.as_slice::<f32>().unwrap();
        assert!(values[if i == 0 { 0 } else { 2 }].is_nan());
        assert_eq!(values[1], 7.0);
        assert_eq!(
            values[if i == 0 { 2 } else { 0 }].to_bits(),
            (-0.0f32).to_bits()
        );
    }
}

#[test]
#[ignore = "kernel-key collection is process-wide; run alone with --ignored"]
fn foreach_jit_reuses_stable_layout_and_distinguishes_descriptors() {
    let _guard = super::step_guard();
    if binding_budget() < 11 {
        return;
    }
    let device = Device::default();
    let candidate = AdamWConfig::new()
        .with_execution(Some(AdamWExecutionConfig::Foreach))
        .build();
    let run = |len, lr| {
        let pending = (0..2)
            .map(|i| {
                candidate
                    .prepare(
                        lr,
                        varying(len, i as f32 + 0.5, &device),
                        varying(len, 0.1, &device),
                        None,
                    )
                    .ok()
                    .unwrap()
                    .1
            })
            .collect();
        device.sync().unwrap();
        let collection = burn_cubecl::cubecl::launched::LaunchedKernels::new();
        let output: Vec<burn_cubecl::kernel::optim::AdamWOutput<CubePrimitive>> =
            burn_cubecl::kernel::optim::execute_adamw_batch(pending)
                .into_iter()
                .map(|result| result.finish())
                .collect();
        let keys = collection.finish();
        device.sync().unwrap();
        drop(output);
        assert_eq!(keys.len(), 1);
        keys
    };
    let a = run(4099, 0.01);
    let b = run(8195, 0.01);
    assert_ne!(a, b);
    assert_eq!(a, run(4099, 0.02));
}
