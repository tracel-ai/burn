//! A storage implementation in a downstream crate can give each AdamW moment its own
//! representation and keep auxiliary tensors in the same checkpoint.
use burn_core::{
    module::Param,
    tensor::{DType, Device, FloatDType, Int, IntDType, Tensor},
};
use burn_optim::{
    AdamW, AdamWConfig, AdamWState, AdaptiveMomentumState, GradientsParams, ModuleOptimizer,
    OptimizerPrecision, OptimizerStateStorage, RecordState,
};

#[derive(Clone)]
struct CompressedMomentum;

#[derive(Clone, RecordState)]
struct Moments<const D: usize> {
    codes: Tensor<D, Int>,
    scale: Tensor<1>,
    variance: Tensor<D>,
    max_variance: Option<Tensor<D>>,
    step: usize,
}

impl OptimizerStateStorage<AdamW> for CompressedMomentum {
    type Stored<const D: usize> = Moments<D>;

    fn store<const D: usize>(&self, state: AdamWState<D>) -> Moments<D> {
        let momentum = state.momentum;
        let scale = momentum
            .moment_1
            .clone()
            .abs()
            .max()
            .div_scalar(127.)
            .clamp_min(f32::MIN_POSITIVE);
        let codes = (momentum.moment_1 / scale.clone().reshape([1; D]))
            .round()
            .cast(IntDType::I8);
        Moments {
            codes,
            scale,
            variance: momentum.moment_2,
            max_variance: momentum.max_moment_2,
            step: momentum.time,
        }
    }

    fn restore<const D: usize>(&self, stored: Moments<D>, compute: FloatDType) -> AdamWState<D> {
        AdamWState {
            momentum: AdaptiveMomentumState {
                moment_1: stored.codes.cast(compute) * stored.scale.cast(compute).reshape([1; D]),
                moment_2: stored.variance.cast(compute),
                max_moment_2: stored.max_variance.map(|t| t.cast(compute)),
                time: stored.step,
            },
        }
    }

    fn to_device<const D: usize>(stored: Moments<D>, device: &Device) -> Moments<D> {
        Moments {
            codes: stored.codes.to_device(device),
            scale: stored.scale.to_device(device),
            variance: stored.variance.to_device(device),
            max_variance: stored.max_variance.map(|t| t.to_device(device)),
            step: stored.step,
        }
    }
}

#[test]
fn compressed_first_moment_resumes_with_its_scale_and_full_precision_variance() {
    let device = Device::default();
    let create = || {
        let adam = AdamWConfig::new()
            .with_weight_decay(0.)
            .with_beta_1(0.5)
            .with_beta_2(0.5)
            .with_epsilon(1e-8)
            .build();
        ModuleOptimizer::from(
            OptimizerPrecision::new(FloatDType::F32)
                .with_state_storage(CompressedMomentum)
                .wrap(adam),
        )
    };
    let param = Param::from_tensor(Tensor::<2>::ones([1, 2], &device.clone().autodiff()));
    let mut optimizer = create();
    let mut grads = GradientsParams::new();
    grads.register(param.id, Tensor::<2>::from_data([[1., -0.01]], &device));
    let param = optimizer.step(0.001, param, grads);
    let mut resumed = create()
        .from_bytes(optimizer.into_bytes().unwrap())
        .unwrap();
    let record = burn_pack::Reader::from_bytes(resumed.into_bytes().unwrap())
        .unwrap()
        .into_tensors()
        .unwrap();
    assert_eq!(record.len(), 3);
    for tensor in record {
        let dtype = if tensor.name.ends_with(".codes") {
            DType::I8
        } else {
            DType::F32
        };
        assert_eq!(tensor.dtype, dtype, "{}", tensor.name);
    }
    let mut grads = GradientsParams::new();
    grads.register(param.id, Tensor::<2>::zeros([1, 2], &device));
    let param = resumed.step(0.001, param, grads);
    // The small first moment rounds to code -1; variance retains 0.5 * 0.01².
    // On step two, bias correction divides each decayed moment by 0.75.
    let expected = 1.001 + 0.001 * (0.25 / 127. / 0.75) / (0.25_f32 * 0.01 * 0.01 / 0.75).sqrt();
    let actual = param.val().try_into_vec_as::<f32>().unwrap();
    assert!(
        (actual[1] - expected).abs() < 1e-6,
        "{actual:?}, expected {expected}"
    );
}
