//! Device defaults can only be set before the first tensor is created.
use burn_core::{
    module::Param,
    tensor::{DType, Device, FloatDType, Tensor},
};
use burn_optim::{AdamWConfig, GradientsParams, ModuleOptimizer, OptimizerPrecision};
use burn_pack::Reader;

fn step(optimizer: &mut ModuleOptimizer, param: Param<Tensor<1>>, grad: f32) -> Param<Tensor<1>> {
    let mut grads = GradientsParams::new();
    grads.register(
        param.id,
        Tensor::<1>::from_data([grad], (&param.val().device().inner(), DType::F16)),
    );
    optimizer.step(0.0002, param, grads)
}

#[test]
fn checkpoint_preserves_master_and_moments_on_f16_default_device() {
    let mut device = Device::default();
    device.configure(FloatDType::F16).unwrap();
    let param = Param::from_tensor(Tensor::<1>::from_data([1.0], &device.clone().autodiff()));
    let adamw = AdamWConfig::new().with_amsgrad(true).with_weight_decay(0.0);
    let create = || {
        let policy = OptimizerPrecision::new(FloatDType::F32).with_state_storage(FloatDType::BF16);
        ModuleOptimizer::from(policy.wrap_with_master_weights(adamw.build()))
    };

    let mut original = create();
    // The second moment rounds to zero in F16, and the first weight update is
    // too small to change the model. The next update needs both saved values.
    let param = step(&mut original, param, 1.0 / 4096.0);
    let bytes = original.into_bytes().unwrap();
    let mut restored = create().from_bytes(bytes.clone()).unwrap();

    // Bare and wrapped AdamW read the same named statistics with different dtype policies.
    let bare = adamw.init().from_bytes(bytes).unwrap();
    let bare = Reader::from_bytes(bare.into_bytes().unwrap())
        .unwrap()
        .into_tensors()
        .unwrap();
    assert_eq!(bare.len(), 3);
    assert!(bare.iter().all(|tensor| tensor.dtype == DType::F16));

    let saved = Reader::from_bytes(restored.into_bytes().unwrap())
        .unwrap()
        .into_tensors()
        .unwrap();
    assert_eq!(saved.len(), 4);
    for tensor in saved {
        let dtype = if tensor.name.ends_with(".__precision.master") {
            DType::F32
        } else {
            DType::BF16
        };
        assert_eq!(tensor.dtype, dtype, "{}", tensor.name);
    }

    let expected = step(&mut original, param.clone(), 0.0);
    let actual = step(&mut restored, param, 0.0);
    assert_eq!(actual.to_data(), expected.to_data());
    assert_eq!(actual.val().into_scalar::<f32>(), 1.0 - 1.0 / 2048.0);

    let expected = Reader::from_bytes(original.into_bytes().unwrap()).unwrap();
    let actual = Reader::from_bytes(restored.into_bytes().unwrap()).unwrap();
    assert_eq!(actual.scalars(), expected.scalars());
    assert_eq!(actual.tensor_names(), expected.tensor_names());
    for name in expected.tensor_names() {
        assert_eq!(
            actual.tensor_data(name).unwrap(),
            expected.tensor_data(name).unwrap(),
            "{name}"
        );
    }
}
