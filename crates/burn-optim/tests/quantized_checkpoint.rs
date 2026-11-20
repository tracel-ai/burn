use burn_core::{
    module::Param,
    tensor::{DType, Device, FloatDType, Tensor},
};
use burn_optim::{GradientsParams, ModuleOptimizer, OptimizerPrecision, OptimizerStateStorage};

#[test]
fn quantized_checkpoint_decodes_directly_to_compute_dtype() {
    let mut device = Device::default();
    device.configure(FloatDType::F16).unwrap();
    // Quantized moments must decode straight to F32 rather than underflowing in the default F16.
    use burn_core::tensor::quantization::{QuantScheme, QuantStore, QuantValue};
    let scheme = QuantScheme::default()
        .with_value(QuantValue::Q8S)
        .with_store(QuantStore::Native);
    let create = || {
        ModuleOptimizer::from(
            OptimizerPrecision {
                compute: FloatDType::F32,
                master_weights: true,
                state: OptimizerStateStorage::Quantized(scheme),
            }
            .wrap(
                burn_optim::LionConfig::new()
                    .with_beta_1(0.5)
                    .with_beta_2(0.5)
                    .build(),
            ),
        )
    };
    let mut optimizer = create();
    let param = Param::from_tensor(Tensor::<1>::from_data([1.0], &device.clone().autodiff()));
    let mut grads = GradientsParams::new();
    grads.register(
        param.id,
        Tensor::<1>::from_data([1e-10_f32], (&device, DType::F32)),
    );
    let param = optimizer.step(0.125, param, grads);
    let mut restored = create()
        .from_bytes(optimizer.into_bytes().unwrap())
        .unwrap();
    let mut grads = GradientsParams::new();
    grads.register(
        param.id,
        Tensor::<1>::from_data([-4e-11_f32], (&device, DType::F32)),
    );
    let param = restored.step(0.125, param, grads);
    assert_eq!(param.val().into_scalar::<f32>(), 0.75);
}
