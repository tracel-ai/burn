use burn_core::{
    module::Param,
    tensor::{DType, Device, FloatDType, Tensor},
};
use burn_optim::{
    GradientsParams, ModuleOptimizer, OptimizerPrecision, OptimizerStateMapper, StatePath,
};

use burn_core::tensor::quantization::{QuantScheme, QuantStore, QuantValue, ScaleDtype};

#[derive(Clone)]
struct Quantized(QuantScheme);

impl OptimizerStateMapper for Quantized {
    fn map<const D: usize>(&self, tensor: Tensor<D>, _: &StatePath<'_>) -> Tensor<D> {
        tensor.quantize_dynamic(&self.0)
    }
}

#[test]
fn quantized_checkpoint_decodes_directly_to_compute_dtype() {
    let mut device = Device::default();
    device.configure(FloatDType::F16).unwrap();
    // Quantized moments must decode straight to F32 rather than underflowing in the default F16.
    let scheme = QuantScheme::default()
        .with_value(QuantValue::Q8S)
        .with_store(QuantStore::Native)
        .per_tensor(ScaleDtype::F32);
    let create = || {
        ModuleOptimizer::from(
            OptimizerPrecision::new(FloatDType::F32)
                .with_state_storage(Quantized(scheme))
                .wrap_with_master_weights(
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
    let bytes = optimizer.into_bytes().unwrap();
    let bare = burn_optim::LionConfig::new()
        .init()
        .from_bytes(bytes.clone())
        .unwrap();
    let tensors = burn_pack::Reader::from_bytes(bare.into_bytes().unwrap())
        .unwrap()
        .into_tensors()
        .unwrap();
    assert_eq!(tensors[0].dtype, DType::F16);
    let mut restored = create().from_bytes(bytes).unwrap();
    let mut grads = GradientsParams::new();
    grads.register(
        param.id,
        Tensor::<1>::from_data([-4e-11_f32], (&device, DType::F32)),
    );
    let param = restored.step(0.125, param, grads);
    assert_eq!(param.val().into_scalar::<f32>(), 0.75);
}
