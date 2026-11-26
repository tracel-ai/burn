//! A downstream statistic codec used with the same moment policy by Adam and AdamW.
use burn_core::{
    module::Param,
    tensor::{DType, Device, FloatDType, Int, IntDType, Tensor},
};
use burn_optim::{
    AdamConfig, AdamMomentStorage, AdamWConfig, GradientsParams, ModuleOptimizer,
    OptimizerPrecision, RecordState, StatisticCodec,
};

/// Use each row along the last dimension as a quantization block.
#[derive(Clone, Copy)]
struct RowwiseI8;

#[derive(Clone, RecordState)]
struct QuantizedRows<const D: usize> {
    codes: Tensor<D, Int>,
    scales: Tensor<D>,
}

impl StatisticCodec for RowwiseI8 {
    type Encoded<const D: usize> = QuantizedRows<D>;

    fn encode<const D: usize>(&self, tensor: Tensor<D>) -> QuantizedRows<D> {
        let tensor = tensor.cast(FloatDType::F32);
        let maxima = if D == 0 || tensor.shape().num_elements() == 0 {
            tensor.clone().abs()
        } else {
            tensor.clone().max_abs_dim(D - 1)
        };
        let scales = (maxima / 127.).clamp_min(f32::MIN_POSITIVE);
        let codes = (tensor / scales.clone()).round().cast(IntDType::I8);
        QuantizedRows { codes, scales }
    }

    fn decode<const D: usize>(&self, stored: QuantizedRows<D>, compute: FloatDType) -> Tensor<D> {
        (stored.codes.cast(FloatDType::F32) * stored.scales).cast(compute)
    }

    fn to_device<const D: usize>(stored: QuantizedRows<D>, device: &Device) -> QuantizedRows<D> {
        QuantizedRows {
            codes: stored.codes.to_device(device),
            scales: stored.scales.to_device(device),
        }
    }
}

#[test]
fn adam_and_adamw_resume_with_the_same_rowwise_first_moment_codec() {
    let device = Device::default();
    let moments = AdamMomentStorage::new(RowwiseI8, FloatDType::F32).with_maximum(FloatDType::BF16);
    let precision = OptimizerPrecision::new(FloatDType::F32).with_state_storage(moments);
    for amsgrad in [false, true] {
        let create = || {
            if amsgrad {
                ModuleOptimizer::from(
                    precision.wrap(
                        AdamWConfig::new()
                            .with_amsgrad(true)
                            .with_weight_decay(0.)
                            .with_beta_1(0.5)
                            .with_beta_2(0.5)
                            .with_epsilon(1e-8)
                            .build(),
                    ),
                )
            } else {
                ModuleOptimizer::from(
                    precision.wrap(
                        AdamConfig::new()
                            .with_beta_1(0.5)
                            .with_beta_2(0.5)
                            .with_epsilon(1e-8)
                            .build(),
                    ),
                )
            }
        };
        let param = Param::from_tensor(Tensor::<2>::ones([2, 2], &device.clone().autodiff()));
        let mut grads = GradientsParams::new();
        let ratio = 3. / 256.;
        grads.register(
            param.id,
            Tensor::<2>::from_data([[1., -ratio], [1. / 64., -ratio / 64.]], &device),
        );
        let mut optimizer = create();
        let param = optimizer.step(0.001, param, grads);
        let bytes = optimizer.into_bytes().unwrap();
        let record = burn_pack::Reader::from_bytes(bytes.clone())
            .unwrap()
            .into_tensors()
            .unwrap();
        assert_eq!(record.len(), if amsgrad { 4 } else { 3 });
        for tensor in record {
            let name = tensor.name.rsplit_once(".momentum.").unwrap().1;
            let dtype = match name {
                "moment_1.codes" => DType::I8,
                "moment_1.scales" | "moment_2" => DType::F32,
                "max_moment_2" if amsgrad => DType::BF16,
                _ => panic!("unexpected leaf {name}"),
            };
            assert_eq!(tensor.dtype, dtype, "{name}");
        }
        let mut resumed = create().from_bytes(bytes).unwrap();
        let mut grads = GradientsParams::new();
        grads.register(param.id, Tensor::<2>::zeros([2, 2], &device));
        let param = resumed.step(0.001, param, grads);
        let actual = param.val().try_into_vec_as::<f32>().unwrap();
        // Each small first moment rounds to code -1 at its own row scale. Variance
        // decays on the second step, while the AMSGrad maximum retains the first value.
        for (value, row_max) in [(actual[1], 1_f32), (actual[3], 1. / 64.)] {
            let g = row_max * ratio;
            let first_weight = 1. + 0.001 * g / (g + 1e-8);
            let variance = (if amsgrad { 0.5 } else { 0.25 }) * g * g / 0.75;
            let expected =
                first_weight + 0.001 * (0.25 * row_max / 127. / 0.75) / (variance.sqrt() + 1e-8);
            assert!(
                (value - expected).abs() < 1e-6,
                "{value}, expected {expected}"
            );
        }
    }
}
