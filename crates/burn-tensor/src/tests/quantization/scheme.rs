#[burn_tensor_testgen::testgen(scheme)]
mod tests {
    use super::*;
    use burn_tensor::{
        DType, Element, Tensor, TensorData,
        quantization::{
            CalibrationRange, QuantFloatPrecision, QuantInputType, QuantLevel, QuantMode,
            QuantPropagation, QuantScheme,
        },
    };
    use burn_tensor::{Tolerance, ops::FloatElem};
    type FT = FloatElem<TestBackend>;

    #[test]
    fn per_tensor_symmetric_int8() {
        let device = Default::default();
        let scheme = QuantScheme::default();
        let range = CalibrationRange {
            min: TestTensor::<1>::from_floats([0.5], &device),
            max: TestTensor::<1>::from_floats([1.8], &device),
        };

        let qparams = scheme.compute_q_params(range);

        qparams
            .scales
            .into_data()
            .assert_approx_eq::<FT>(&TensorData::from([0.014_173_23]), Tolerance::default());
    }

    #[test]
    fn per_block_symmetric_int8() {
        let device = Default::default();
        let scheme = QuantScheme::default().set_level(QuantLevel::Block(4));
        let range = CalibrationRange {
            min: TestTensor::<1>::from_floats([-1.8, -0.5, 0.01, -0.04], &device),
            max: TestTensor::<1>::from_floats([0.5, 1.8, 0.04, -0.01], &device),
        };

        let qparams = scheme.compute_q_params(range);

        qparams.scales.into_data().assert_approx_eq::<FT>(
            &TensorData::from([0.014_173_23, 0.014_173_23, 0.000_314_96, 0.000_314_96]),
            Tolerance::default(),
        );
    }

    #[test]
    fn quant_scheme_should_propagate() {
        let device = Default::default();
        let scheme = QuantScheme {
            propagation: QuantPropagation::Propagate,
            ..Default::default()
        };

        let tensor_1 = TestTensor::<2>::from_floats([[1.0, 6.35], [2.0, 3.0], [1.0, 3.0]], &device)
            .quantize_dynamic(&scheme);
        let tensor_2 = TestTensor::<2>::from_floats([[4.0, 8.0, 12.7], [2.0, 3.0, 6.0]], &device)
            .quantize_dynamic(&scheme);

        let tensor_3 = tensor_1.matmul(tensor_2);
        assert_eq!(tensor_3.to_data().dtype, DType::QFloat(scheme));

        let tensor_4 = tensor_3.add_scalar(1.);
        assert_eq!(tensor_4.to_data().dtype, DType::QFloat(scheme));
    }

    #[test]
    fn quant_scheme_should_not_propagate() {
        let device = Default::default();
        let scheme = QuantScheme {
            propagation: QuantPropagation::Inhibit,
            acc_precision: QuantFloatPrecision::F32,
            ..Default::default()
        };

        let tensor_1 = TestTensor::<2>::from_floats([[1.0, 6.35], [2.0, 3.0], [1.0, 3.0]], &device)
            .quantize_dynamic(&scheme);
        let tensor_2 = TestTensor::<2>::from_floats([[4.0, 8.0, 12.7], [2.0, 3.0, 6.0]], &device)
            .quantize_dynamic(&scheme);

        // Some ops like reshape, swap_dims, permute, expand, select, slice, etc. do not affect
        // the propagation. It mostly applies to compute kernels.
        let tensor_1 = tensor_1
            .permute([1, 0])
            .swap_dims(0, 1)
            .reshape([1, 6])
            .reshape([3, 2]);
        assert_eq!(tensor_1.to_data().dtype, DType::QFloat(scheme));

        // When propagation is not desired, compute kernels like matmul should return tensor
        // in floating point precision
        let tensor_3 = tensor_1.matmul(tensor_2);
        let dtype = tensor_3.to_data().dtype;
        assert!(dtype.is_float());

        // Subsequent ops will therefore be performed on floats
        let tensor_4 = tensor_3.add(TestTensor::<2>::ones([3, 3], &device).cast(dtype));
        assert!(tensor_4.to_data().dtype.is_float());
    }
}
