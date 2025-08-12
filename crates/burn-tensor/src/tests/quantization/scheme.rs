#[burn_tensor_testgen::testgen(scheme)]
mod tests {
    use super::*;
    use burn_tensor::{
        DType, Element, Tensor, TensorData,
        quantization::{
            CalibrationRange, QuantLevel, QuantMode, QuantParam, QuantPropagation, QuantScheme,
            QuantStore, QuantValue, compute_q_params,
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

        let qparams = compute_q_params(&scheme, range);

        qparams
            .scales
            .into_data()
            .assert_approx_eq::<FT>(&TensorData::from([0.014_173_23]), Tolerance::default());
    }

    #[test]
    fn per_block_symmetric_int8() {
        let device = Default::default();
        let scheme = QuantScheme::default().with_level(QuantLevel::Block(4));
        let range = CalibrationRange {
            min: TestTensor::<1>::from_floats([-1.8, -0.5, 0.01, -0.04], &device),
            max: TestTensor::<1>::from_floats([0.5, 1.8, 0.04, -0.01], &device),
        };

        let qparams = compute_q_params(&scheme, range);

        qparams.scales.into_data().assert_approx_eq::<FT>(
            &TensorData::from([0.014_173_23, 0.014_173_23, 0.000_314_96, 0.000_314_96]),
            Tolerance::default(),
        );
    }

    #[test]
    fn quant_scheme_should_inhibit_by_default() {
        let device = Default::default();
        let scheme = QuantScheme::default();

        let tensor_1 = TestTensor::<2>::from_floats(
            [[1.0, 6.35, 0., 0.], [2.0, 3.0, 0., 0.], [1.0, 3.0, 0., 0.]],
            &device,
        )
        .quantize_dynamic(&scheme);
        let tensor_2 = TestTensor::<2>::from_floats(
            [
                [4.0, 8.0, 12.7, 0.],
                [2.0, 3.0, 6.0, 0.],
                [0., 0., 0., 0.],
                [0., 0., 0., 0.],
            ],
            &device,
        )
        .quantize_dynamic(&scheme);

        // let tensor_3 = tensor_1.clone().matmul(tensor_2);
        // assert_eq!(tensor_3.to_data().dtype, FT::dtype());

        let tensor_4 = tensor_1.add_scalar(1.);
        assert_eq!(tensor_4.to_data().dtype, FT::dtype());
    }
}
