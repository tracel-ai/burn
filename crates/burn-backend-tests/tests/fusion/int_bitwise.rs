use super::*;
use burn_fusion::inspect::{BlockKind, FusionInspector, matchers};
use burn_tensor::TensorData;

#[test]
fn int_bitwise_chain_and_float_cast_fuse_into_single_kernel() {
    let stream = test_stream();
    stream.executes(|| {
        let device = Default::default();

        let lhs = TestTensorInt::<1>::from_data([13, 7, 3, 12], &device);
        let rhs = TestTensorInt::<1>::from_data([11, 3, 5, 9], &device);
        device.sync().unwrap();

        let inspector = FusionInspector::install(stream);

        let bitwise = lhs
            .clone()
            .bitwise_xor(rhs)
            .bitwise_and_scalar(7)
            .bitwise_right_shift_scalar(1)
            .bitwise_or(lhs.bitwise_not());

        let output = bitwise.float().mul_scalar(0.5);
        let dtype = output.dtype();
        output
            .into_data()
            .assert_eq(&TensorData::from([-6.5_f32, -3.0, -0.5, -6.5]), false);
        device.sync().unwrap();

        let reports = inspector.drain();
        let tables = reports
            .iter()
            .map(|report| report.format_table())
            .collect::<Vec<_>>()
            .join("\n\n");

        let block = reports
            .iter()
            .flat_map(|report| report.blocks.iter())
            .find(|block| block.operations.iter().any(matchers::is_bitwise_xor_int()))
            .unwrap_or_else(|| panic!("no bitwise fused block found\n\n{tables}"));

        assert!(
            matches!(
                block.kind,
                BlockKind::Fused {
                    name: "ElementWise",
                    ..
                }
            ),
            "expected ElementWise fused block, got {:?}\n\n{tables}",
            block.kind,
        );

        assert!(
            block.operations.iter().any(matchers::is_bitwise_xor_int()),
            "BitwiseXor missing from fused block\n\n{tables}",
        );
        assert!(
            block
                .operations
                .iter()
                .any(matchers::is_bitwise_and_scalar_int()),
            "BitwiseAndScalar missing from fused block\n\n{tables}",
        );
        assert!(
            block
                .operations
                .iter()
                .any(matchers::is_bitwise_right_shift_scalar_int()),
            "BitwiseRightShiftScalar missing from fused block\n\n{tables}",
        );
        assert!(
            block.operations.iter().any(matchers::is_bitwise_not_int()),
            "BitwiseNot missing from fused block\n\n{tables}",
        );
        assert!(
            block.operations.iter().any(matchers::is_bitwise_or_int()),
            "BitwiseOr missing from fused block\n\n{tables}",
        );
        assert!(
            block.operations.iter().any(matchers::is_int_into_float()),
            "IntoFloat missing from fused block\n\n{tables}",
        );
        assert!(
            block
                .operations
                .iter()
                .any(matchers::is_mul_scalar_float(dtype)),
            "MulScalar missing from fused block\n\n{tables}",
        );
    });
}
