use super::*;
use burn_tensor::quantization::{
    BlockExtent, BlockScale, BlockSize, QuantScheme, QuantStore, QuantValue, ScaleDtype,
};
use burn_tensor::{TensorData, Tolerance};

const VALUES: [f32; 12] = [
    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
];

#[test]
fn a_fused_dequantize_reads_a_scale_per_row() {
    test_stream().executes(|| {
        let device: burn_tensor::Device = Default::default();
        let row = BlockSize::with_rest(&[BlockExtent::WholeAxis], BlockExtent::VALUE);
        let scheme = QuantScheme::default()
            .with_value(QuantValue::Q8S)
            .with_levels([BlockScale::per_block(row, ScaleDtype::F32)])
            .with_store(QuantStore::PackedU32(0));
        let rows = [VALUES, VALUES.map(|value| value * 1000.0)];
        let quant = TestTensor::<2>::from_data(rows, &device).quantize_dynamic(&scheme);
        let zeros = TestTensor::<2>::zeros([2, VALUES.len()], &device);
        let output = zeros + quant.dequantize();
        output
            .into_data()
            .assert_approx_eq::<FloatElem>(&TensorData::from(rows), Tolerance::rel_abs(0.05, 0.05));
    });
}

#[test]
fn dequantize_is_unchanged_by_input_drop() {
    test_stream().executes(|| {
        let device: burn_tensor::Device = Default::default();
        let run = |drop_input| {
            let source = TestTensor::<2>::from_data([VALUES], &device);
            let quant = source.quantize_dynamic(&device.settings().quantization.scheme);
            let input = TestTensor::<2>::from_data(VALUES.map(|v| [v]), &device).swap_dims(0, 1);
            let output = input.clone() + quant.dequantize();
            if drop_input {
                drop(input);
            }
            output.into_data()
        };
        let without_drop = run(false);
        let with_drop = run(true);
        with_drop.assert_approx_eq::<FloatElem>(&without_drop, Tolerance::default());
    });
}
