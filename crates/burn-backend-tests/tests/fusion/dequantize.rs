use super::*;

const VALUES: [f32; 12] = [
    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
];

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
        with_drop.assert_eq(&without_drop, true);
    });
}
