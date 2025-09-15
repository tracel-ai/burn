use burn_core::nn::attention::generate_windowed_causal_mask;
use burn_core::prelude::Backend;
use burn_core::tensor::TensorData;
type TB = burn_ndarray::NdArray<f32>;

#[test]
fn windowed_causal_mask_diagonal_and_future() {
    let device = <TB as Backend>::Device::default();
    let b = 1;
    let t = 6;
    let mask = generate_windowed_causal_mask::<TB>(b, t, Some(2), 1, &device).squeeze::<2>(0);
    // Diagonal should never be masked
    for i in 0..t {
        let diag = mask.clone().slice([i..i + 1, i..i + 1]).into_data();
        let exp = TensorData::from([[false]]);
        diag.assert_eq(&exp, false);
    }
    // Future positions must be masked: j > i -> true
    for i in 0..t {
        for j in i + 1..t {
            let val = mask.clone().slice([i..i + 1, j..j + 1]).into_data();
            let exp = TensorData::from([[true]]);
            val.assert_eq(&exp, false);
        }
    }
}
