use burn_core::nn::attention::{
    generate_causal_mask_1d, generate_chunked_causal_mask_1d, lengths_to_mask,
};
use burn_core::prelude::Backend;
use burn_core::tensor::TensorData;

type TB = burn_ndarray::NdArray<f32>;

#[test]
fn lengths_to_mask_basic() {
    let device = <TB as Backend>::Device::default();
    let mask = lengths_to_mask::<TB>(&[3, 1], 5, &device).into_data();
    mask.assert_eq(
        &TensorData::from([
            [false, false, false, true, true],
            [false, true, true, true, true],
        ]),
        false,
    );
}

#[test]
fn causal_mask_1d_has_future_masked() {
    let device = <TB as Backend>::Device::default();
    let mask = generate_causal_mask_1d::<TB>(4, &device).into_data();
    mask.assert_eq(
        &TensorData::from([
            [false, true, true, true],
            [false, false, true, true],
            [false, false, false, true],
            [false, false, false, false],
        ]),
        false,
    );
}

#[test]
fn chunked_causal_mask_respects_window() {
    let device = <TB as Backend>::Device::default();
    let mask = generate_chunked_causal_mask_1d::<TB>(8, 2, 1, &device).into_data();
    // Spot-check a few rows: allowed region should be unmasked (false), others masked (true)
    // Row 3: chunk_idx=1, start=(1-1)*2=0, end=min(4, i+1=4)=4 -> [0..4) allowed
    // Compare full matrix expectation on row 3 via slicing when available is cumbersome here,
    // so we assert a few positions directly.
    let md = mask.convert::<bool>();
    // Convert to flat bytes and index manually: row-major [i*L + j]
    let bytes = md.bytes;
    let l = 8usize;
    for j in 0..4 {
        let idx = 3 * l + j;
        assert_eq!(bytes[idx], 0u8); // false
    }
    for j in 4..8 {
        let idx = 3 * l + j;
        assert_eq!(bytes[idx], 1u8); // true
    }
}
