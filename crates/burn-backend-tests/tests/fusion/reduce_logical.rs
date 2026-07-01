//! Fusion-path correctness for `any`/`all` reductions.
//!
//! Under `Fusion<CubeBackend>` these route through the recorded `AnyDim`/`AllDim`
//! IR into cubek's dedicated Any (max-of-flags) / All (min-of-flags) instruction,
//! instead of the default `bool->int->sum->compare` emulation. The reduced
//! dimension is kept wide enough to exercise the multi-block (plane/cube) merge.

use super::*;
use burn_tensor::TensorData;

const ROWS: usize = 4;
// Wide reduced dim so the reduction spans several plane/cube blocks that must merge.
const COLS: usize = 257;

/// Deterministic mask: `(r + c) % stride == 0`. With `stride == 3` every row has a
/// mix of set/unset flags; `stride == ROWS * COLS + 1` makes the whole tensor all-false.
fn mask(rows: usize, cols: usize, stride: usize) -> Vec<bool> {
    (0..rows * cols)
        .map(|i| (i / cols + i % cols) % stride == 0)
        .collect()
}

/// Non-zero <-> flag set; mixes magnitudes/signs so the flag normalization is exercised.
fn floats(m: &[bool]) -> Vec<f32> {
    m.iter()
        .enumerate()
        .map(|(i, &b)| {
            if b {
                if i % 2 == 0 { 2.0 } else { -3.0 }
            } else {
                0.0
            }
        })
        .collect()
}

fn ints(m: &[bool]) -> Vec<i32> {
    m.iter()
        .enumerate()
        .map(|(i, &b)| {
            if b {
                if i % 2 == 0 { 2 } else { -3 }
            } else {
                0
            }
        })
        .collect()
}

fn expect_any_dim(m: &[bool], rows: usize, cols: usize) -> TensorData {
    let v: Vec<bool> = (0..rows)
        .map(|r| (0..cols).any(|c| m[r * cols + c]))
        .collect();
    TensorData::new(v, [rows, 1])
}

fn expect_all_dim(m: &[bool], rows: usize, cols: usize) -> TensorData {
    let v: Vec<bool> = (0..rows)
        .map(|r| (0..cols).all(|c| m[r * cols + c]))
        .collect();
    TensorData::new(v, [rows, 1])
}

// ---- dim reductions ----------------------------------------------------------

#[test]
fn bool_any_dim_fused() {
    let device = Default::default();
    let m = mask(ROWS, COLS, 3);
    let tensor = TestTensorBool::<2>::from_data(TensorData::new(m.clone(), [ROWS, COLS]), &device);
    device.sync().unwrap();

    let actual = tensor.any_dim(1).into_data();
    expect_any_dim(&m, ROWS, COLS).assert_eq(&actual, false);
}

#[test]
fn bool_all_dim_fused() {
    let device = Default::default();
    let m = mask(ROWS, COLS, 3);
    let tensor = TestTensorBool::<2>::from_data(TensorData::new(m.clone(), [ROWS, COLS]), &device);
    device.sync().unwrap();

    let actual = tensor.all_dim(1).into_data();
    expect_all_dim(&m, ROWS, COLS).assert_eq(&actual, false);
}

#[test]
fn float_any_dim_fused() {
    let device = Default::default();
    let m = mask(ROWS, COLS, 3);
    let tensor = TestTensor::<2>::from_data(TensorData::new(floats(&m), [ROWS, COLS]), &device);
    device.sync().unwrap();

    let actual = tensor.any_dim(1).into_data();
    expect_any_dim(&m, ROWS, COLS).assert_eq(&actual, false);
}

#[test]
fn float_all_dim_fused() {
    let device = Default::default();
    let m = mask(ROWS, COLS, 3);
    let tensor = TestTensor::<2>::from_data(TensorData::new(floats(&m), [ROWS, COLS]), &device);
    device.sync().unwrap();

    let actual = tensor.all_dim(1).into_data();
    expect_all_dim(&m, ROWS, COLS).assert_eq(&actual, false);
}

#[test]
fn int_any_dim_fused() {
    let device = Default::default();
    let m = mask(ROWS, COLS, 3);
    let tensor = TestTensorInt::<2>::from_data(TensorData::new(ints(&m), [ROWS, COLS]), &device);
    device.sync().unwrap();

    let actual = tensor.any_dim(1).into_data();
    expect_any_dim(&m, ROWS, COLS).assert_eq(&actual, false);
}

#[test]
fn int_all_dim_fused() {
    let device = Default::default();
    let m = mask(ROWS, COLS, 3);
    let tensor = TestTensorInt::<2>::from_data(TensorData::new(ints(&m), [ROWS, COLS]), &device);
    device.sync().unwrap();

    let actual = tensor.all_dim(1).into_data();
    expect_all_dim(&m, ROWS, COLS).assert_eq(&actual, false);
}

// ---- whole-tensor reductions (flatten -> any_dim(0)) -------------------------

#[test]
fn bool_any_all_whole() {
    let device = Default::default();
    let mixed = mask(ROWS, COLS, 3); // at least one set and one unset flag
    let all_false = vec![false; ROWS * COLS];
    let all_true = vec![true; ROWS * COLS];

    let t_mixed = TestTensorBool::<2>::from_data(TensorData::new(mixed, [ROWS, COLS]), &device);
    let t_false = TestTensorBool::<2>::from_data(TensorData::new(all_false, [ROWS, COLS]), &device);
    let t_true = TestTensorBool::<2>::from_data(TensorData::new(all_true, [ROWS, COLS]), &device);
    device.sync().unwrap();

    TensorData::new(vec![true], [1]).assert_eq(&t_mixed.clone().any().into_data(), false);
    TensorData::new(vec![false], [1]).assert_eq(&t_mixed.all().into_data(), false);
    TensorData::new(vec![false], [1]).assert_eq(&t_false.any().into_data(), false);
    TensorData::new(vec![true], [1]).assert_eq(&t_true.all().into_data(), false);
}

#[test]
fn float_int_whole() {
    let device = Default::default();
    let mixed = mask(ROWS, COLS, 3);
    let tf = TestTensor::<2>::from_data(TensorData::new(floats(&mixed), [ROWS, COLS]), &device);
    let ti = TestTensorInt::<2>::from_data(TensorData::new(ints(&mixed), [ROWS, COLS]), &device);
    device.sync().unwrap();

    TensorData::new(vec![true], [1]).assert_eq(&tf.any().into_data(), false);
    TensorData::new(vec![true], [1]).assert_eq(&ti.any().into_data(), false);
}

// ---- fuse-on-read: a reshape feeding a bool reduce ---------------------------

#[test]
fn bool_reshape_then_any_dim() {
    let device = Default::default();
    let m = mask(ROWS, COLS, 3);
    let tensor = TestTensorBool::<2>::from_data(TensorData::new(m.clone(), [ROWS, COLS]), &device);
    device.sync().unwrap();

    // Reshape (a layout-changing read op) feeding the reduce, then reduce the last dim.
    let reshaped = tensor.reshape([COLS, ROWS]);
    let actual = reshaped.any_dim(1).into_data();

    let expected: Vec<bool> = (0..COLS)
        .map(|a| (0..ROWS).any(|b| m[a * ROWS + b]))
        .collect();
    TensorData::new(expected, [COLS, 1]).assert_eq(&actual, false);
}

// ---- fuse-on-read: an elementwise op feeding the reduce ----------------------

#[test]
fn float_elemwise_then_any_dim() {
    let device = Default::default();
    let m = mask(ROWS, COLS, 3);
    let tensor = TestTensor::<2>::from_data(TensorData::new(floats(&m), [ROWS, COLS]), &device);
    device.sync().unwrap();

    // Scaling preserves the non-zero pattern, so the flags (and the result) are
    // unchanged; the multiply should fuse into the reduce's read path.
    let actual = tensor.mul_scalar(5.0).any_dim(1).into_data();
    expect_any_dim(&m, ROWS, COLS).assert_eq(&actual, false);
}
