//! What the autotune checks compare two candidates' outputs by: the floats they hold, close enough.

use cubecl::client::Client;
use cubecl::ir::{ElemType, FloatKind};
use cubecl::std::tensor::TensorHandle;

/// Two candidates' outputs for one problem hold the same floats, within the tolerance two
/// summation orders differ by.
///
/// # Errors
///
/// Where they hold different counts of elements, or differ past the tolerance at one.
pub(crate) fn check_equivalent(
    client: &Client,
    expected: TensorHandle,
    actual: TensorHandle,
) -> Result<(), String> {
    let (expected, actual) = (read_f32(client, expected), read_f32(client, actual));
    if expected.len() != actual.len() {
        return Err(format!(
            "two candidates for one problem returned different element counts: {} vs {}",
            expected.len(),
            actual.len()
        ));
    }
    // Relative, with an absolute floor so values near zero do not fail on the last bit of a
    // denormal.
    const RELATIVE: f32 = 1e-2;
    const ABSOLUTE: f32 = 1e-4;
    let differing = expected.iter().zip(actual.iter()).position(|(want, got)| {
        let tolerance = ABSOLUTE + RELATIVE * want.abs();
        !((want - got).abs() <= tolerance || (want.is_nan() && got.is_nan()))
    });
    match differing {
        Some(index) => Err(format!(
            "candidates disagree at element {index}: {} vs {}",
            expected[index], actual[index]
        )),
        None => Ok(()),
    }
}

/// `tensor`'s buffer read back as `f32` from whatever float it holds.
fn read_f32(client: &Client, tensor: TensorHandle) -> Vec<f32> {
    let bytes = client
        .read_one(tensor.handle)
        .expect("the output reads back");
    match tensor.dtype {
        ElemType::Float(FloatKind::F32) => bytemuck::cast_slice::<u8, f32>(&bytes).to_vec(),
        ElemType::Float(FloatKind::F16) => bytemuck::cast_slice::<u8, half::f16>(&bytes)
            .iter()
            .map(|value| value.to_f32())
            .collect(),
        ElemType::Float(FloatKind::BF16) => bytemuck::cast_slice::<u8, half::bf16>(&bytes)
            .iter()
            .map(|value| value.to_f32())
            .collect(),
        elem => panic!("an output is a float the routines write, not {elem:?}"),
    }
}
