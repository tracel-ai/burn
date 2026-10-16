mod adaptive_avg_pool1d;
mod adaptive_avg_pool2d;
mod adaptive_avg_pool3d;
mod avg_pool1d;
mod avg_pool2d;
mod lp_pool1d;
mod lp_pool2d;
mod max_pool1d;
mod max_pool2d;

/// Calculate the output length for a pooling dimension.
pub(super) fn output_size(input: usize, kernel: usize, stride: usize, ceil_mode: bool) -> usize {
    assert!(
        input > 0,
        "Lp pooling does not support an empty input dimension"
    );
    if ceil_mode {
        if input <= kernel {
            1
        } else {
            let output = (input - kernel).div_ceil(stride) + 1;
            if output > 1 && (output - 1) * stride >= input {
                output - 1
            } else {
                output
            }
        }
    } else {
        assert!(
            input >= kernel,
            "kernel size must not exceed the input size"
        );
        (input - kernel) / stride + 1
    }
}

pub use adaptive_avg_pool1d::*;
pub use adaptive_avg_pool2d::*;
pub use adaptive_avg_pool3d::*;
pub use avg_pool1d::*;
pub use avg_pool2d::*;
pub use lp_pool1d::*;
pub use lp_pool2d::*;
pub use max_pool1d::*;
pub use max_pool2d::*;
