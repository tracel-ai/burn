use crate::{
    element::JitElement,
    kernel::{slice_assign, slice_on_output},
    ops::numeric::zeros_device,
    tensor::JitTensor,
    Runtime,
};
use burn_tensor::{Element, Shape};
use std::ops::Range;

// Output of the pad_round function. Allows to know explicitly if early return occurred
pub(super) enum PaddingOutput<R: Runtime, E: JitElement, const D: usize> {
    Padded(JitTensor<R, E, D>),
    Unchanged(JitTensor<R, E, D>),
}

impl<R: Runtime, E: JitElement, const D: usize> PaddingOutput<R, E, D> {
    pub fn into_tensor(self) -> JitTensor<R, E, D> {
        match self {
            PaddingOutput::Padded(tensor) => tensor,
            PaddingOutput::Unchanged(tensor) => tensor,
        }
    }
}

/// Pads tensor with zeros to make tensor number of rows and columns
/// divisible by some quantity.
/// For instance tensor of shape [1000, 1000] with divisors 64 and 64
/// will be padded to [1024, 1024] with the last 24 elements being zeros
pub(super) fn pad_round<R: Runtime, E: JitElement, const D: usize>(
    tensor: JitTensor<R, E, D>,
    row_divisor: usize,
    col_divisor: usize,
) -> PaddingOutput<R, E, D> {
    let previous_row_dim = tensor.shape.dims[D - 2];
    let previous_col_dim = tensor.shape.dims[D - 1];
    let row_modulo = previous_row_dim % row_divisor;
    let col_modulo = previous_col_dim % col_divisor;

    let new_row_dim = match row_modulo {
        0 => previous_row_dim,
        _ => previous_row_dim + row_divisor - row_modulo,
    };
    let new_col_dim = match col_modulo {
        0 => previous_col_dim,
        _ => previous_col_dim + col_divisor - col_modulo,
    };
    if previous_row_dim == new_row_dim && previous_col_dim == new_col_dim {
        return PaddingOutput::Unchanged(tensor);
    }

    let mut padded_shape = Vec::with_capacity(D);
    for i in 0..D - 2 {
        padded_shape.push(tensor.shape.dims[i]);
    }
    padded_shape.push(new_row_dim);
    padded_shape.push(new_col_dim);

    PaddingOutput::Padded(padding::<R, E, D>(tensor, padded_shape.into()))
}

/// Pads tensor by adding zeros when padded dim is larger than tensor dim
fn padding<R: Runtime, E: JitElement + Element, const D: usize>(
    tensor: JitTensor<R, E, D>,
    padded_shape: Shape<D>,
) -> JitTensor<R, E, D> {
    let ranges = padded_shape
        .dims
        .iter()
        .map(|dim| 0..*dim)
        .collect::<Vec<Range<usize>>>()
        .try_into()
        .unwrap();

    slice_assign::<R, E, D, D>(
        zeros_device::<R, E, D>(tensor.client.clone(), tensor.device.clone(), padded_shape),
        ranges,
        tensor,
    )
}

/// Crops tensor by deleting values when cropped dim is smaller than tensor dim
pub(super) fn crop<R: Runtime, E: JitElement, const D: usize>(
    tensor: JitTensor<R, E, D>,
    output: JitTensor<R, E, D>,
) -> JitTensor<R, E, D> {
    let ranges = output
        .shape
        .dims
        .iter()
        .map(|dim| 0..*dim)
        .collect::<Vec<Range<usize>>>()
        .try_into()
        .unwrap();
    slice_on_output::<R, E, D, D>(tensor, output, ranges)
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::tests::TestTensor;

    #[test]
    fn padding_already_round_should_have_same_shape() {
        let row = 10;
        let row_divisor = 5;
        let col = 12;
        let col_divisor = 3;
        let tensor = TestTensor::random(
            [row, col],
            burn_tensor::Distribution::Default,
            &Default::default(),
        );
        let expected_shape = [row, col].into();

        let padded = pad_round(tensor.into_primitive(), row_divisor, col_divisor).into_tensor();

        assert!(padded.shape == expected_shape);
    }

    #[test]
    fn padding_already_round_should_have_same_values() {
        let row = 10;
        let row_divisor = 5;
        let col = 12;
        let col_divisor = 3;
        let tensor = TestTensor::random(
            [row, col],
            burn_tensor::Distribution::Default,
            &Default::default(),
        );

        let padded = pad_round(tensor.clone().into_primitive(), row_divisor, col_divisor);

        let padded = TestTensor::from_primitive(padded.into_tensor());
        padded.into_data().assert_approx_eq(&tensor.into_data(), 3);
    }

    #[test]
    fn padding_not_round_should_have_rounded_shape() {
        let row = 10;
        let row_divisor = 6;
        let col = 12;
        let col_divisor = 5;
        let tensor = TestTensor::random(
            [row, col],
            burn_tensor::Distribution::Default,
            &Default::default(),
        );
        let expected_shape = [12, 15].into();

        let padded = pad_round(tensor.into_primitive(), row_divisor, col_divisor).into_tensor();

        assert!(padded.shape == expected_shape);
    }

    #[test]
    fn padding_not_round_should_have_same_values() {
        let row = 10;
        let row_divisor = 6;
        let col = 12;
        let col_divisor = 5;
        let tensor = TestTensor::random(
            [row, col],
            burn_tensor::Distribution::Default,
            &Default::default(),
        );

        let padded =
            pad_round(tensor.clone().into_primitive(), row_divisor, col_divisor).into_tensor();

        let padded = TestTensor::from_primitive(padded).to_data();
        let tensor = tensor.into_data();
        for i in 0..row {
            for j in 0..col {
                assert!(padded.value[i * 15 + j] == tensor.value[i * col + j]);
            }
        }
    }

    #[test]
    fn padding_not_round_should_have_zero_padding() {
        let row = 10;
        let row_divisor = 6;
        let col = 12;
        let col_divisor = 5;
        let tensor = TestTensor::random(
            [row, col],
            burn_tensor::Distribution::Default,
            &Default::default(),
        );

        let padded = pad_round(tensor.into_primitive(), row_divisor, col_divisor).into_tensor();
        let padded = TestTensor::from_primitive(padded).to_data();

        // check right of matrix
        for i in 0..row {
            for j in col..15 {
                assert!(padded.value[i * 15 + j] == 0.0);
            }
        }
        // check below matrix, including bottom right
        for i in row..12 {
            for j in 0..15 {
                assert!(padded.value[i * 15 + j] == 0.0);
            }
        }
    }

    #[test]
    fn padding_works_with_batch() {
        let row = 10;
        let row_divisor = 4;
        let col = 12;
        let col_divisor = 5;
        let tensor = TestTensor::random(
            [2, 3, row, col],
            burn_tensor::Distribution::Default,
            &Default::default(),
        );
        let expected_shape = [2, 3, 12, 15].into();

        let padded = pad_round(tensor.into_primitive(), row_divisor, col_divisor).into_tensor();

        assert!(padded.shape == expected_shape);
    }

    #[test]
    fn padding_with_row_divisor_larger_than_row() {
        let row = 10;
        let row_divisor = 32;
        let col = 4;
        let col_divisor = 3;
        let tensor = TestTensor::random(
            [row, col],
            burn_tensor::Distribution::Default,
            &Default::default(),
        );
        let expected_shape = [row_divisor, 2 * col_divisor].into();

        let padded = pad_round(tensor.into_primitive(), row_divisor, col_divisor).into_tensor();

        assert!(padded.shape == expected_shape);
    }

    #[test]
    fn padding_with_row_divisor_equal_to_row_but_col_must_be_padded() {
        let row = 32;
        let row_divisor = 32;
        let col = 4;
        let col_divisor = 64;
        let tensor = TestTensor::random(
            [row, col],
            burn_tensor::Distribution::Default,
            &Default::default(),
        );
        let expected_shape = [32, 64].into();

        let padded = pad_round(tensor.into_primitive(), row_divisor, col_divisor).into_tensor();

        assert!(padded.shape == expected_shape);
    }

    #[test]
    fn crop_same_shape_should_be_unchanged_shape() {
        let row = 10;
        let col = 12;
        let device = Default::default();
        let tensor = TestTensor::random([row, col], burn_tensor::Distribution::Default, &device);
        let expected_shape = [row, col].into();

        let unpadded = crop(
            tensor.clone().into_primitive(),
            TestTensor::empty([row, col], &device).into_primitive(),
        );

        assert!(unpadded.shape == expected_shape);
    }

    #[test]
    fn crop_same_shape_should_have_unchanged_values() {
        let row = 10;
        let col = 12;
        let device = Default::default();
        let tensor = TestTensor::random([row, col], burn_tensor::Distribution::Default, &device);

        let unpadded = crop(
            tensor.clone().into_primitive(),
            TestTensor::empty([row, col], &device).into_primitive(),
        );

        let unpadded = TestTensor::from_primitive(unpadded).to_data();
        let tensor = tensor.into_data();
        for i in 0..row {
            for j in 0..col {
                assert!(unpadded.value[i * col + j] == tensor.value[i * col + j]);
            }
        }
    }

    #[test]
    fn crop_should_decrease_shape() {
        let row = 10;
        let keep_rows = 8;
        let col = 12;
        let keep_cols = 10;
        let device = Default::default();
        let tensor = TestTensor::random([row, col], burn_tensor::Distribution::Default, &device);
        let expected_shape = [keep_rows, keep_cols].into();

        let unpadded = crop(
            tensor.clone().into_primitive(),
            TestTensor::empty([keep_rows, keep_cols], &device).into_primitive(),
        );

        assert!(unpadded.shape == expected_shape);
    }

    #[test]
    fn crop_should_keep_same_values() {
        let row = 4;
        let keep_rows = 3;
        let col = 4;
        let keep_cols = 3;
        let device = Default::default();
        let tensor = TestTensor::random([row, col], burn_tensor::Distribution::Default, &device);

        let unpadded = crop(
            tensor.clone().into_primitive(),
            TestTensor::empty([keep_rows, keep_cols], &device).into_primitive(),
        );

        let unpadded = TestTensor::from_primitive(unpadded).to_data();
        let tensor = tensor.into_data();

        for i in 0..keep_rows {
            for j in 0..keep_cols {
                assert!(unpadded.value[i * keep_cols + j] == tensor.value[i * col + j]);
            }
        }
    }
}
