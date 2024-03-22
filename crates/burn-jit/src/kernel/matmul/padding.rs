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
pub enum PaddingOutput<R: Runtime, E: JitElement, const D: usize> {
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
pub fn pad_round<R: Runtime, E: JitElement, const D: usize>(
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
pub fn padding<R: Runtime, E: JitElement + Element, const D: usize>(
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
pub fn crop<R: Runtime, E: JitElement, const D: usize>(
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
