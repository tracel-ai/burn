use crate::{backend::Backend, Int, Shape, Tensor};
use alloc::vec::Vec;

/// Generates a cartesian grid for the given tensor shape on the specified device.
/// The generated tensor is of dimension `D2 = D + 1`, where each element at dimension D contains the cartesian grid coordinates for that element.
///
/// # Arguments
///
/// * `shape` - The shape specifying the dimensions of the tensor.
/// * `device` - The device to create the tensor on.
///
/// # Panics
///
/// Panics if `D2` is not equal to `D+1`.
///
/// # Examples
///
/// ```rust
///    use burn_tensor::Int;
///    use burn_tensor::{backend::Backend, Shape, Tensor};
///    fn example<B: Backend>() {
///        let device = Default::default();
///        let result: Tensor<B, 3, _> = Tensor::<B, 2, Int>::cartesian_grid([2, 3], &device);
///        println!("{}", result);
///    }
/// ```
pub fn cartesian_grid<B: Backend, S: Into<Shape>, const D: usize, const D2: usize>(
    shape: S,
    device: &B::Device,
) -> Tensor<B, D2, Int> {
    if D2 != D + 1 {
        panic!("D2 must equal D + 1 for Tensor::cartesian_grid")
    }

    let dims = shape.into().dims;
    let mut indices: Vec<Tensor<B, D, Int>> = Vec::new();

    for dim in 0..D {
        let dim_range: Tensor<B, 1, Int> = Tensor::arange(0..dims[dim] as i64, device);

        let mut shape = [1; D];
        shape[dim] = dims[dim];
        let mut dim_range = dim_range.reshape(shape);

        for (i, &item) in dims.iter().enumerate() {
            if i == dim {
                continue;
            }
            dim_range = dim_range.repeat_dim(i, item);
        }

        indices.push(dim_range);
    }

    Tensor::stack::<D2>(indices, D)
}
