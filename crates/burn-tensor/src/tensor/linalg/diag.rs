use crate::backend::Backend;
use crate::tensor::{Tensor, Shape, Int};
use crate::{TensorKind, BasicOps};

/// Returns the diag of the of a matrix.
///
/// For batched inputs, returns of each matrix in the batch independently.
///
/// The diag operation extracts the diagonal elements of the last two dimensions,
/// treating them as the matrix dimensions, while preserving all leading batch dimensions.
///
/// # Arguments
///
/// * `tensor` - The input tensor with at least 2 dimensions.
///
/// # Returns
///
/// Tensor with same shape as the input, where the D - 2 dimension contains the diagonal for that batch
/// and the last dimension is 1 to preserve rank
///
pub fn diag<B: Backend, const D: usize, K>(tensor: Tensor<B, D, K>) -> Tensor<B, D, K>
    where K: TensorKind<B> + BasicOps<B>{
        let shape = tensor.shape();
        let rows = shape.dims[D - 2];
        let cols = shape.dims[D - 1];
        let diag_len = rows.min(cols);
        let device = tensor.device();

        // create the indices for the dia

        let mut flat_shape = shape.dims.to_vec();
        flat_shape[D - 2] = rows * cols;
        flat_shape[D - 1] = 1;
        let flat: Tensor<B, D, K> = tensor.reshape(Shape::from(flat_shape));

        let step = cols + 1;
        let diag_indices: Vec<i32> = (0..diag_len)
            .map(|i| (i * step) as i32)
            .collect();

        let indices = Tensor::<B, 1, Int>::from_data(diag_indices.as_slice(), &device);
        //let diag = flat.take::<1, D>(D - 2, indices);
        flat.take::<1, D>(D - 2, indices)

        // let mut result_shape = shape.dims.to_vec();
        // result_shape[D - 2] = diag_len;
        // result_shape[D - 1] = 1;
        // diag.reshape(Shape::from(result_shape))

 }

// pub fn diag<B: Backend, const D: usize, K>(tensor: Tensor<B, D, K>) -> Tensor<B, D, K>
//     where K: Numeric<B> {
//         let shape = tensor.shape();

//         let mat_shape = [shape.dims[D - 2], shape.dims[D - 1]];
//         let mask = Tensor::<B, 2, Bool>::diag_mask(mat_shape, 0, &tensor.device());

//         let mut mask_shape = vec![1; D];
//         mask_shape[D - 2] = mat_shape[0];
//         mask_shape[D - 1] = mat_shape[1];
//         let mask_shape = Shape::from(mask_shape);
//         let mask = mask.reshape(mask_shape);

//         tensor.mask_select(mask, 0)

//     }