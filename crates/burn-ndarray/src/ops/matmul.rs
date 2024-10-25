use crate::{element::FloatNdArrayElement, tensor::NdArrayTensor, NdArray, UnsafeSharedRef};

use alloc::{vec, vec::Vec};
use burn_common::{iter_range_par, run_par};
use burn_tensor::ElementConversion;
use burn_tensor::{ops::FloatTensorOps, Shape};
use ndarray::s;

pub(crate) fn matmul<E>(lhs: NdArrayTensor<E>, rhs: NdArrayTensor<E>) -> NdArrayTensor<E>
where
    E: FloatNdArrayElement,
{
    let shape_lhs = lhs.shape();
    let shape_rhs = rhs.shape();
    let ndims = shape_lhs.num_dims();
    let m = shape_lhs.dims[ndims - 2]; // # of left rows
    let k = shape_rhs.dims[ndims - 2]; // # of left cols and right rows
    let n = shape_rhs.dims[ndims - 1]; // # of right cols

    let (out_shape, strides_lhs, strides_rhs, strides_out) = output_shape(&shape_lhs, &shape_rhs);
    let l_mat_size = m * k; // size of matrix component of left array
    let r_mat_size = k * n; // size of matrix component of right array
    let out_mat_size = m * n; // size of matrix component of output array

    let num_l_batches = shape_lhs.num_elements() / l_mat_size;
    let num_r_batches = shape_rhs.num_elements() / r_mat_size;
    let num_out_batches = out_shape.num_elements() / out_mat_size;

    let alpha: E = 1.0.elem();
    let beta: E = 0.0.elem();

    let out: NdArrayTensor<E> = run_par!(|| {
        let mut out_array = ndarray::Array3::<E>::zeros((num_out_batches, m, n));
        let unsafe_shared_out_array = UnsafeSharedRef::new(&mut out_array);

        let lhs_array = NdArray::<E>::float_reshape(lhs, Shape::new([num_l_batches, m, k])).array;
        let rhs_array = NdArray::<E>::float_reshape(rhs, Shape::new([num_r_batches, k, n])).array;

        iter_range_par!(0, num_out_batches).for_each(|out_batch| {
            // Here, we:
            //   1. Un-flatten the output batch into a component-based batch index.
            //   2. Use the strides for left and right batch indices to convert it to a flattened
            //      batch for left and right.
            let out_index = strides_out.unflatten(out_batch);
            let l_batch = strides_lhs.flatten(&out_index);
            let r_batch = strides_rhs.flatten(&out_index);

            let lhs_slice = lhs_array.slice(s!(l_batch, .., ..));
            let rhs_slice = rhs_array.slice(s!(r_batch, .., ..));

            unsafe {
                let mut out_slice = unsafe_shared_out_array
                    .get()
                    .slice_mut(s!(out_batch, .., ..));

                ndarray::linalg::general_mat_mul(
                    alpha,
                    &lhs_slice,
                    &rhs_slice,
                    beta,
                    &mut out_slice,
                )
            }
        });

        NdArrayTensor::new(out_array.into_shared().into_dyn())
    });

    NdArray::<E>::float_reshape(out, out_shape)
}

#[derive(Debug, PartialEq)]
struct Strides {
    strides: Vec<usize>,
}
impl Strides {
    fn new(strides: Vec<usize>) -> Self {
        Strides { strides }
    }

    fn unflatten(&self, linear_index: usize) -> Vec<usize> {
        let mut coord = Vec::with_capacity(self.strides.len());
        let mut rem = linear_index;
        for stride in self.strides.iter() {
            coord.push(rem / stride);
            rem %= stride;
        }
        coord
    }

    fn flatten(&self, index: &Vec<usize>) -> usize {
        assert_eq!(self.strides.len(), index.len());
        self.strides
            .iter()
            .zip(index)
            .map(|(stride, index)| stride * index)
            .sum()
    }
}

/// Compute the (broadcasted) output shape of matrix multiplication, along with strides for
/// the non-matrix dimensions of all arrays.
///
/// # Arguments
/// * `lsh`: Shape of the first (left-hand) matrix multiplication argument.
/// * `rsh`: Shape of the second (right-hand) matrix multiplication argument.
///
/// # Panics
/// * If `D` is not at least 2.
/// * If the matrix multiplication dimensions (last 2) are incompatible.
/// * If any other dimension is not the same for both tensors, or equal to 1. (Any dimension where
///   one dim is equal to 1 is broadcast.)
fn output_shape(lsh: &Shape, rsh: &Shape) -> (Shape, Strides, Strides, Strides) {
    let ndims = lsh.num_dims();
    if ndims < 2 {
        panic!("Matrix multiplication requires an array with at least 2 dimensions.");
    }

    // Fetch matrix dimensions and check compatibility.
    let l_rows = lsh.dims[ndims - 2];
    let l_cols = lsh.dims[ndims - 1];
    let r_rows = rsh.dims[ndims - 2];
    let r_cols = rsh.dims[ndims - 1];
    if l_cols != r_rows {
        panic!("Dimensions are incompatible for matrix multiplication.");
    }
    // Set matrix dimensions of the output shape.
    let mut osh = vec![0; ndims];
    osh[ndims - 2] = l_rows;
    osh[ndims - 1] = r_cols;

    // Set other array dimensions, broadcasting as necessary.
    // Compute the strides inline.
    let mut cur_l_stride: usize = 1;
    let mut cur_r_stride: usize = 1;
    let mut cur_o_stride: usize = 1;
    let mut l_strides = Vec::with_capacity(ndims - 2);
    let mut r_strides = Vec::with_capacity(ndims - 2);
    let mut o_strides = Vec::with_capacity(ndims - 2);
    for i in (0..ndims - 2).rev() {
        let l_dim = lsh.dims[i];
        let r_dim = rsh.dims[i];

        // Compatible dimensions are:
        //   1. Both dimensions are equal.
        //   2. One of the dimensions is equal to 1.
        let o_dim: usize;
        if l_dim == r_dim {
            o_dim = l_dim; // both dimensions are equal
            l_strides.push(cur_l_stride);
            r_strides.push(cur_r_stride);
        } else if l_dim == 1 {
            o_dim = r_dim; // broadcast the left
            l_strides.push(0);
            r_strides.push(cur_r_stride);
        } else if r_dim == 1 {
            o_dim = l_dim; // broadcast the right
            l_strides.push(cur_l_stride);
            r_strides.push(0);
        } else {
            panic!("Dimensions differ and cannot be broadcasted.");
        }
        osh[i] = o_dim;
        o_strides.push(cur_o_stride);
        cur_o_stride *= o_dim;

        cur_l_stride *= l_dim;
        cur_r_stride *= r_dim;
    }
    l_strides.reverse();
    r_strides.reverse();
    o_strides.reverse();

    (
        Shape::from(osh),
        Strides::new(l_strides),
        Strides::new(r_strides),
        Strides::new(o_strides),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    impl Strides {
        fn empty() -> Self {
            Strides {
                strides: Vec::with_capacity(0),
            }
        }
    }

    #[test]
    fn test_output_shape() {
        // plain matrix multiply
        assert_eq!(
            output_shape(&Shape::from([5, 3]), &Shape::from([3, 7])),
            (
                Shape::from([5, 7]),
                Strides::empty(),
                Strides::empty(),
                Strides::empty()
            )
        );
        // matrix multiply with one extra stack dimension
        assert_eq!(
            output_shape(&Shape::from([4, 5, 3]), &Shape::from([4, 3, 7])),
            (
                Shape::from([4, 5, 7]),
                Strides::new(vec![1]),
                Strides::new(vec![1]),
                Strides::new(vec![1])
            )
        );
        // rank 3, broadcast left
        assert_eq!(
            output_shape(&Shape::from([1, 5, 3]), &Shape::from([4, 3, 7])),
            (
                Shape::from([4, 5, 7]),
                Strides::new(vec![0]),
                Strides::new(vec![1]),
                Strides::new(vec![1])
            )
        );
        // rank 3, broadcast right
        assert_eq!(
            output_shape(&Shape::from([4, 5, 3]), &Shape::from([1, 3, 7])),
            (
                Shape::from([4, 5, 7]),
                Strides::new(vec![1]),
                Strides::new(vec![0]),
                Strides::new(vec![1])
            )
        );
        // rank 4, multi broadcast
        assert_eq!(
            output_shape(&Shape::from([1, 4, 5, 3]), &Shape::from([8, 1, 3, 7])),
            (
                Shape::from([8, 4, 5, 7]),
                Strides::new(vec![0, 1]),
                Strides::new(vec![1, 0]),
                Strides::new(vec![4, 1])
            )
        );
        // rank 5, multi-broadcast
        assert_eq!(
            output_shape(&Shape::from([1, 3, 4, 5, 3]), &Shape::from([8, 3, 1, 3, 7])),
            (
                Shape::from([8, 3, 4, 5, 7]),
                Strides::new(vec![0, 4, 1]),
                Strides::new(vec![3, 1, 0]),
                Strides::new(vec![12, 4, 1])
            )
        )
    }

    #[test]
    #[should_panic(
        expected = "Matrix multiplication requires an array with at least 2 dimensions."
    )]
    fn test_output_shape_too_small() {
        output_shape(&Shape::from([4]), &Shape::from([4]));
    }

    #[test]
    #[should_panic(expected = "Dimensions are incompatible for matrix multiplication.")]
    fn test_output_shape_bad_matrix_dims() {
        output_shape(&Shape::from([5, 3]), &Shape::from([4, 7]));
    }

    #[test]
    #[should_panic(expected = "Dimensions differ and cannot be broadcasted.")]
    fn test_output_shape_non_broadcast() {
        output_shape(&Shape::from([4, 5, 3]), &Shape::from([2, 3, 7]));
    }
}
