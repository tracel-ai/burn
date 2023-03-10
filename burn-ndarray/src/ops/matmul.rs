use crate::{element::FloatNdArrayElement, tensor::NdArrayTensor, NdArrayBackend};
use crate::{iter_par, run_par, UnsafeSharedRef};
use burn_tensor::ElementConversion;
use burn_tensor::{ops::TensorOps, Shape};
use ndarray::s;

pub(crate) fn matmul<E, const D: usize>(
    lhs: NdArrayTensor<E, D>,
    rhs: NdArrayTensor<E, D>,
) -> NdArrayTensor<E, D>
where
    E: FloatNdArrayElement,
{
    let shape_ori_lhs = lhs.shape();
    let shape_ori_rhs = rhs.shape();

    let lhs = reshape(lhs);
    let rhs = reshape(rhs);

    let [batch_size_lhs, m, _] = lhs.shape().dims;
    let [batch_size_rhs, _, n] = rhs.shape().dims;

    let mut shape_out = match batch_size_lhs > batch_size_rhs {
        true => shape_ori_lhs,
        false => shape_ori_rhs,
    };
    shape_out.dims[D - 2] = m;
    shape_out.dims[D - 1] = n;

    let out = general_matmul(lhs, rhs);

    NdArrayBackend::<E>::reshape(out, shape_out)
}

fn general_matmul<E: FloatNdArrayElement>(
    lhs: NdArrayTensor<E, 3>,
    rhs: NdArrayTensor<E, 3>,
) -> NdArrayTensor<E, 3> {
    run_par!(|| {
        let [batch_size_lhs, m, _] = lhs.shape().dims;
        let [batch_size_rhs, k, n] = rhs.shape().dims;
        let batch_size = usize::max(batch_size_rhs, batch_size_lhs);

        if batch_size_lhs > batch_size && batch_size_lhs != 1 {
            panic!("Broadcast on multiple dimensions is not yet supported");
        }

        if batch_size_rhs > batch_size && batch_size_rhs != 1 {
            panic!("Broadcast on multiple dimensions is not yet supported");
        }

        let alpha: E = 1.0.elem();
        let beta: E = 0.0.elem();

        let mut out_array = ndarray::Array3::<E>::zeros((batch_size, m, n));
        let unsafe_shared_out_array = UnsafeSharedRef::new(&mut out_array);

        let lhs_array = lhs.array.into_shape((batch_size_lhs, m, k)).unwrap();
        let rhs_array = rhs.array.into_shape((batch_size_rhs, k, n)).unwrap();

        iter_par!(0, batch_size).for_each(|b| {
            let lhs_slice = match batch_size_lhs == 1 {
                true => lhs_array.slice(s!(0, .., ..)),
                false => lhs_array.slice(s!(b, .., ..)),
            };
            let rhs_slice = match batch_size_rhs == 1 {
                true => rhs_array.slice(s!(0, .., ..)),
                false => rhs_array.slice(s!(b, .., ..)),
            };

            unsafe {
                let mut out_slice = unsafe_shared_out_array.get().slice_mut(s!(b, .., ..));

                ndarray::linalg::general_mat_mul(
                    alpha,
                    &lhs_slice,
                    &rhs_slice,
                    beta,
                    &mut out_slice,
                );
            }
        });

        NdArrayTensor::new(out_array.into_shared().into_dyn())
    })
}

fn reshape<E: FloatNdArrayElement, const D: usize>(
    tensor: NdArrayTensor<E, D>,
) -> NdArrayTensor<E, 3> {
    let shape = tensor.shape();

    if D < 2 {
        NdArrayBackend::<E>::reshape(tensor, Shape::new([1, 1, shape.dims[0]]))
    } else {
        let batch_size = batch_size(&shape);
        let size0 = shape.dims[D - 2];
        let size1 = shape.dims[D - 1];

        NdArrayBackend::<E>::reshape(tensor, Shape::new([batch_size, size0, size1]))
    }
}

fn batch_size<const D: usize>(shape: &Shape<D>) -> usize {
    let mut num_batch = 1;
    for i in 0..D - 2 {
        num_batch *= shape.dims[i];
    }

    num_batch
}
