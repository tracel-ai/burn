use crate::{
    CubeRuntime,
    kernel::{
        into_contiguous_aligned,
        pool::pool2d::{Position, view4d},
        utils::{address_type, decompose_linear, shape_divmod},
    },
    ops::{
        max_vector_size, numeric::empty_device_dtype, permute_nchw_to_nhwc, permute_nhwc_to_nchw,
    },
    tensor::CubeTensor,
};
use burn_backend::Shape;
use cubecl::{
    calculate_cube_count_elemwise,
    num_traits::Zero,
    prelude::*,
    std::{FastDivmod, tensor::View},
};

#[cube(launch, address_type = "dynamic")]
fn adaptive_avg_pool2d_backward_direct<E: Numeric, N: Size>(
    grad: &Tensor<Vector<E, N>>,
    output: &mut View<Vector<E, N>, Position, ReadWrite>,
    out_shape: Sequence<FastDivmod<usize>>,
    working_units: usize,
    #[define(E)] _dtype: StorageType,
) {
    if ABSOLUTE_POS >= working_units {
        terminate!();
    }

    let (_, out_h, out_w, _) = output.shape();
    let (grad_stride_h, grad_stride_w) = (grad.stride(1), grad.stride(2));
    let (grad_h, grad_w) = (grad.shape(1), grad.shape(2));

    let (_, pos) = decompose_linear(ABSOLUTE_POS * output.vector_size(), &out_shape);
    let [b, ih, iw, c] = *pos else { unreachable!() };

    let oh_start = start_index(ih, out_h, grad_h);
    let oh_end = end_index(ih, out_h, grad_h);

    let ow_start = start_index(iw, out_w, grad_w);
    let ow_end = end_index(iw, out_w, grad_w);

    let mut grad_acc = Vector::zero();

    let index_base = b * grad.stride(0) + (c * grad.stride(3));

    for oh in oh_start..oh_end {
        let ih_start = start_index(oh, grad_h, out_h);
        let ih_end = end_index(oh, grad_h, out_h);

        if ih >= ih_start && ih < ih_end {
            for ow in ow_start..ow_end {
                let iw_start = start_index(ow, grad_w, out_w);
                let iw_end = end_index(ow, grad_w, out_w);

                if iw >= iw_start && iw < iw_end {
                    let num_ih = ih_end - ih_start;
                    let num_iw = iw_end - iw_start;

                    let index = index_base + (oh * grad_stride_h) + (ow * grad_stride_w);
                    grad_acc +=
                        grad[index / grad.vector_size()] / Vector::cast_from(num_iw * num_ih);
                }
            }
        }
    }

    output[(b, ih, iw, c)] = grad_acc;
}

#[cube]
fn start_index(output_size_index: usize, output_size: usize, input_size: usize) -> usize {
    (output_size_index * input_size) / output_size
}

#[cube]
fn end_index(output_size_index: usize, output_size: usize, input_size: usize) -> usize {
    let index = (output_size_index + 1) * input_size;
    let index = index.div_ceil(output_size);

    if input_size < index {
        input_size
    } else {
        index
    }
}

pub(crate) fn adaptive_avg_pool2d_backward<R: CubeRuntime>(
    x: CubeTensor<R>,
    out_grad: CubeTensor<R>,
) -> CubeTensor<R> {
    let [batches, channels, height, width] = x.meta.shape().dims();

    let out_grad = into_contiguous_aligned(permute_nchw_to_nhwc(out_grad));
    let vector_size = max_vector_size(&out_grad);

    let out_shape = Shape::new([batches, height, width, channels]);
    let output = empty_device_dtype(x.client.clone(), x.device.clone(), out_shape, x.dtype);

    let num_elems = output.meta.num_elements();

    let working_units = num_elems / vector_size as usize;
    let cube_dim = CubeDim::new(&x.client, working_units);
    let cube_count = calculate_cube_count_elemwise(&x.client, working_units, cube_dim);

    adaptive_avg_pool2d_backward_direct::launch(
        &output.client,
        cube_count,
        cube_dim,
        address_type!(out_grad, output),
        vector_size,
        out_grad.into_tensor_arg(),
        view4d(output.clone(), vector_size),
        shape_divmod(&output),
        working_units,
        output.dtype.into(),
    );

    permute_nhwc_to_nchw(output)
}
