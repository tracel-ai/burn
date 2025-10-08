use burn_tensor::Shape;
use cubecl::prelude::SequenceArg;
use cubecl::{
    prelude::ArrayArg,
    std::{
        FastDivmod, FastDivmodArgs,
        tensor::layout::linear::{LinearLayoutArgs, LinearViewLaunch},
    },
};

use crate::{CubeRuntime, tensor::CubeTensor};

pub fn shape_divmod<'a, R: CubeRuntime>(tensor: &CubeTensor<R>) -> SequenceArg<'a, R, FastDivmod> {
    let mut arg = SequenceArg::new();
    for dim in tensor.shape.iter() {
        arg.push(FastDivmodArgs::new(&tensor.client, *dim as u32));
    }
    arg
}

pub fn linear_layout<'a, R: CubeRuntime>(
    tensor: &'a CubeTensor<R>,
    line_size: u8,
) -> LinearLayoutArgs<'a, R> {
    LinearLayoutArgs::from_shape_strides(&tensor.client, &tensor.shape, &tensor.strides, line_size)
}

pub fn linear_layout_ref<'a, R: CubeRuntime>(
    tensor: &'a CubeTensor<R>,
    reference: &'a CubeTensor<R>,
    line_size: u8,
) -> LinearLayoutArgs<'a, R> {
    LinearLayoutArgs::from_shape_strides_with_reference(
        &tensor.client,
        &tensor.shape,
        &reference.shape,
        &tensor.strides,
        line_size,
    )
}

pub fn linear_view<'a, R: CubeRuntime>(
    tensor: &'a CubeTensor<R>,
    line_size: u8,
) -> LinearViewLaunch<'a, R> {
    let len = tensor.shape.iter().product::<usize>();
    let layout = linear_layout(tensor, line_size);
    let buffer = unsafe {
        ArrayArg::from_raw_parts_and_size(&tensor.handle, len, line_size, tensor.elem_size())
    };
    LinearViewLaunch::new(buffer, layout)
}

pub fn linear_view_ref<'a, R: CubeRuntime>(
    tensor: &'a CubeTensor<R>,
    reference: &'a CubeTensor<R>,
    line_size: u8,
) -> LinearViewLaunch<'a, R> {
    let len = tensor.shape.iter().product::<usize>();
    let layout = linear_layout_ref(tensor, reference, line_size);
    let buffer = unsafe {
        ArrayArg::from_raw_parts_and_size(&tensor.handle, len, line_size, tensor.elem_size())
    };
    LinearViewLaunch::new(buffer, layout)
}

pub fn linear_view_alias<'a, R: CubeRuntime>(
    tensor: &'a CubeTensor<R>,
    line_size: u8,
    pos: usize,
) -> LinearViewLaunch<'a, R> {
    let layout = linear_layout(tensor, line_size);
    let buffer = ArrayArg::Alias { input_pos: pos };
    LinearViewLaunch::new(buffer, layout)
}

pub fn split_dim<R: CubeRuntime>(
    mut tensor: CubeTensor<R>,
    dim: usize,
    shape: &[usize],
) -> CubeTensor<R> {
    let mut stride = tensor.strides[dim];
    tensor.shape.remove(dim);
    tensor.strides.remove(dim);

    for size in shape.iter().rev() {
        tensor.shape.insert(dim, *size);
        tensor.strides.insert(dim, stride);
        stride *= size;
    }

    tensor
}

pub fn merge_dims<R: CubeRuntime>(
    mut tensor: CubeTensor<R>,
    dim0: usize,
    dim1: usize,
) -> CubeTensor<R> {
    tensor.shape[dim1] *= tensor.shape[dim0];
    tensor.shape.remove(dim0);
    tensor.strides.remove(dim0);
    tensor
}

pub fn broadcast_shape<R: CubeRuntime>(tensors: &[&CubeTensor<R>]) -> Shape {
    let rank = tensors[0].shape.num_dims();
    debug_assert!(
        tensors.iter().all(|it| it.shape.num_dims() == rank),
        "Broadcast tensors must have the same rank"
    );

    let dims = (0..rank).map(|dim| {
        let max = tensors.iter().map(|it| it.shape[dim]).max();
        let max = max.unwrap_or(1);
        debug_assert!(
            tensors
                .iter()
                .all(|it| it.shape[dim] == max || it.shape[dim] == 1),
            "Broadcast dims must be size 1"
        );
        max
    });

    Shape {
        dims: dims.collect(),
    }
}
