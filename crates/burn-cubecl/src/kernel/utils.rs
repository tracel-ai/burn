use cubecl::{prelude::SequenceArg, std::tensor::StridedLayoutArgs};
use cubecl_std::{FastDivmod, FastDivmodArgs};

use crate::{CubeRuntime, tensor::CubeTensor};

pub fn shape_divmod<'a, R: CubeRuntime>(tensor: &CubeTensor<R>) -> SequenceArg<'a, R, FastDivmod> {
    let mut arg = SequenceArg::new();
    for dim in tensor.shape.dims.iter() {
        arg.push(FastDivmodArgs::new(&tensor.client, *dim as u32));
    }
    arg
}

pub fn strided_layout<'a, R: CubeRuntime>(tensor: &CubeTensor<R>) -> StridedLayoutArgs<'a, R> {
    let rank = tensor.shape.num_dims();
    if rank <= 1 || tensor.shape.dims[rank - 1] == tensor.strides[rank - 2] {
        StridedLayoutArgs::none()
    } else {
        StridedLayoutArgs::strided(&tensor.client, tensor.shape.dims[rank - 1] as u32)
    }
}

pub fn split_dim<R: CubeRuntime>(
    mut tensor: CubeTensor<R>,
    dim: usize,
    shape: &[usize],
) -> CubeTensor<R> {
    let mut stride = tensor.strides[dim];
    tensor.shape.dims.remove(dim);
    tensor.strides.remove(dim);

    for size in shape.iter().rev() {
        tensor.shape.dims.insert(dim, *size);
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
    tensor.shape.dims[dim1] *= tensor.shape.dims[dim0];
    tensor.shape.dims.remove(dim0);
    tensor.strides.remove(dim0);
    tensor
}
