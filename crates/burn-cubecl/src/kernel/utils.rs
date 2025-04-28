use cubecl::{linalg::tensor::StridedLayoutArgs, prelude::SequenceArg};
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
