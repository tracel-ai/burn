use burn_backend::Shape;
use cubecl::prelude::SequenceArg;
use cubecl::{
    prelude::*,
    std::{FastDivmod, FastDivmodInt},
};

use crate::{CubeRuntime, tensor::CubeTensor};

pub fn shape_divmod<R: CubeRuntime>(tensor: &CubeTensor<R>) -> SequenceArg<R, FastDivmod<usize>> {
    let mut arg = SequenceArg::new();
    for dim in tensor.meta.shape().iter() {
        arg.push(*dim);
    }
    arg
}

pub fn shape_divmod_range<R: CubeRuntime>(
    tensor: &CubeTensor<R>,
    range: core::ops::Range<usize>,
) -> SequenceArg<R, FastDivmod<usize>> {
    let mut arg = SequenceArg::new();
    let shape = &tensor.meta.shape;
    for i in range {
        arg.push(shape[i]);
    }
    arg
}

pub fn split_dim<R: CubeRuntime>(
    mut tensor: CubeTensor<R>,
    dim: usize,
    shape: &[usize],
) -> CubeTensor<R> {
    let mut stride = tensor.meta.strides()[dim];
    tensor.meta.remove(dim);

    for size in shape.iter().rev() {
        tensor.meta.insert(dim, *size, stride);
        stride *= size;
    }

    tensor
}

pub fn broadcast_shape<R: CubeRuntime>(tensors: &[&CubeTensor<R>]) -> Shape {
    let rank = tensors[0].meta.num_dims();
    debug_assert!(
        tensors.iter().all(|it| it.meta.num_dims() == rank),
        "Broadcast tensors must have the same rank"
    );

    let dims = (0..rank).map(|dim| {
        let max = tensors.iter().map(|it| it.meta.shape()[dim]).max();
        let max = max.unwrap_or(1);
        debug_assert!(
            tensors
                .iter()
                .all(|it| it.meta.shape()[dim] == max || it.meta.shape()[dim] == 1),
            "Broadcast dims must be size 1"
        );
        max
    });

    Shape::from(dims)
}

pub fn broadcast_strides<R: CubeRuntime>(
    reference: &CubeTensor<R>,
    tensor: &CubeTensor<R>,
) -> SequenceArg<R, usize> {
    if reference.meta.shape() != tensor.meta.shape() {
        tensor
            .meta
            .strides()
            .iter()
            .zip(
                tensor
                    .meta
                    .shape()
                    .iter()
                    .zip(reference.meta.shape().iter()),
            )
            .map(|(stride, (shape, ref_shape))| if *shape == *ref_shape { *stride } else { 0 })
            .collect()
    } else {
        tensor.meta.strides().iter().copied().collect()
    }
}

#[cube]
pub(crate) fn decompose_linear<I: FastDivmodInt>(
    pos: I,
    shape: &Sequence<FastDivmod<I>>,
) -> (I, Sequence<I>) {
    let rank = comptime![shape.len()];
    let mut offs = pos;
    let mut out = Sequence::new();

    #[unroll]
    for i in 0..rank {
        let dim = comptime![rank - i - 1];
        let (rem, offs_local) = shape.index(dim).div_mod(offs);
        out.push(offs_local);
        offs = rem;
    }

    (offs, out.rev())
}

pub(crate) trait RequiredAddrType {
    fn required_address_type(&self) -> AddressType;
}

impl<R: CubeRuntime> RequiredAddrType for CubeTensor<R> {
    fn required_address_type(&self) -> AddressType {
        self.required_address_type()
    }
}
impl<R: CubeRuntime> RequiredAddrType for Option<CubeTensor<R>> {
    fn required_address_type(&self) -> AddressType {
        self.as_ref()
            .map(|it| it.required_address_type())
            .unwrap_or_default()
    }
}

macro_rules! address_type {
    ($($tensor: tt),*) => {
        [$($crate::kernel::utils::RequiredAddrType::required_address_type(&$tensor)),*]
        .into_iter()
        .max()
        .unwrap_or_default()
    };
}
pub(crate) use address_type;
