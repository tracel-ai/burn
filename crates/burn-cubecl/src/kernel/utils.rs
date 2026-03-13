use burn_backend::Shape;
use cubecl::{
    ir::VectorSize,
    prelude::*,
    std::{
        FastDivmod, FastDivmodArgs, FastDivmodInt,
        tensor::layout::linear::{LinearLayoutArgs, LinearViewLaunch},
    },
};
use cubecl::{prelude::SequenceArg, std::tensor::layout::linear::LinearLayout};

use crate::{CubeRuntime, tensor::CubeTensor};

pub fn shape_divmod<R: CubeRuntime>(tensor: &CubeTensor<R>) -> SequenceArg<R, FastDivmod<usize>> {
    let mut arg = SequenceArg::new();
    for dim in tensor.meta.shape().iter() {
        arg.push(FastDivmodArgs::<usize>::new(&tensor.client, *dim));
    }
    arg
}

pub fn linear_layout<R: CubeRuntime>(
    tensor: &CubeTensor<R>,
    vector_size: VectorSize,
) -> LinearLayoutArgs<R> {
    LinearLayoutArgs::from_shape_strides(
        &tensor.client,
        tensor.meta.shape(),
        tensor.meta.strides(),
        vector_size,
    )
}

pub fn linear_layout_ref<R: CubeRuntime>(
    tensor: &CubeTensor<R>,
    reference: &CubeTensor<R>,
    vector_size: VectorSize,
) -> LinearLayoutArgs<R> {
    LinearLayoutArgs::from_shape_strides_with_reference(
        &tensor.client,
        tensor.meta.shape(),
        reference.meta.shape(),
        tensor.meta.strides(),
        vector_size,
    )
}

pub fn linear_view<R: CubeRuntime>(
    tensor: CubeTensor<R>,
    vector_size: VectorSize,
) -> LinearViewLaunch<R> {
    let len = tensor.meta.num_elements();
    let layout = linear_layout(&tensor, vector_size);
    let buffer = unsafe { ArrayArg::from_raw_parts(tensor.handle, len) };
    LinearViewLaunch::new::<LinearLayout>(buffer, layout)
}

pub fn linear_view_ref<R: CubeRuntime>(
    tensor: CubeTensor<R>,
    reference: &CubeTensor<R>,
    vector_size: VectorSize,
) -> LinearViewLaunch<R> {
    let len = tensor.meta.num_elements();
    let layout = linear_layout_ref(&tensor, reference, vector_size);
    let buffer = unsafe { ArrayArg::from_raw_parts(tensor.handle, len) };
    LinearViewLaunch::new::<LinearLayout>(buffer, layout)
}

pub fn linear_view_alias<R: CubeRuntime>(
    tensor: &CubeTensor<R>,
    vector_size: VectorSize,
    pos: usize,
) -> LinearViewLaunch<R> {
    let layout = linear_layout(tensor, vector_size);
    let buffer = ArrayArg::Alias { input_pos: pos };
    LinearViewLaunch::new::<LinearLayout>(buffer, layout)
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
