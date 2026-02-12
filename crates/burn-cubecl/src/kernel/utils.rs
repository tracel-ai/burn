use burn_backend::Shape;
use cubecl::{
    ir::LineSize,
    prelude::*,
    std::{
        FastDivmod, FastDivmodArgs, FastDivmodInt,
        tensor::layout::linear::{LinearLayoutArgs, LinearViewLaunch},
    },
};
use cubecl::{prelude::SequenceArg, std::tensor::layout::linear::LinearLayout};

use crate::{CubeRuntime, tensor::CubeTensor};

pub fn shape_divmod<'a, R: CubeRuntime>(
    tensor: &CubeTensor<R>,
) -> SequenceArg<'a, R, FastDivmod<usize>> {
    let mut arg = SequenceArg::new();
    for dim in tensor.shape.iter() {
        arg.push(FastDivmodArgs::<usize>::new(&tensor.client, *dim));
    }
    arg
}

pub fn linear_layout<'a, R: CubeRuntime>(
    tensor: &'a CubeTensor<R>,
    line_size: LineSize,
) -> LinearLayoutArgs<'a, R> {
    LinearLayoutArgs::from_shape_strides(&tensor.client, &tensor.shape, &tensor.strides, line_size)
}

pub fn linear_layout_ref<'a, R: CubeRuntime>(
    tensor: &'a CubeTensor<R>,
    reference: &'a CubeTensor<R>,
    line_size: LineSize,
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
    line_size: LineSize,
) -> LinearViewLaunch<'a, R> {
    let len = tensor.shape.iter().product::<usize>();
    let layout = linear_layout(tensor, line_size);
    let buffer = unsafe {
        ArrayArg::from_raw_parts_and_size(&tensor.handle, len, line_size, tensor.elem_size())
    };
    LinearViewLaunch::new::<LinearLayout>(buffer, layout)
}

pub fn linear_view_ref<'a, R: CubeRuntime>(
    tensor: &'a CubeTensor<R>,
    reference: &'a CubeTensor<R>,
    line_size: LineSize,
) -> LinearViewLaunch<'a, R> {
    let len = tensor.shape.iter().product::<usize>();
    let layout = linear_layout_ref(tensor, reference, line_size);
    let buffer = unsafe {
        ArrayArg::from_raw_parts_and_size(&tensor.handle, len, line_size, tensor.elem_size())
    };
    LinearViewLaunch::new::<LinearLayout>(buffer, layout)
}

pub fn linear_view_alias<'a, R: CubeRuntime>(
    tensor: &'a CubeTensor<R>,
    line_size: LineSize,
    pos: usize,
) -> LinearViewLaunch<'a, R> {
    let layout = linear_layout(tensor, line_size);
    let buffer = ArrayArg::Alias { input_pos: pos };
    LinearViewLaunch::new::<LinearLayout>(buffer, layout)
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

pub fn broadcast_strides<'a, R: CubeRuntime>(
    reference: &CubeTensor<R>,
    tensor: &'a CubeTensor<R>,
) -> SequenceArg<'a, R, usize> {
    if reference.shape != tensor.shape {
        tensor
            .strides
            .iter()
            .zip(tensor.shape.dims.iter().zip(&reference.shape.dims))
            .map(|(stride, (shape, ref_shape))| if *shape == *ref_shape { *stride } else { 0 })
            .map(ScalarArg::new)
            .collect()
    } else {
        tensor.strides.iter().copied().map(ScalarArg::new).collect()
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
