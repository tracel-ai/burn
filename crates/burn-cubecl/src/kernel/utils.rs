use alloc::borrow::Cow;
use burn_backend::{Shape, TensorMetadata};
use cubecl::prelude::SequenceArg;
use cubecl::{
    prelude::*,
    std::{FastDivmod, FastDivmodInt},
};

use crate::tensor::CubeTensor;

pub fn shape_divmod(tensor: &CubeTensor) -> SequenceArg<FastDivmod<usize>> {
    let mut arg = SequenceArg::new();
    for dim in tensor.meta.shape().iter() {
        arg.push(*dim);
    }
    arg
}

pub fn shape_divmod_range(
    tensor: &CubeTensor,
    range: core::ops::Range<usize>,
) -> SequenceArg<FastDivmod<usize>> {
    let mut arg = SequenceArg::new();
    let shape = &tensor.meta.shape;
    for i in range {
        arg.push(shape[i]);
    }
    arg
}

pub fn split_dim(mut tensor: CubeTensor, dim: usize, shape: &[usize]) -> CubeTensor {
    let mut stride = tensor.meta.strides()[dim];
    tensor.meta.remove(dim);

    for size in shape.iter().rev() {
        tensor.meta.insert(dim, *size, stride);
        stride *= size;
    }

    tensor
}

/// The shape the logical shapes of `tensors` broadcast to.
pub fn broadcast_shape<const N: usize>(tensors: &[&CubeTensor; N]) -> Shape {
    // Rows are read in place; only a storage-tiled tensor builds its logical shape.
    let shapes = tensors
        .each_ref()
        .map(|tensor| match tensor.meta.is_tiled() {
            false => Cow::Borrowed(tensor.meta.shape()),
            true => Cow::Owned(tensor.shape()),
        });
    let rank = shapes[0].num_dims();
    debug_assert!(
        shapes.iter().all(|shape| shape.num_dims() == rank),
        "Broadcast tensors must have the same rank"
    );

    let dims = (0..rank).map(|dim| {
        // Broadcasting stretches a size-1 dim to the other operand's size, which may be 0.
        // Every size that is not 1 must agree, and that shared size (0 included) is the result.
        // Taking the max would be wrong when a 0 meets a 1: max picks 1 and drops the 0.
        let mut broadcast = 1;
        for shape in &shapes {
            let size = shape[dim];
            if size != 1 {
                debug_assert!(
                    broadcast == 1 || broadcast == size,
                    "Broadcast dims must match or be 1"
                );
                broadcast = size;
            }
        }
        broadcast
    });

    Shape::from(dims)
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

    (offs, out.reversed())
}

pub(crate) trait RequiredAddrType {
    fn required_address_type(&self) -> AddressType;
}

impl RequiredAddrType for CubeTensor {
    fn required_address_type(&self) -> AddressType {
        self.required_address_type()
    }
}
impl RequiredAddrType for Option<CubeTensor> {
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

/// The threads it takes to keep every core of `tensor`'s device busy: one per CPU core, or a full
/// cube on each multiprocessor of a GPU (cubecl's estimate where the runtime reports no count).
pub(crate) fn device_threads(tensor: &CubeTensor) -> usize {
    let hardware = &tensor.client.properties().hardware;
    match hardware.num_cpu_cores {
        Some(cores) => cores as usize,
        None => {
            let multiprocessors = hardware
                .num_streaming_multiprocessors
                .unwrap_or(cubecl::std::tensor::NUM_SM_APPROX);
            multiprocessors as usize * hardware.max_units_per_cube as usize
        }
    }
}

/// Whether `outputs` values, each reduced over `length` terms, are too few to fill a device that runs
/// `threads` at once with a thread each while their terms are not: a thread per output would leave
/// cores idle.
pub(crate) fn few_outputs_over_a_long_axis(threads: usize, outputs: usize, length: usize) -> bool {
    outputs < threads && threads <= outputs.saturating_mul(length)
}
