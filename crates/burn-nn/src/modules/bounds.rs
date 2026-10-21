//! The floors several modules share the shape of, priced as their
//! [`Bounds`](burn::instrument::Bounds) document: what each must move and compute to run alone.

use burn_core as burn;

use burn::instrument::{self, Bounds};
use burn::module::Param;
use burn::tensor::Tensor;
use burn::tensor::ops::{AvgPoolOptions, ConvOptions, MaxPoolOptions};

use alloc::vec;

/// The spatial size of `input`: its last `N` dimensions.
pub(crate) fn spatial<const D: usize, const N: usize>(input: &Tensor<D>) -> [usize; N] {
    let dims = input.dims();
    core::array::from_fn(|i| dims[D - N + i])
}

/// The bytes of a parameter a module may not have.
pub(crate) fn optional_parameter<const D: usize>(parameter: &Option<Param<Tensor<D>>>) -> u64 {
    parameter
        .as_ref()
        .map_or(0, |parameter| instrument::tensor_bytes(&parameter.val()))
}

/// How the windows of a pool or a convolution slide over each spatial dimension of its input:
/// the taps in a window, the step between windows, the padding before the input and the spacing
/// between taps.
pub(crate) struct Windows<const N: usize> {
    pub(crate) kernel_size: [usize; N],
    pub(crate) stride: [usize; N],
    pub(crate) padding_begin: [usize; N],
    pub(crate) dilation: [usize; N],
}

impl<const N: usize> Windows<N> {
    /// The windows of a convolution with a kernel of `kernel_size`.
    pub(crate) fn convolution(kernel_size: [usize; N], options: &ConvOptions<N>) -> Self {
        Self {
            kernel_size,
            stride: options.stride,
            padding_begin: options.padding_begin(),
            dilation: options.dilation,
        }
    }

    /// The windows of a max pool.
    pub(crate) fn max_pool(options: &MaxPoolOptions<N>) -> Self {
        Self {
            kernel_size: options.kernel_size,
            stride: options.stride,
            padding_begin: options.padding.map(|(begin, _)| begin),
            dilation: options.dilation,
        }
    }

    /// The windows of an average pool.
    pub(crate) fn avg_pool(options: &AvgPoolOptions<N>) -> Self {
        Self {
            kernel_size: options.kernel_size,
            stride: options.stride,
            padding_begin: options.padding.map(|(begin, _)| begin),
            dilation: [1; N],
        }
    }

    /// What `output_size` of these windows make of `input`: the output's size, and the input
    /// positions some window reads, below the input's size where the windows step over
    /// positions, as a strided kernel narrower than its stride does.
    pub(crate) fn over<const D: usize>(
        &self,
        input: &Tensor<D>,
        output_size: [usize; N],
    ) -> Windowing<N> {
        let input_size: [usize; N] = spatial(input);
        let read_size = core::array::from_fn(|i| {
            let mut read = vec![false; input_size[i]];
            for window in 0..output_size[i] {
                for tap in 0..self.kernel_size[i] {
                    let at = (window * self.stride[i] + tap * self.dilation[i])
                        .checked_sub(self.padding_begin[i]);
                    if let Some(position) = at.filter(|at| *at < input_size[i]) {
                        read[position] = true;
                    }
                }
            }
            read.into_iter().filter(|read| *read).count()
        });
        Windowing {
            output_size,
            read_size,
        }
    }
}

/// How a pool, an interpolation or a convolution maps its input's spatial dimensions to its
/// output's: the size of the output, and how many positions of the input it reads along each.
pub(crate) struct Windowing<const N: usize> {
    pub(crate) output_size: [usize; N],
    pub(crate) read_size: [usize; N],
}

impl<const N: usize> Windowing<N> {
    /// The floor for resampling `input` this way alone: read the positions it reads once, write
    /// the output once. A window is a pass over the same bytes, so no arithmetic is declared.
    /// The pools and the interpolations.
    pub(crate) fn resampling<const D: usize>(&self, input: &Tensor<D>) -> Bounds {
        let leading = elements(&input.dims()[..D - N]);
        Bounds {
            read: self.read(input),
            write: leading * elements(&self.output_size) * input.dtype().size() as u64,
            flops: 0,
        }
    }

    /// The floor for convolving `input` this way alone: read the positions it reads and the
    /// parameters once, write the output once, each output element multiply-adding one output
    /// channel's kernel over its window.
    ///
    /// `weight` is `[channels_out, channels_in / groups, kernel_size..]`.
    pub(crate) fn convolution<const D: usize>(
        &self,
        input: &Tensor<D>,
        weight: &Tensor<D>,
        bias: &Option<Param<Tensor<1>>>,
    ) -> Bounds {
        let channels_out = weight.dims()[0] as u64;
        let outputs = input.dims()[0] as u64 * channels_out * elements(&self.output_size);
        let kernel = weight.shape().num_elements() as u64 / channels_out;
        Bounds {
            read: self.read(input) + instrument::tensor_bytes(weight) + optional_parameter(bias),
            write: outputs * input.dtype().size() as u64,
            flops: 2 * outputs * kernel,
        }
    }

    /// The bytes of `input` at the positions read, every leading index included.
    fn read<const D: usize>(&self, input: &Tensor<D>) -> u64 {
        match elements(&input.dims()[D - N..]) {
            0 => 0,
            spatial => {
                let read =
                    instrument::tensor_bytes(input) as u128 * elements(&self.read_size) as u128;
                (read / spatial as u128) as u64
            }
        }
    }
}

/// The floor for a transposed convolution alone: read the input and the parameters once, write
/// the output once, each input element multiply-adding one input channel's kernel into its
/// window.
///
/// `weight` is `[channels_in, channels_out / groups, kernel_size..]` and `output_size` the
/// spatial size of the output.
pub(crate) fn transposed_convolution<const D: usize, const N: usize>(
    input: &Tensor<D>,
    weight: &Tensor<D>,
    bias: &Option<Param<Tensor<1>>>,
    groups: usize,
    output_size: [usize; N],
) -> Bounds {
    let [channels_in, channels_out_per_group] = [weight.dims()[0], weight.dims()[1]];
    let channels_out = (channels_out_per_group * groups) as u64;
    let outputs = input.dims()[0] as u64 * channels_out * elements(&output_size);
    let kernel = weight.shape().num_elements() as u64 / channels_in as u64;
    let inputs = input.shape().num_elements() as u64;
    Bounds {
        read: instrument::tensor_bytes(input)
            + instrument::tensor_bytes(weight)
            + optional_parameter(bias),
        write: outputs * input.dtype().size() as u64,
        flops: 2 * inputs * kernel,
    }
}

fn elements(dims: &[usize]) -> u64 {
    dims.iter().map(|dim| *dim as u64).product()
}
