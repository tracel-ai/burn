use crate::{CubeAutotuneKey, CubeRuntime, kernel::autotune_bounds};
use burn_backend::cubecl::dtype_to_storage_type;
use cubecl::{client::ComputeClient, std::throughput::roofline_bounds, tune::TunableSet};
use cubek::convolution::{components::ConvolutionOperation, definition::ConvolutionCost};
use cubek::matmul::definition::MatmulGlobalElems;

use super::tune_key::{ConvAutotuneKey, ConvTranspose2dAutotuneKey};

/// Dimensions of a convolution, read from the tuned inputs rather than from the autotune key.
///
/// The key anchors its channel counts, batch size and spatial dimensions to powers of two, which
/// would inflate every term of the cost. Candidates are benchmarked on the real tensors, so the
/// bounds they are compared against have to be built from the real dimensions too. Everything the
/// key holds exactly (kernel size, stride, padding, dilation, groups, dtype) still comes from it.
pub(super) struct ConvDims<'a> {
    pub batch_size: usize,
    pub in_channels: usize,
    pub out_channels: usize,
    /// Spatial dimensions of the input activation.
    pub in_shape: &'a [usize],
}

/// Registers the performance bounds used for convolution autotuning.
pub(super) fn with_conv_bounds<R, I, Out>(
    set: TunableSet<CubeAutotuneKey, I, Out>,
    operation: ConvolutionOperation,
    dims: impl Fn(&I) -> (&ComputeClient<R>, ConvDims<'_>) + Send + Sync + 'static,
) -> TunableSet<CubeAutotuneKey, I, Out>
where
    R: CubeRuntime,
    I: Clone + Send + Sync + 'static,
    Out: 'static,
{
    with_cost(set, dims, move |key, dims| match key {
        CubeAutotuneKey::Conv(key) => Some(cost_conv2d(key, dims, operation)),
        // A convolution tuner only ever keys on a convolution. Should another key reach here,
        // no cost model applies to it, and no bounds is the honest answer.
        CubeAutotuneKey::ConvTranspose(_) | CubeAutotuneKey::Sum(_) => None,
    })
}

/// Registers the performance bounds used for transposed convolution autotuning.
pub(super) fn with_conv_transpose2d_bounds<R, I, Out>(
    set: TunableSet<CubeAutotuneKey, I, Out>,
    dims: impl Fn(&I) -> (&ComputeClient<R>, ConvDims<'_>) + Send + Sync + 'static,
) -> TunableSet<CubeAutotuneKey, I, Out>
where
    R: CubeRuntime,
    I: Clone + Send + Sync + 'static,
    Out: 'static,
{
    with_cost(set, dims, |key, dims| match key {
        CubeAutotuneKey::ConvTranspose(key) => Some(cost_conv_transpose2d(key, dims)),
        CubeAutotuneKey::Conv(_) | CubeAutotuneKey::Sum(_) => None,
    })
}

/// Registers roofline bounds built from `cost`, leaving them empty when no cost model applies.
fn with_cost<R, I, Out>(
    set: TunableSet<CubeAutotuneKey, I, Out>,
    dims: impl Fn(&I) -> (&ComputeClient<R>, ConvDims<'_>) + Send + Sync + 'static,
    cost: impl Fn(&CubeAutotuneKey, &ConvDims<'_>) -> Option<ConvolutionCost> + Send + Sync + 'static,
) -> TunableSet<CubeAutotuneKey, I, Out>
where
    R: CubeRuntime,
    I: Clone + Send + Sync + 'static,
    Out: 'static,
{
    autotune_bounds::with_bounds(set, move |key, inputs, thresholds| {
        let (client, dims) = dims(inputs);

        match cost(key, &dims) {
            Some(cost) => {
                roofline_bounds(client, cost.compute_key(client), cost.work(), thresholds)
            }
            None => autotune_bounds::no_bounds(),
        }
    })
}

/// Computes an output spatial dimension for a convolution.
///
/// Saturates instead of wrapping, so a kernel wider than its padded input still costs the one
/// output position every launch writes rather than underflowing.
fn spatial_out(
    in_size: usize,
    kernel_size: usize,
    padding: usize,
    stride: usize,
    dilation: usize,
) -> usize {
    let padded = in_size + 2 * padding;
    let span = dilation * kernel_size.saturating_sub(1) + 1;

    padded.saturating_sub(span) / stride + 1
}

/// Computes an output spatial dimension for a transposed convolution.
fn spatial_out_transposed(
    in_size: usize,
    kernel_size: usize,
    padding: usize,
    padding_out: usize,
    stride: usize,
    dilation: usize,
) -> usize {
    let span = in_size.saturating_sub(1) * stride + dilation * kernel_size.saturating_sub(1) + 1;

    span.saturating_sub(2 * padding).max(1) + padding_out
}

/// Builds the convolution cost from a [`ConvAutotuneKey`] and the dimensions of its inputs.
///
/// `dims.in_channels` and `dims.out_channels` are the tensors' true, ungrouped channel counts, but
/// the weight tensor only holds `in_channels / groups` channels per output channel. Every term
/// that scales with the weight's channel footprint is divided by `groups` accordingly, so byte
/// traffic and compute work reflect the actual per-group contraction rather than a full one.
fn cost_conv2d(
    key: &ConvAutotuneKey,
    dims: &ConvDims<'_>,
    operation: ConvolutionOperation,
) -> ConvolutionCost {
    let in_spatial: usize = dims.in_shape.iter().product();
    let kernel_spatial: usize = key.kernel_size.iter().product();
    let in_channels_per_group = dims.in_channels / key.groups;
    let out_channels_per_group = dims.out_channels / key.groups;

    let out_spatial: usize = (0..dims.in_shape.len())
        .map(|i| {
            spatial_out(
                dims.in_shape[i],
                key.kernel_size[i],
                key.padding[i],
                key.stride[i],
                key.dilation[i],
            )
        })
        .product();

    let act_in_elements = dims.batch_size * dims.in_channels * in_spatial;
    let act_out_elements = dims.batch_size * dims.out_channels * out_spatial;
    let weight_elements = dims.out_channels * in_channels_per_group * kernel_spatial;

    let (m, n, k) = match operation {
        ConvolutionOperation::Forward => (
            dims.batch_size * out_spatial,
            dims.out_channels,
            in_channels_per_group * kernel_spatial,
        ),
        ConvolutionOperation::BackwardData | ConvolutionOperation::ForwardTransposed => (
            dims.batch_size * in_spatial,
            dims.in_channels,
            out_channels_per_group * kernel_spatial,
        ),
        ConvolutionOperation::BackwardWeight => (
            dims.out_channels,
            in_channels_per_group * kernel_spatial,
            dims.batch_size * out_spatial,
        ),
    };

    ConvolutionCost {
        m,
        n,
        k,
        act_in_elements,
        act_out_elements,
        weight_elements,
        operation,
        dtypes: global_elems(key.dtype),
    }
}

/// Builds the transposed convolution cost from a [`ConvTranspose2dAutotuneKey`] and the dimensions
/// of its inputs.
///
/// The weight tensor is laid out as `[in_channels, out_channels / groups, kernel_h, kernel_w]`, so
/// its footprint and the contraction both cover a single group, the same way [`cost_conv2d`]
/// handles them.
fn cost_conv_transpose2d(key: &ConvTranspose2dAutotuneKey, dims: &ConvDims<'_>) -> ConvolutionCost {
    let in_spatial: usize = dims.in_shape.iter().product();
    let kernel_spatial = key.kernel_size[0] * key.kernel_size[1];
    let in_channels_per_group = dims.in_channels / key.groups;
    let out_channels_per_group = dims.out_channels / key.groups;

    let out_spatial: usize = (0..2)
        .map(|i| {
            spatial_out_transposed(
                dims.in_shape[i],
                key.kernel_size[i],
                key.padding[i],
                key.padding_out[i],
                key.stride[i],
                key.dilation[i],
            )
        })
        .product();

    let act_in_elements = dims.batch_size * dims.in_channels * in_spatial;
    let act_out_elements = dims.batch_size * dims.out_channels * out_spatial;
    let weight_elements = dims.in_channels * out_channels_per_group * kernel_spatial;

    ConvolutionCost {
        m: dims.batch_size * in_spatial,
        n: dims.out_channels,
        k: in_channels_per_group * kernel_spatial,
        act_in_elements,
        act_out_elements,
        weight_elements,
        operation: ConvolutionOperation::ForwardTransposed,
        dtypes: global_elems(key.dtype),
    }
}

/// Both activations and the weight of a convolution share a single element type.
fn global_elems(dtype: burn_backend::DType) -> MatmulGlobalElems {
    let storage_type = dtype_to_storage_type(dtype);

    MatmulGlobalElems {
        lhs: storage_type,
        rhs: storage_type,
        out: storage_type,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_backend::DType;

    fn conv_key(groups: usize) -> ConvAutotuneKey {
        ConvAutotuneKey {
            kernel_size: vec![3, 3],
            stride: vec![1, 1],
            padding: vec![1, 1],
            dilation: vec![1, 1],
            groups,
            in_channels: 0,
            out_channels: 0,
            shape: Vec::new(),
            batch_size: 0,
            has_bias: false,
            dtype: DType::F32,
            lhs_shape_align: 0,
            lhs_stride_align: 0,
            rhs_shape_align: 0,
            rhs_stride_align: 0,
        }
    }

    fn transpose_key(groups: usize) -> ConvTranspose2dAutotuneKey {
        ConvTranspose2dAutotuneKey {
            kernel_size: [3, 3],
            stride: [1, 1],
            padding: [1, 1],
            padding_out: [0, 0],
            dilation: [1, 1],
            groups,
            in_channels: 0,
            out_channels: 0,
            height: 0,
            width: 0,
            batch_size: 0,
            has_bias: false,
            dtype: DType::F32,
        }
    }

    fn dims(in_shape: &[usize]) -> ConvDims<'_> {
        ConvDims {
            batch_size: 2,
            in_channels: 8,
            out_channels: 16,
            in_shape,
        }
    }

    #[test]
    fn a_padded_kernel_keeps_the_spatial_dimensions() {
        let cost = cost_conv2d(
            &conv_key(1),
            &dims(&[32, 32]),
            ConvolutionOperation::Forward,
        );

        // 2 batches of 32x32 outputs, 16 channels each.
        assert_eq!(cost.m, 2 * 32 * 32);
        assert_eq!(cost.n, 16);
        assert_eq!(cost.k, 8 * 3 * 3);
        assert_eq!(cost.act_out_elements, 2 * 16 * 32 * 32);
    }

    #[test]
    fn a_stride_divides_the_output_positions() {
        let mut key = conv_key(1);
        key.stride = vec![2, 2];

        let cost = cost_conv2d(&key, &dims(&[32, 32]), ConvolutionOperation::Forward);

        assert_eq!(cost.m, 2 * 16 * 16);
    }

    #[test]
    fn a_dilation_widens_the_kernel_span() {
        let mut key = conv_key(1);
        key.dilation = vec![2, 2];

        let cost = cost_conv2d(&key, &dims(&[32, 32]), ConvolutionOperation::Forward);

        // A 3x3 kernel dilated by 2 spans 5, leaving 30 positions out of a 34-wide padded input.
        assert_eq!(cost.m, 2 * 30 * 30);
    }

    #[test]
    fn a_kernel_wider_than_its_input_still_costs_one_position() {
        let mut key = conv_key(1);
        key.kernel_size = vec![7, 7];

        let cost = cost_conv2d(&key, &dims(&[2, 2]), ConvolutionOperation::Forward);

        assert_eq!(cost.m, 2);
    }

    #[test]
    fn groups_shrink_the_contraction_and_the_weight() {
        let grouped = cost_conv2d(
            &conv_key(4),
            &dims(&[32, 32]),
            ConvolutionOperation::Forward,
        );
        let dense = cost_conv2d(
            &conv_key(1),
            &dims(&[32, 32]),
            ConvolutionOperation::Forward,
        );

        assert_eq!(grouped.k * 4, dense.k);
        assert_eq!(grouped.weight_elements * 4, dense.weight_elements);
        // Activations are whole tensors, so grouping leaves them untouched.
        assert_eq!(grouped.act_in_elements, dense.act_in_elements);
        assert_eq!(grouped.act_out_elements, dense.act_out_elements);
    }

    #[test]
    fn a_data_gradient_contracts_over_the_output_channels() {
        let cost = cost_conv2d(
            &conv_key(4),
            &dims(&[32, 32]),
            ConvolutionOperation::BackwardData,
        );

        assert_eq!(cost.m, 2 * 32 * 32);
        assert_eq!(cost.n, 8);
        assert_eq!(cost.k, (16 / 4) * 3 * 3);
    }

    #[test]
    fn a_weight_gradient_contracts_over_the_output_positions() {
        let cost = cost_conv2d(
            &conv_key(4),
            &dims(&[32, 32]),
            ConvolutionOperation::BackwardWeight,
        );

        assert_eq!(cost.m, 16);
        assert_eq!(cost.n, (8 / 4) * 3 * 3);
        assert_eq!(cost.k, 2 * 32 * 32);
    }

    #[test]
    fn a_transposed_convolution_contracts_over_the_input_channels() {
        let cost = cost_conv_transpose2d(&transpose_key(1), &dims(&[32, 32]));

        assert_eq!(cost.m, 2 * 32 * 32);
        assert_eq!(cost.n, 16);
        assert_eq!(cost.k, 8 * 3 * 3);
        assert_eq!(cost.weight_elements, 8 * 16 * 3 * 3);
    }

    #[test]
    fn transposed_groups_shrink_the_contraction_and_the_weight() {
        let grouped = cost_conv_transpose2d(&transpose_key(4), &dims(&[32, 32]));
        let dense = cost_conv_transpose2d(&transpose_key(1), &dims(&[32, 32]));

        assert_eq!(grouped.k * 4, dense.k);
        assert_eq!(grouped.weight_elements * 4, dense.weight_elements);
    }

    #[test]
    fn a_transposed_stride_expands_the_output_positions() {
        let mut key = transpose_key(1);
        key.stride = [2, 2];
        key.padding_out = [1, 1];

        let cost = cost_conv_transpose2d(&key, &dims(&[32, 32]));

        // (32 - 1) * 2 + 3 - 2 * 1 + 1 output positions, and one more for the output padding.
        assert_eq!(cost.act_out_elements, 2 * 16 * 64 * 64);
    }
}
