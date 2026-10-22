use burn_backend::{cubecl::dtype_to_storage_type, ops::ConvOptions};
use burn_std::{Metadata, Shape};
use cubek::{
    convolution::{
        AcceleratedTileKind, ConvAlgorithm, ConvolutionArgs, ConvolutionInputs, Strategy,
        components::ConvSetupError, launch_ref,
    },
    matmul::definition::{MatmulElems, MatmulGlobalElems},
    std::InputBinding,
};

use super::fallback::conv_data_backward_fallback;
use crate::{CubeDevice, ops::numeric::empty_device_dtype, tensor::CubeTensor};

/// Selects one residue in each NHWC spatial axis using metadata and a storage offset.
/// Preserves rank, batch, channels, dtype, and device. Spatial axes shrink by `stride`.
/// Requires untiled rank `N + 2` storage, in-range starts, and aligned phase offsets.
fn spatial_phase<const N: usize>(
    tensor: &CubeTensor,
    start: &[usize; N],
    stride: &[usize; N],
) -> CubeTensor {
    let mut shape = tensor.meta.shape().to_vec();
    let mut strides = tensor.meta.strides().to_vec();
    let mut offset = 0;
    for axis in 0..N {
        offset += start[axis] * strides[axis + 1];
        shape[axis + 1] = (shape[axis + 1] - start[axis]).div_ceil(stride[axis]);
        strides[axis + 1] *= stride[axis];
    }
    CubeTensor::new(
        tensor.client.clone(),
        tensor
            .handle
            .clone()
            .offset_start((offset * tensor.dtype.size()) as u64),
        Metadata::new(shape, strides),
        tensor.device.clone(),
        tensor.dtype,
    )
}

/// Computes NHWC input gradients from NHWC output gradients and channel-last weights.
/// Splits input pixels by their spatial coordinates modulo the convolution stride.
/// Each phase uses a unit-stride convolution with only the matching filter taps.
pub(crate) fn dgrad_stride_phases<const N: usize>(
    out_grad: CubeTensor,
    weights: CubeTensor,
    input_shape: Shape,
    options: ConvOptions<N>,
) -> Result<CubeTensor, ConvSetupError> {
    if options.groups != 1 {
        return Err(ConvSetupError::Groups(options.groups));
    }
    let phase_count = options
        .stride
        .iter()
        .try_fold(1usize, |count, stride| count.checked_mul(*stride));
    if !matches!(out_grad.device, CubeDevice::Cuda(_))
        || !matches!(phase_count, Some(2..=16))
        || options.dilation.iter().any(|dilation| *dilation != 1)
        || options
            .padding
            .iter()
            .any(|padding| padding.0 > i32::MAX as usize || padding.1 > i32::MAX as usize)
        || !out_grad.dtype.is_float()
        || !weights.dtype.is_float()
    {
        return Err(ConvSetupError::Unknown);
    }
    let out_grad = crate::kernel::untile(out_grad);
    let weights = crate::kernel::into_contiguous_aligned(weights);
    let contraction = weights.meta.shape()[..N + 1]
        .iter()
        .try_fold(1usize, |size, extent| size.checked_mul(*extent));
    if !contraction.is_some_and(|size| size > 0 && size <= u32::MAX as usize) {
        return Err(ConvSetupError::Unknown);
    }
    let alignment = out_grad.client.properties().io_width() as usize / 8;
    // CUDA supports byte-offset pointers; keep each view aligned to vector I/O.
    // These constraints use channel counts and stride alignment held by the tuning key.
    if !(input_shape[N + 1] * out_grad.dtype.size()).is_multiple_of(alignment)
        || weights.meta.strides()[N + 1] != 1
        || weights.meta.strides()[..N + 1]
            .iter()
            .any(|stride| !(stride * weights.dtype.size()).is_multiple_of(alignment))
        || (0..N).any(|axis| options.stride[axis] > weights.meta.shape()[axis + 1])
    {
        return Err(ConvSetupError::Unknown);
    }
    // A cached shape can cross the layout's coordinate limits. Keep that case
    // usable through the existing fallback instead of failing a cached selection.
    if !input_shape[..N + 1]
        .iter()
        .try_fold(1usize, |size, extent| size.checked_mul(*extent))
        .is_some_and(|size| size <= u32::MAX as usize)
        || (0..N).any(|axis| {
            input_shape[axis + 1] as u128
                + options.padding[axis].0 as u128
                + options.padding[axis].1 as u128
                > i32::MAX as u128
        })
    {
        return conv_data_backward_fallback(out_grad, weights, input_shape, options);
    }

    let input_grad = empty_device_dtype(
        out_grad.client.clone(),
        out_grad.device.clone(),
        input_shape.clone(),
        out_grad.dtype,
    );
    launch_phases(&out_grad, &weights, &input_grad, &options)
        .map(|()| input_grad)
        .or_else(|_| conv_data_backward_fallback(out_grad, weights, input_shape, options))
}

/// Reads NHWC gradients and channel-last weights into disjoint views of the NHWC input gradient.
/// Requires unit dilation, nonzero strides, matching filter taps, and aligned view offsets.
fn launch_phases<const N: usize>(
    out_grad: &CubeTensor,
    weights: &CubeTensor,
    input_grad: &CubeTensor,
    options: &ConvOptions<N>,
) -> Result<(), ConvSetupError> {
    let phase_count = options.stride.iter().product::<usize>();
    let dtypes = MatmulElems::from_globals(&MatmulGlobalElems {
        lhs: dtype_to_storage_type(out_grad.dtype),
        rhs: dtype_to_storage_type(weights.dtype),
        out: dtype_to_storage_type(out_grad.dtype),
    });
    let strategy = Strategy::Inferred {
        algorithm: ConvAlgorithm::SimpleSyncStrided,
        tile_kind: AcceleratedTileKind::Mma,
    };
    for mut phase in 0..phase_count {
        let mut input_start = [0; N];
        let mut weight_start = [0; N];
        let mut padding = [0; N];
        for axis in (0..N).rev() {
            input_start[axis] = phase % options.stride[axis];
            phase /= options.stride[axis];
            let shifted = input_start[axis] + options.padding[axis].0;
            weight_start[axis] = shifted % options.stride[axis];
            padding[axis] = shifted / options.stride[axis];
        }
        if (0..N).any(|axis| input_start[axis] >= input_grad.meta.shape()[axis + 1]) {
            continue;
        }
        let output = spatial_phase(input_grad, &input_start, &options.stride);
        let filter = spatial_phase(weights, &weight_start, &options.stride);
        launch_ref::<N>(
            &strategy,
            &out_grad.client,
            ConvolutionInputs::BackwardData {
                out_grad: InputBinding::new(
                    (*out_grad).clone().binding(),
                    dtype_to_storage_type(out_grad.dtype),
                ),
                weights: InputBinding::new(filter.binding(), dtype_to_storage_type(weights.dtype)),
                in_grad: output.binding(),
            },
            ConvolutionArgs {
                stride: [1; N],
                padding,
                dilation: [1; N],
            },
            dtypes.clone(),
        )?;
    }
    Ok(())
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;
    use burn_backend::{TensorData, ops::conv::calculate_conv_output_sizes};

    /// Checks NHWC input gradients against host accumulation of each output-gradient contribution.
    fn assert_input_gradient(
        kernel: [usize; 2],
        options: ConvOptions<2>,
        strided_weights: bool,
        channels: usize,
        check_kernel: bool,
    ) {
        let device = CubeDevice::cuda(0).unwrap();
        let probe = crate::ops::from_data(TensorData::new(vec![0.0_f32], [1]), &device);
        let channels = (probe.client.properties().io_width() as usize / 8 / 4).max(channels);
        let shape = [2, 11, 13, channels];
        let weight_shape = [8, kernel[0], kernel[1], channels];
        let weights: Vec<f32> = (0..weight_shape.iter().product())
            .map(|i| ((i % 11) as i32 - 5) as f32 / 8.0)
            .collect();
        let mut weights = crate::ops::from_data(TensorData::new(weights, weight_shape), &device);
        if strided_weights {
            weights = crate::ops::swap_dims(weights, 1, 2);
        }
        let kernel = [weights.meta.shape()[1], weights.meta.shape()[2]];
        let weight_values = crate::ops::into_data_sync(weights.clone())
            .try_to_vec::<f32>()
            .unwrap();
        let out_spatial = calculate_conv_output_sizes(
            &kernel,
            &options.stride,
            &options.padding,
            &options.dilation,
            &shape[1..3],
        );
        let grad_shape = [2, out_spatial[0], out_spatial[1], 8];
        let grad_values: Vec<f32> = (0..grad_shape.iter().product())
            .map(|i| ((i % 7) as i32 - 3) as f32 / 4.0)
            .collect();
        let mut expected = vec![0.0; shape.iter().product()];
        for b in 0..2 {
            for y in 0..out_spatial[0] {
                for x in 0..out_spatial[1] {
                    for ky in 0..kernel[0] {
                        for kx in 0..kernel[1] {
                            let input_y = (y * options.stride[0] + ky) as isize
                                - options.padding[0].0 as isize;
                            let input_x = (x * options.stride[1] + kx) as isize
                                - options.padding[1].0 as isize;
                            if !(0..11).contains(&input_y) || !(0..13).contains(&input_x) {
                                continue;
                            }
                            for co in 0..8 {
                                for ci in 0..channels {
                                    expected[((b * 11 + input_y as usize) * 13
                                        + input_x as usize)
                                        * channels
                                        + ci] += grad_values
                                        [((b * out_spatial[0] + y) * out_spatial[1] + x) * 8 + co]
                                        * weight_values[((co * kernel[0] + ky) * kernel[1] + kx)
                                            * channels
                                            + ci];
                                }
                            }
                        }
                    }
                }
            }
        }
        let grad = crate::ops::from_data(TensorData::new(grad_values, grad_shape), &device);
        let checked =
            dgrad_stride_phases(grad.clone(), weights.clone(), shape.into(), options.clone())
                .unwrap();
        assert_values(checked, &expected);
        if check_kernel {
            let output = empty_device_dtype(grad.client.clone(), device, shape.into(), grad.dtype);
            launch_phases(&grad, &weights, &output, &options).unwrap();
            assert_values(output, &expected);
        }
    }

    /// Checks rank-four NHWC values in batch, row, column, channel order.
    fn assert_values(output: CubeTensor, expected: &[f32]) {
        let actual = crate::ops::into_data_sync(output)
            .try_to_vec::<f32>()
            .unwrap();
        assert_eq!(actual.len(), expected.len());
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert_eq!(actual, expected, "input-gradient element {index}");
        }
    }

    #[test]
    fn stride_phases_match_host_input_gradients() {
        assert_input_gradient(
            [4, 4],
            ConvOptions::new([2; 2], [0; 2], [1; 2], 1),
            false,
            32,
            true,
        );
    }

    #[test]
    fn stride_phases_handle_asymmetric_padding_and_unequal_strides() {
        assert_input_gradient(
            [3, 4],
            ConvOptions::new_with_padding([2, 3], [(1, 2), (2, 1)], [1; 2], 1),
            false,
            32,
            true,
        );
    }

    #[test]
    fn stride_phases_handle_strided_weights() {
        assert_input_gradient(
            [4, 4],
            ConvOptions::new([2; 2], [1; 2], [1; 2], 1),
            true,
            32,
            true,
        );
    }

    #[test]
    fn stride_phases_support_small_channel_counts() {
        assert_input_gradient(
            [3, 4],
            ConvOptions::new_with_padding([2, 3], [(1, 2), (2, 1)], [1; 2], 1),
            false,
            4,
            false,
        );
    }
    #[test]
    fn stride_phases_accept_channel_strided_weights() {
        let device = CubeDevice::cuda(0).unwrap();
        let weights: Vec<f32> = (0..8 * 32 * 4 * 4)
            .map(|index| (index % 7) as f32 / 8.0)
            .collect();
        let weights = crate::ops::from_data(TensorData::new(weights, [8, 32, 4, 4]), &device);
        let weights = crate::ops::permute_nchw_to_nhwc(weights);
        assert_ne!(weights.meta.strides()[3], 1);
        let gradients: Vec<f32> = (0..2 * 4 * 5 * 8)
            .map(|index| (index % 5) as f32 / 4.0)
            .collect();
        let gradients = crate::ops::from_data(TensorData::new(gradients, [2, 4, 5, 8]), &device);
        let shape = Shape::new([2, 11, 13, 32]);
        let options = ConvOptions::new([2; 2], [0; 2], [1; 2], 1);
        let expected = conv_data_backward_fallback(
            gradients.clone(),
            weights.clone(),
            shape.clone(),
            options.clone(),
        )
        .unwrap();
        let expected = crate::ops::into_data_sync(expected)
            .try_to_vec::<f32>()
            .unwrap();
        let actual = dgrad_stride_phases(gradients, weights, shape, options).unwrap();
        assert_values(actual, &expected);
    }
}
