use burn_backend::{DType, cubecl::dtype_to_storage_type, ops::ConvOptions};
use burn_std::Shape;
use cubecl::{
    prelude::*,
    std::tensor::{
        launch::ViewArg,
        layout::{
            Layout, LayoutExpand, VirtualLayoutLaunch,
            chain::{Chain, ChainLaunch},
        },
    },
    tune::anchor,
};
use cubek::{
    convolution::components::{
        ConvSetupError, ConvolutionOperation, ConvolutionParams, ConvolutionProblem,
        Dimensionality,
        global::layout::{
            Im2colLayout, Im2colLayoutLaunch, NhwcLayout, NhwcLayoutLaunch, OutLayout,
            OutLayoutLaunch,
        },
    },
    matmul::{
        definition::{AccumulatorOperand, AvailableVectorSizes, MatmulGlobalElems},
        multi_level::{
            BatchMatmulRoutine,
            args::{BatchedCoords, TensorArgs, TensorInputsLaunch, TensorOutputLaunch},
            components::{
                global::{
                    memory::{
                        GlobalLayout, GlobalLayoutLaunch, NoopLayout, NoopLayoutLaunch, Transpose,
                        TransposeLaunch,
                    },
                    read::sync_full_strided::SyncFullStridedLoading,
                },
                tile::TileMatmulKind,
            },
            definition::Blueprint,
            routines::batch::simple::{SimpleAlgorithm, SimpleArgs},
        },
        routine::BlueprintStrategy,
    },
    std::MatrixLayout,
};

use crate::{
    kernel::{
        reduce::{KernelReduceStrategy, reduce_dim},
        utils::address_type,
    },
    ops::{numeric::empty_device_dtype, reshape},
    tensor::CubeTensor,
};
use cubek::reduce::components::instructions::ReduceOperationConfig;

/// Checks NHWC shapes and convolution coordinates before the layouts narrow indices.
fn check_coordinates<const N: usize>(
    input: &CubeTensor,
    weight_shape: &Shape,
    options: &ConvOptions<N>,
) -> Result<(), ConvSetupError> {
    // Use the largest spatial shape in the tuning key, so a cached selection
    // cannot cross the coordinate limit when the actual spatial sizes change.
    let mut max_rows = input.meta.shape()[0] as u128;
    let fits = input
        .meta
        .shape()
        .iter()
        .all(|value| *value <= u32::MAX as usize)
        && (0..N).all(|axis| {
            let extent = anchor(input.meta.shape()[axis + 1], None, None, None) as u128;
            let kernel = weight_shape[axis + 1] as u128;
            let stride = options.stride[axis] as u128;
            let dilation = options.dilation[axis] as u128;
            let padding = options.padding[axis].0 as u128;
            let end_padding = options.padding[axis].1 as u128;
            let limit = i32::MAX as u128;
            let padded = extent + padding + end_padding;
            let kernel_extent = kernel.saturating_sub(1) * dilation;
            let valid = extent > 0
                && kernel > 0
                && stride > 0
                && stride <= limit
                && dilation <= limit
                && padding <= limit
                && kernel_extent <= limit
                && padded <= limit + 1
                && padded > kernel_extent;
            if valid {
                max_rows *= (padded - kernel_extent - 1) / stride + 1;
            }
            valid && max_rows <= u32::MAX as u128
        });
    if !fits {
        return Err(ConvSetupError::InvalidConfig(Box::new(
            "implicit weight-gradient source coordinates exceed the layout index range",
        )));
    }
    Ok(())
}

/// Maps a split's matrix coordinates to the full contraction without copying inputs.
#[derive(CubeType, CubeLaunch, Clone)]
struct SplitContraction {
    rows: u32,
    cols: u32,
    contraction: u32,
    #[cube(comptime)]
    offset_columns: bool,
    #[cube(comptime)]
    spatial_last: bool,
    #[cube(comptime)]
    taps: u32,
    #[cube(comptime)]
    channels: u32,
}

#[cube]
impl Layout for SplitContraction {
    type Coordinates = BatchedCoords;
    type SourceCoordinates = BatchedCoords;

    fn to_source_pos(&self, pos: BatchedCoords) -> BatchedCoords {
        let (split, row, col) = pos;
        let offset = split as u32 * self.contraction;
        if self.offset_columns.comptime() {
            (0, row, col + offset)
        } else {
            let col = if self.spatial_last.comptime() {
                (col % self.taps) * self.channels + col / self.taps
            } else {
                col
            };
            (0, row + offset, col)
        }
    }

    fn to_source_pos_checked(&self, pos: BatchedCoords) -> (BatchedCoords, bool) {
        (self.to_source_pos(pos), self.is_in_bounds(pos))
    }

    fn shape(&self) -> BatchedCoords {
        (1, self.rows, self.cols)
    }

    fn is_in_bounds(&self, pos: BatchedCoords) -> bool {
        pos.1 < self.rows && pos.2 < self.cols
    }
}

/// Computes weights `[out_channels, ...kernel, in_channels]` from NHWC input and gradients.
/// Independent contraction slices produce partial weights, then reduce the added split axis.
/// Preserves the gradient dtype and device. Input and gradient ranks are `N + 2`.
/// `SPATIAL_LAST` stores partial weights as `[split, out_channels, in_channels, ...kernel]`.
pub(crate) fn wgrad_split_implicit<const N: usize, const SPATIAL_LAST: bool>(
    input: CubeTensor,
    out_grad: CubeTensor,
    weight_shape: Shape,
    options: ConvOptions<N>,
) -> Result<CubeTensor, ConvSetupError> {
    if options.groups != 1 {
        return Err(ConvSetupError::Groups(options.groups));
    }

    let input = crate::kernel::untile(input);
    let out_grad = crate::kernel::untile(out_grad);
    check_coordinates(&input, &weight_shape, &options)?;
    let dim_c = N + 1;
    let channels = input.meta.shape()[dim_c];
    if channels == 0 {
        return Err(ConvSetupError::Unknown);
    }
    let out_channels = out_grad.meta.shape()[dim_c];
    let rows = out_grad.meta.shape()[..dim_c].iter().product::<usize>();
    let columns = weight_shape[1..].iter().product::<usize>();
    // Bound partial weights to 64 MiB even with eight-byte elements.
    let max_splits =
        (64 * 1024 * 1024 / weight_shape.num_elements().max(1) / DType::F64.size()).clamp(1, 64);
    let splits = (1..=max_splits)
        .rev()
        .find(|split| rows.is_multiple_of(*split) && rows / split >= 512)
        .unwrap_or(1);
    if rows > u32::MAX as usize || columns > u32::MAX as usize || out_channels > u32::MAX as usize {
        return Err(ConvSetupError::InvalidConfig(Box::new(
            "convolution matrix exceeds u32",
        )));
    }

    let partial = empty_device_dtype(
        input.client.clone(),
        input.device.clone(),
        [splits, out_channels, columns].into(),
        out_grad.dtype,
    );
    let conv = ConvolutionProblem {
        m: out_channels,
        n: columns,
        k: rows,
        lhs_strides: input.meta.strides().into(),
        rhs_strides: out_grad.meta.strides().into(),
        lhs_layout: MatrixLayout::ColMajor,
        rhs_layout: MatrixLayout::RowMajor,
        kernel_size: weight_shape[1..dim_c].iter().map(|v| *v as u32).collect(),
        stride: options.stride.iter().map(|v| *v as u32).collect(),
        padding: options.padding_begin().iter().map(|v| *v as i32).collect(),
        dilation: options.dilation.iter().map(|v| *v as u32).collect(),
        batches: input.meta.shape()[0],
        channels,
        out_channels,
        in_shape: input.meta.shape()[1..dim_c].into(),
        out_shape: out_grad.meta.shape()[1..dim_c].into(),
        padded_channels: channels,
        operation: ConvolutionOperation::BackwardWeight,
        dimensionality: match N {
            1 => Dimensionality::Dim1,
            2 => Dimensionality::Dim2,
            3 => Dimensionality::Dim3,
            _ => {
                return Err(ConvSetupError::InvalidConfig(Box::new(
                    "convolution requires one to three spatial axes",
                )));
            }
        },
        global_dtypes: MatmulGlobalElems {
            lhs: dtype_to_storage_type(out_grad.dtype),
            rhs: dtype_to_storage_type(input.dtype),
            out: dtype_to_storage_type(out_grad.dtype),
        },
        address_type: address_type!(input, out_grad, partial),
    };
    let mut problem = conv.as_matmul_problem(AccumulatorOperand::Absent);
    problem.k /= splits;
    problem.lhs_batches = [splits].into();
    problem.rhs_batches = [splits].into();
    problem.out_batches = [splits].into();
    problem.lhs_shape = [splits, out_channels, problem.k].into();
    problem.rhs_shape = [splits, problem.k, columns].into();
    problem.out_shape = [splits, out_channels, columns].into();
    problem.out_strides = partial.meta.strides().into();

    let client = input.client.clone();
    let vectors = AvailableVectorSizes::from_type_sizes(
        &client,
        out_grad.dtype.size(),
        input.dtype.size(),
        partial.dtype.size(),
    )
    .filter_lhs_with_tensor(
        out_grad.meta.strides(),
        out_grad.meta.shape(),
        MatrixLayout::RowMajor,
    );
    let vectors = if const { SPATIAL_LAST } {
        // A vector stays within one filter row. Each source row must also start
        // at a vector-aligned pixel, including convolution stride and padding.
        vectors.filter_rhs(|size| {
            *size == 1
                || (input.meta.strides()[N] == 1
                    && input.meta.shape()[N].is_multiple_of(*size)
                    && options.dilation[N - 1] == 1
                    && weight_shape[N].is_multiple_of(*size)
                    && options.stride[N - 1].is_multiple_of(*size)
                    && options.padding[N - 1].0.is_multiple_of(*size)
                    && input
                        .meta
                        .strides()
                        .iter()
                        .enumerate()
                        .all(|(axis, stride)| axis == N || stride.is_multiple_of(*size)))
        })
    } else {
        vectors.filter_rhs_with_tensor(
            input.meta.strides(),
            input.meta.shape(),
            MatrixLayout::RowMajor,
        )
    };
    let vectors = vectors
        .filter_out_with_tensor(partial.meta.strides(), partial.meta.shape())
        .pick_max()?;
    let settings =
        <SimpleAlgorithm<SyncFullStridedLoading, SyncFullStridedLoading> as BatchMatmulRoutine<
            (),
        >>::device_settings(&client, vectors);
    let strategy = BlueprintStrategy::Inferred(SimpleArgs {
        tile_matmul: TileMatmulKind::Mma,
        multi_rows: false,
    });
    let expanded =
        <SimpleAlgorithm<SyncFullStridedLoading, SyncFullStridedLoading> as BatchMatmulRoutine<
            (),
        >>::expand_blueprint(&problem, &settings, &strategy)?;
    let launch =
        <SimpleAlgorithm<SyncFullStridedLoading, SyncFullStridedLoading> as BatchMatmulRoutine<
            (),
        >>::prepare(&problem, &settings, expanded)?;
    let blueprint = &launch.blueprint;

    let lhs_layout = ChainLaunch::new(
        NhwcLayoutLaunch::unchecked(),
        TransposeLaunch::new(OutLayoutLaunch::from_args(
            &conv,
            blueprint.lhs_global_layout_config(),
        )),
    );
    let lhs_layout = ChainLaunch::new(
        lhs_layout,
        SplitContractionLaunch::new(
            out_channels as u32,
            problem.k as u32,
            problem.k as u32,
            true,
            false,
            1,
            1,
        ),
    );
    let rhs_layout = ChainLaunch::new(
        NhwcLayoutLaunch::checked(
            [cubek::convolution::components::global::layout::NhwcCheck::Spatial]
                .into_iter()
                .collect(),
        ),
        Im2colLayoutLaunch::from_args(
            &conv,
            ConvolutionParams::from_problem(&conv),
            blueprint.rhs_global_layout_config(),
        ),
    );
    let rhs_layout = ChainLaunch::new(
        rhs_layout,
        SplitContractionLaunch::new(
            problem.k as u32,
            columns as u32,
            problem.k as u32,
            false,
            SPATIAL_LAST,
            (columns / channels) as u32,
            channels as u32,
        ),
    );
    let inputs = TensorInputsLaunch::new(
        VirtualLayoutLaunch::new::<NoopLayout>(NoopLayoutLaunch::new()),
        ViewArg::new_tensor::<Chain<Chain<NhwcLayout, Transpose<OutLayout>>, SplitContraction>>(
            out_grad.binding().into_tensor_arg(),
            lhs_layout,
        ),
        VirtualLayoutLaunch::new::<NoopLayout>(NoopLayoutLaunch::new()),
        ViewArg::new_tensor::<Chain<Chain<NhwcLayout, Im2colLayout>, SplitContraction>>(
            input.binding().into_tensor_arg(),
            rhs_layout,
        ),
        ComptimeOptionArgs::None,
        ComptimeOptionArgs::None,
    );
    let binding = partial.clone().binding();
    let output_layout = GlobalLayoutLaunch::from_handle_batched(
        &binding,
        &problem,
        launch.vector_sizes.out,
        blueprint.out_global_layout_config(),
    );
    let output = TensorOutputLaunch::new(
        ViewArg::new_tensor::<GlobalLayout>(binding.into_tensor_arg(), output_layout),
        VirtualLayoutLaunch::new::<NoopLayout>(NoopLayoutLaunch::new()),
    );
    cubek::matmul::multi_level::launch_kernel::<
        TensorArgs,
        SimpleAlgorithm<SyncFullStridedLoading, SyncFullStridedLoading>,
    >(&client, inputs, output, (), launch)?;

    // Spatial-last columns store [out_channels, in_channels, ...kernel].
    // Restore the documented channel-last shape through metadata after reduction.
    let finish = |gradient| {
        if const { SPATIAL_LAST } {
            let mut shape = vec![out_channels, channels];
            shape.extend_from_slice(&weight_shape[1..N + 1]);
            crate::ops::permute_nchw_to_nhwc(reshape(gradient, shape.into()))
        } else {
            reshape(gradient, weight_shape.clone())
        }
    };
    if splits == 1 {
        return Ok(finish(partial));
    }

    let gradient = reduce_dim(
        partial,
        None,
        0,
        KernelReduceStrategy::default(),
        ReduceOperationConfig::Sum,
    )
    .map_err(|error| ConvSetupError::InvalidConfig(Box::new(error)))?;
    Ok(finish(gradient))
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;
    use crate::CubeDevice;
    use burn_backend::{DType, TensorData, ops::conv::calculate_conv_output_sizes};

    /// Compares NHWC inputs and gradients with a host weight-gradient sum.
    fn assert_weight_gradient(
        batch: usize,
        options: ConvOptions<2>,
        channels: usize,
        dtype: DType,
        strided: bool,
    ) {
        let shape = [batch, 13, 15, channels];
        let stride = options.stride[0];
        let dilation = options.dilation[0];
        let padding = options.padding[0].0;
        let spatial = calculate_conv_output_sizes(
            &[3, 3],
            &options.stride,
            &options.padding,
            &options.dilation,
            &shape[1..3],
        );
        let grad_shape = [batch, spatial[0], spatial[1], 8];
        let input: Vec<f32> = (0..shape.iter().product())
            .map(|i| ((i % 11) as i32 - 5) as f32 / 8.0)
            .collect();
        let grad: Vec<f32> = (0..grad_shape.iter().product())
            .map(|i| ((i % 7) as i32 - 3) as f32 / 4.0)
            .collect();
        let mut expected = vec![0.0; 8 * 3 * 3 * channels];
        for b in 0..batch {
            for y in 0..spatial[0] {
                for x in 0..spatial[1] {
                    for ky in 0..3 {
                        for kx in 0..3 {
                            let input_y = (y * stride + ky * dilation) as isize - padding as isize;
                            let input_x = (x * stride + kx * dilation) as isize - padding as isize;
                            if !(0..13).contains(&input_y) || !(0..15).contains(&input_x) {
                                continue;
                            }
                            for co in 0..8 {
                                for ci in 0..channels {
                                    expected[((co * 3 + ky) * 3 + kx) * channels + ci] +=
                                        input[((b * 13 + input_y as usize) * 15
                                            + input_x as usize)
                                            * channels
                                            + ci]
                                            * grad
                                                [((b * spatial[0] + y) * spatial[1] + x) * 8 + co];
                                }
                            }
                        }
                    }
                }
            }
        }
        let device = CubeDevice::cuda(0).unwrap();
        let input = if strided {
            let mut stored = Vec::with_capacity(input.len());
            for b in 0..batch {
                for y in 0..13 {
                    for c in 0..channels {
                        for x in 0..15 {
                            stored.push(input[((b * 13 + y) * 15 + x) * channels + c]);
                        }
                    }
                }
            }
            crate::ops::swap_dims(
                crate::ops::from_data(
                    TensorData::new(stored, [batch, 13, channels, 15]).convert_dtype(dtype),
                    &device,
                ),
                2,
                3,
            )
        } else {
            crate::ops::from_data(TensorData::new(input, shape).convert_dtype(dtype), &device)
        };
        let grad = crate::ops::from_data(
            TensorData::new(grad, grad_shape).convert_dtype(dtype),
            &device,
        );
        let outputs = [
            wgrad_split_implicit::<2, false>(
                input.clone(),
                grad.clone(),
                [8, 3, 3, channels].into(),
                options.clone(),
            ),
            wgrad_split_implicit::<2, true>(input, grad, [8, 3, 3, channels].into(), options),
        ];
        for output in outputs {
            let output = output.unwrap();
            assert_eq!(output.dtype, dtype);
            let actual = crate::ops::into_data_sync(output)
                .convert::<f32>()
                .try_to_vec::<f32>()
                .unwrap();
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn split_implicit_weight_gradient_matches_host() {
        assert_weight_gradient(
            8,
            ConvOptions::new([1; 2], [0; 2], [1; 2], 1),
            4,
            DType::F32,
            false,
        );
    }

    #[test]
    fn split_implicit_weight_gradient_handles_padding_stride_and_dilation() {
        assert_weight_gradient(
            32,
            ConvOptions::new([2; 2], [1; 2], [2; 2], 1),
            4,
            DType::F32,
            false,
        );
    }

    #[test]
    fn split_implicit_weight_gradient_preserves_reduced_precision() {
        for dtype in [DType::F16, DType::BF16] {
            assert_weight_gradient(
                8,
                ConvOptions::new([1; 2], [0; 2], [1; 2], 1),
                4,
                dtype,
                false,
            );
        }
    }

    #[test]
    fn split_implicit_weight_gradient_handles_strided_channels() {
        assert_weight_gradient(
            8,
            ConvOptions::new([1; 2], [0; 2], [1; 2], 1),
            3,
            DType::F32,
            true,
        );
    }

    #[test]
    fn split_implicit_rejects_wide_source_coordinates() {
        let device = CubeDevice::cuda(0).unwrap();
        let input = crate::ops::from_data(TensorData::new(vec![1.0_f32; 4], [1, 1, 1, 4]), &device);
        let grad = crate::ops::from_data(TensorData::new(vec![1.0_f32; 8], [1, 1, 1, 8]), &device);
        let options = ConvOptions::new([1; 2], [0; 2], [1usize << 31; 2], 1);
        assert!(matches!(
            wgrad_split_implicit::<2, false>(input, grad, [8, 3, 3, 4].into(), options),
            Err(ConvSetupError::InvalidConfig(_))
        ));
    }

    /// Compares rank `N + 2` NHWC gradients with materialized column multiplication.
    fn assert_spatial_axes<const N: usize>(shape: &[usize], kernel: &[usize]) {
        let options = ConvOptions::new([1; N], [0; N], [1; N], 1);
        let mut grad_shape = vec![shape[0]];
        grad_shape.extend(calculate_conv_output_sizes(
            kernel,
            &options.stride,
            &options.padding,
            &options.dilation,
            &shape[1..N + 1],
        ));
        grad_shape.push(8);
        let mut weight_shape = vec![8];
        weight_shape.extend_from_slice(kernel);
        weight_shape.push(shape[N + 1]);
        let weight_shape = Shape::from(weight_shape);
        let device = CubeDevice::cuda(0).unwrap();
        let input: Vec<f32> = (0..shape.iter().product())
            .map(|i| ((i % 11) as i32 - 5) as f32 / 8.0)
            .collect();
        let grad: Vec<f32> = (0..grad_shape.iter().product())
            .map(|i| ((i % 7) as i32 - 3) as f32 / 4.0)
            .collect();
        let input = crate::ops::from_data(TensorData::new(input, shape), &device);
        let grad = crate::ops::from_data(TensorData::new(grad, grad_shape), &device);
        let expected = crate::kernel::conv::im2col::wgrad_im2col(
            input.clone(),
            grad.clone(),
            weight_shape.clone(),
            options.clone(),
        )
        .unwrap();
        let outputs = [
            wgrad_split_implicit::<N, false>(
                input.clone(),
                grad.clone(),
                weight_shape.clone(),
                options.clone(),
            ),
            wgrad_split_implicit::<N, true>(input, grad, weight_shape, options),
        ];
        let expected = crate::ops::into_data_sync(expected)
            .try_to_vec::<f32>()
            .unwrap();
        for actual in outputs {
            assert_eq!(
                crate::ops::into_data_sync(actual.unwrap())
                    .try_to_vec::<f32>()
                    .unwrap(),
                expected
            );
        }
    }

    #[test]
    fn split_implicit_supports_one_and_three_spatial_axes() {
        assert_spatial_axes::<1>(&[16, 129, 4], &[3]);
        assert_spatial_axes::<3>(&[16, 5, 7, 9, 4], &[3, 3, 3]);
    }

    /// Checks NHWC views of NCHW storage with aligned and unaligned filter-row starts.
    #[test]
    fn spatial_last_weight_gradient_handles_vector_alignment() {
        let device = CubeDevice::cuda(0).unwrap();
        let shape = [32usize, 4, 16, 20];
        let values: Vec<f32> = (0..shape.iter().product())
            .map(|i| ((i % 11) as i32 - 5) as f32 / 8.0)
            .collect();
        let input = crate::ops::permute_nchw_to_nhwc(crate::ops::from_data(
            TensorData::new(values, shape),
            &device,
        ));

        for width in [20, 19] {
            let mut input = input.clone();
            *input.meta = burn_std::Metadata::new([32, 16, width, 4], input.meta.strides().clone());
            for options in [
                ConvOptions::new([2, 4], [0, 0], [1; 2], 1),
                ConvOptions::new([2, 4], [1, 4], [1; 2], 1),
                ConvOptions::new([2, 4], [1, 1], [1; 2], 1),
                ConvOptions::new([2, 3], [0, 0], [1; 2], 1),
                ConvOptions::new([2, 4], [0, 0], [2; 2], 1),
            ] {
                let spatial = calculate_conv_output_sizes(
                    &[4, 8],
                    &options.stride,
                    &options.padding,
                    &options.dilation,
                    &[16, width],
                );
                let grad_shape = [32, spatial[0], spatial[1], 8];
                let values: Vec<f32> = (0..grad_shape.iter().product())
                    .map(|i| ((i % 7) as i32 - 3) as f32 / 4.0)
                    .collect();
                let grad = crate::ops::from_data(TensorData::new(values, grad_shape), &device);
                let expected = crate::kernel::conv::im2col::wgrad_im2col(
                    input.clone(),
                    grad.clone(),
                    [8, 4, 8, 4].into(),
                    options.clone(),
                )
                .unwrap();
                let actual = wgrad_split_implicit::<2, true>(
                    input.clone(),
                    grad,
                    [8, 4, 8, 4].into(),
                    options,
                )
                .unwrap();
                assert_eq!(
                    crate::ops::into_data_sync(actual)
                        .try_to_vec::<f32>()
                        .unwrap(),
                    crate::ops::into_data_sync(expected)
                        .try_to_vec::<f32>()
                        .unwrap(),
                );
            }
        }
    }
}
