use crate::{
    kernel::{
        into_contiguous_aligned,
        utils::{address_type, decompose_linear, shape_divmod},
    },
    ops::{
        max_vector_size_many, numeric::empty_device_dtype, permute_nchw_to_nhwc,
        permute_nhwc_to_nchw,
    },
    tensor::CubeTensor,
};
use burn_backend::cubecl::dtype_to_storage_type;
use burn_backend::{Shape, ops::ConvTransposeOptions};
use cubecl::{
    calculate_cube_count_elemwise,
    prelude::*,
    std::{FastDivmod, tensor::layout::linear::LinearViewMut},
};
use cubek::convolution::components::ConvSetupError;

/// The convolution's parameters, as the kernel is told them.
#[derive(CubeLaunch, CubeType)]
struct ConvArgs {
    conv_stride_0: usize,
    conv_stride_1: usize,
    dilation_0: usize,
    dilation_1: usize,
    padding_0: usize,
    padding_1: usize,
    groups: usize,
}

/// The `start..end` of input positions that reach `out_pos` along one axis: those `i` with
/// `i * stride + k * dilation == out_pos + padding` for some `0 <= k < kernel`. Already clipped
/// to `size`, so a caller only has to test `k`'s divisibility.
///
/// Both ends are exact. Deriving the end from the start and one fixed window width — as this
/// kernel's predecessor did — yields an empty range whenever `stride` exceeds the dilated
/// kernel extent, silently zeroing every output position that does have contributions.
#[cube]
fn contributing_range(
    out_pos: usize,
    padding: usize,
    kernel: usize,
    dilation: usize,
    stride: usize,
    size: usize,
) -> (usize, usize) {
    let numerator = out_pos + padding;
    let extent = (kernel - 1) * dilation;
    // `k <= kernel - 1` bounds `i` from below, `k >= 0` from above.
    let start = if numerator > extent {
        (numerator - extent).div_ceil(stride)
    } else {
        0usize.runtime()
    };
    (start, clamp_max(numerator / stride + 1, size))
}

/// Perform a 2D convolution transposition using the direct algorithm.
///
/// Takes and returns NCHW. The kernel itself is NHWC — that is the layout its reads coalesce in,
/// and the one the rest of the convolution stack already works in — so this permutes on the way
/// in and out. Both permutes are metadata-only; the contiguity they imply is materialized once,
/// inside [`conv_transpose2d_direct_nhwc`].
///
/// * `input` - The input feature map
/// * `weight` - The weights (filter) applied to each kernel
/// * `bias` - The bias added to each channel
/// * `options` - The options to use for the convolution
pub fn conv_transpose2d_direct(
    input: CubeTensor,
    weight: CubeTensor,
    bias: Option<CubeTensor>,
    options: ConvTransposeOptions<2>,
) -> Result<CubeTensor, ConvSetupError> {
    let out = conv_transpose2d_direct_nhwc(
        permute_nchw_to_nhwc(input),
        permute_nchw_to_nhwc(weight),
        bias,
        options,
    )?;
    Ok(permute_nhwc_to_nchw(out))
}

/// Direct transposed convolution over NHWC tensors.
///
/// Which axis a thread owns is the whole point. The output decomposes as
/// `[batch, out_y, out_x, oc_out]`, so neighbouring threads differ only in the output channel:
/// they read the *same* input element at each step, which the cache broadcasts, and their
/// weight reads are consecutive addresses — `oc_out` is an NHWC weight's contiguous axis — so
/// they coalesce. Under NCHW `out_x` varies fastest, so neighbouring threads read unrelated
/// input and the `in_c` reduction strides by a whole feature map: the same multiply-accumulates
/// over far more traffic.
///
/// Reducing over `in_c` innermost is what leaves both patterns intact, and it also lifts the
/// dilation test out of the reduction, since divisibility depends only on spatial position.
///
/// Reuse of the weight across *spatial* positions is left on the table; that needs shared-memory
/// tiling over a blocked launch, which is the shape of the implicit-GEMM candidates
/// `dgrad_autotune` already offers one level up.
#[cube(launch, address_type = "dynamic")]
fn conv_transpose2d_direct_nhwc_kernel<E: Numeric, N: Size>(
    input: &Tensor<E>,
    weight: &Tensor<Vector<E, N>>,
    bias: ComptimeOption<&[Vector<E, N>]>,
    mut output: LinearViewMut<'_, Vector<E, N>>,
    out_shape: Sequence<FastDivmod<usize>>,
    args: ConvArgs,
    #[define(E)] _dtype: ElemType,
) {
    if ABSOLUTE_POS >= output.shape() {
        terminate!();
    }

    // `out_shape` counts elements, so step into it by whole vectors. Only the channel dim is
    // affected, and `oc_out` lands on a multiple of `line`: the launcher picks a width dividing
    // `out_c_per_group`, which puts every group start `group * out_c_per_group` on a boundary
    // too, so a vector never straddles two groups.
    let line = weight.vector_size();
    let in_c_per_group = weight.shape(0) / args.groups;
    let kernel_h = weight.shape(1);
    let kernel_w = weight.shape(2);
    let out_c_per_group = weight.shape(3);

    let (_, pos) = decompose_linear(ABSOLUTE_POS * line, &out_shape);
    let [batch, out_y, out_x, oc_out] = *pos else {
        unreachable!()
    };

    let group = (oc_out / out_c_per_group) % args.groups;
    let in_c_start = group * in_c_per_group;
    let in_c_end = in_c_start + in_c_per_group;

    let (y_start, y_end) = contributing_range(
        out_y,
        args.padding_0,
        kernel_h,
        args.dilation_0,
        args.conv_stride_0,
        input.shape(1),
    );
    let (x_start, x_end) = contributing_range(
        out_x,
        args.padding_1,
        kernel_w,
        args.dilation_1,
        args.conv_stride_1,
        input.shape(2),
    );

    let numerator_h_base = out_y + args.padding_0;
    let numerator_w_base = out_x + args.padding_1;

    let idx_input_batch = batch * input.stride(0);
    let idx_weight_oc = oc_out - out_c_per_group * group;

    let bias: ComptimeOption<Vector<E, N>> = bias.as_ref().map(|bias| bias[oc_out / line]);
    let mut sum = bias.unwrap_or(Vector::broadcast(E::from_int(0)));

    for in_y in y_start..y_end {
        // The range already bounds `in_y * stride` by the numerator, so only `kernel_y`'s
        // divisibility is left to test.
        let numerator_h = numerator_h_base - in_y * args.conv_stride_0;

        if numerator_h.is_multiple_of(args.dilation_0) {
            let idx_y = in_y * input.stride(1);
            let idx_ky = (numerator_h / args.dilation_0) * weight.stride(1);

            for in_x in x_start..x_end {
                let numerator_w = numerator_w_base - in_x * args.conv_stride_1;

                if numerator_w.is_multiple_of(args.dilation_1) {
                    let idx_input = idx_input_batch + idx_y + in_x * input.stride(2);
                    let idx_weight =
                        idx_weight_oc + idx_ky + (numerator_w / args.dilation_1) * weight.stride(2);

                    for in_c in in_c_start..in_c_end {
                        // One input element feeds all `line` output channels: loaded once into
                        // a register, broadcast across the vector. The weight read beside it is
                        // `line` consecutive addresses in one instruction.
                        let value = input[idx_input + in_c * input.stride(3)];
                        sum += Vector::broadcast(value)
                            * weight[(idx_weight + in_c * weight.stride(0)) / line];
                    }
                }
            }
        }
    }

    output.write(ABSOLUTE_POS, sum);
}

/// Perform a 2D convolution transposition on NHWC tensors using the direct algorithm.
///
/// * `input` - The input feature map, `[batch, height, width, in_channels]`
/// * `weight` - The weights (filter) applied to each kernel,
///   `[in_channels, kernel_h, kernel_w, out_channels / groups]`
/// * `bias` - The bias added to each channel
/// * `options` - The options to use for the convolution
///
/// The output is `[batch, height, width, out_channels]`.
pub fn conv_transpose2d_direct_nhwc(
    mut input: CubeTensor,
    mut weight: CubeTensor,
    bias: Option<CubeTensor>,
    options: ConvTransposeOptions<2>,
) -> Result<CubeTensor, ConvSetupError> {
    // Both tensors arrive permuted from NCHW, so the channel axis is the one that is *not*
    // contiguous — exactly the axis this kernel wants dense. Materialize it the way
    // `conv_direct` does on the forward pass.
    //
    // For the weight this is correctness, not tuning: the kernel vectorizes that axis and
    // indexes it without applying `stride(3)`, which only holds at unit stride. For the input
    // it is a measured call — on the conv1d data gradient the copy costs less than the strided
    // reads it saves on most shapes, though the margin narrowed once the channel axis was
    // vectorized and the reads began coming from a register.
    if input.meta.strides()[3] != 1 {
        input = into_contiguous_aligned(input);
    }
    if weight.meta.strides()[3] != 1 {
        weight = into_contiguous_aligned(weight);
    }

    let [batch_size, in_height, in_width, _] = input.meta.shape().dims();
    let [_, kernel_0, kernel_1, out_c_per_group] = weight.meta.shape().dims();
    let out_channels = out_c_per_group * options.groups;

    let out_0 = (in_height - 1) * options.stride[0]
        + options.dilation[0] * (kernel_0 - 1)
        + options.padding_out[0]
        - 2 * options.padding[0]
        + 1;
    let out_1 = (in_width - 1) * options.stride[1]
        + options.dilation[1] * (kernel_1 - 1)
        + options.padding_out[1]
        - 2 * options.padding[1]
        + 1;

    let output = empty_device_dtype(
        input.client.clone(),
        input.device.clone(),
        Shape::new([batch_size, out_0, out_1, out_channels]),
        input.dtype,
    );

    // Vectorize along the output channels, the axis threads already run along: a thread then
    // owns `line` of them, so one input element feeds `line` multiply-accumulates from a single
    // register and the weights beside it arrive in one instruction.
    //
    // Asking `weight` keeps a vector from straddling a group boundary for free: its channel
    // axis *is* the per-group count, and a width has to divide the axis it vectorizes.
    let line = max_vector_size_many(&[&weight, &output], 3).min(match &bias {
        // Indexed by that same axis, so it has to agree — when there is one.
        Some(bias) => max_vector_size_many(&[bias], 0),
        None => VectorSize::MAX,
    });

    let working_units = output.meta.num_elements() / line;
    let cube_dim = CubeDim::new(&input.client, working_units);
    let cube_count = calculate_cube_count_elemwise(&input.client, working_units, cube_dim);
    let dtype = input.dtype;

    conv_transpose2d_direct_nhwc_kernel::launch(
        &output.client,
        cube_count,
        cube_dim,
        address_type!(input, weight, bias, output),
        line,
        input.into_tensor_arg(),
        weight.into_tensor_arg(),
        bias.map(|bias| bias.into_buffer_arg()).into(),
        output.clone().into_linear_view(),
        shape_divmod(&output),
        ConvArgsLaunch::new(
            options.stride[0],
            options.stride[1],
            options.dilation[0],
            options.dilation[1],
            options.padding[0],
            options.padding[1],
            options.groups,
        ),
        dtype_to_storage_type(dtype),
    );

    Ok(output)
}

/// The kernel against the definition of a transposed convolution, computed on the host.
#[cfg(all(
    test,
    any(feature = "wgpu", feature = "cpu", feature = "cuda", feature = "hip")
))]
mod tests {
    use burn_std::TensorData;

    use crate::{
        CubeDevice,
        ops::{from_data, into_data_sync},
    };

    use super::*;

    fn data(shape: &[usize], seed: u32) -> Vec<f32> {
        let n: usize = shape.iter().product();
        (0..n as u32)
            .map(|i| {
                ((i.wrapping_mul(2654435761).wrapping_add(seed) >> 8) % 97) as f32 / 97.0 - 0.5
            })
            .collect()
    }

    /// Scatter each input element into the output through the kernel, which is what a
    /// transposed convolution *is*. Deliberately unlike the kernel, which gathers.
    fn reference(
        input: &[f32],
        [n, cin, ih, iw]: [usize; 4],
        weight: &[f32],
        [_, cout_pg, kh, kw]: [usize; 4],
        bias: Option<&[f32]>,
        options: &ConvTransposeOptions<2>,
    ) -> (Vec<f32>, [usize; 4]) {
        let ConvTransposeOptions {
            stride,
            padding,
            padding_out,
            dilation,
            groups,
        } = options;
        let cin_pg = cin / groups;
        let cout = cout_pg * groups;
        let oh =
            (ih - 1) * stride[0] + dilation[0] * (kh - 1) + padding_out[0] - 2 * padding[0] + 1;
        let ow =
            (iw - 1) * stride[1] + dilation[1] * (kw - 1) + padding_out[1] - 2 * padding[1] + 1;

        let mut out = vec![0f32; n * cout * oh * ow];
        if let Some(bias) = bias {
            for (i, v) in out.iter_mut().enumerate() {
                *v = bias[(i / (oh * ow)) % cout];
            }
        }
        for b in 0..n {
            for ci in 0..cin {
                let g = ci / cin_pg;
                for y in 0..ih {
                    for x in 0..iw {
                        let v = input[((b * cin + ci) * ih + y) * iw + x];
                        for co in 0..cout_pg {
                            for a in 0..kh {
                                let yy = (y * stride[0] + a * dilation[0]) as isize
                                    - padding[0] as isize;
                                if yy < 0 || yy >= oh as isize {
                                    continue;
                                }
                                for c in 0..kw {
                                    let xx = (x * stride[1] + c * dilation[1]) as isize
                                        - padding[1] as isize;
                                    if xx < 0 || xx >= ow as isize {
                                        continue;
                                    }
                                    let co_abs = g * cout_pg + co;
                                    out[((b * cout + co_abs) * oh + yy as usize) * ow
                                        + xx as usize] +=
                                        v * weight[((ci * cout_pg + co) * kh + a) * kw + c];
                                }
                            }
                        }
                    }
                }
            }
        }
        (out, [n, cout, oh, ow])
    }

    #[track_caller]
    fn assert_matches_definition(
        case: &str,
        input_shape: [usize; 4],
        weight_shape: [usize; 4],
        with_bias: bool,
        options: ConvTransposeOptions<2>,
    ) {
        let device = CubeDevice::default();
        let (input, weight) = (data(&input_shape, 1), data(&weight_shape, 2));
        let bias_shape = [weight_shape[1] * options.groups];
        let bias = with_bias.then(|| data(&bias_shape, 3));

        let (expected, expected_shape) = reference(
            &input,
            input_shape,
            &weight,
            weight_shape,
            bias.as_deref(),
            &options,
        );

        let tensor = |shape: &[usize], values: &[f32]| {
            from_data(TensorData::new(values.to_vec(), shape.to_vec()), &device)
        };
        let actual = conv_transpose2d_direct(
            tensor(&input_shape, &input),
            tensor(&weight_shape, &weight),
            bias.as_ref().map(|b| tensor(&bias_shape, b)),
            options,
        )
        .expect("the kernel accepts these shapes");

        assert_eq!(
            actual.meta.shape().dims::<4>(),
            expected_shape,
            "{case}: output shapes differ"
        );

        let actual = into_data_sync(actual);
        for (i, (a, e)) in actual
            .as_slice::<f32>()
            .unwrap()
            .iter()
            .zip(&expected)
            .enumerate()
        {
            assert!((a - e).abs() < 1e-4, "{case}: at {i}, got {a}, want {e}");
        }
    }

    #[test]
    fn matches_the_definition() {
        let opts = ConvTransposeOptions::new;
        let cases = [
            (
                "plain",
                [6, 4, 3, 3],
                false,
                opts([1, 1], [0, 0], [0, 0], [1, 1], 1),
            ),
            (
                "bias",
                [6, 4, 3, 3],
                true,
                opts([1, 1], [0, 0], [0, 0], [1, 1], 1),
            ),
            (
                "strided",
                [6, 4, 3, 3],
                false,
                opts([2, 2], [0, 0], [0, 0], [1, 1], 1),
            ),
            (
                "padded",
                [6, 4, 3, 3],
                false,
                opts([2, 2], [1, 1], [1, 1], [1, 1], 1),
            ),
            (
                "dilated",
                [6, 4, 3, 3],
                false,
                opts([1, 1], [1, 1], [0, 0], [3, 3], 1),
            ),
            // Groups slice the channel reduction, the axis this kernel reorders.
            (
                "grouped",
                [6, 4, 3, 3],
                true,
                opts([1, 1], [0, 0], [0, 0], [1, 1], 3),
            ),
            (
                "depthwise",
                [6, 1, 3, 3],
                false,
                opts([1, 1], [1, 1], [0, 0], [1, 1], 6),
            ),
            (
                "pointwise",
                [6, 4, 1, 1],
                false,
                opts([1, 1], [0, 0], [0, 0], [1, 1], 1),
            ),
            // Sizes and options differing per axis: a height-for-width mix-up still agrees on a
            // square problem, and disagrees here.
            (
                "asymmetric",
                [6, 4, 2, 5],
                true,
                opts([3, 2], [2, 1], [1, 0], [1, 2], 1),
            ),
            // Stride past the dilated kernel extent, where no input position reaches some
            // outputs and every window bound has to be exact.
            (
                "sparse",
                [6, 4, 2, 2],
                false,
                opts([4, 3], [0, 0], [0, 0], [1, 1], 1),
            ),
            // Widths where the vector divides the per-group count but the *absolute* channel
            // index of a group start does not sit on a vector boundary unless groups agree.
            (
                "grouped wide channels",
                [8, 4, 3, 3],
                true,
                opts([1, 1], [1, 1], [0, 0], [1, 1], 2),
            ),
            (
                "grouped four ways",
                [8, 2, 3, 3],
                false,
                opts([2, 2], [1, 1], [1, 1], [1, 1], 4),
            ),
            // A single output channel per group: vectorization must fall back to width 1.
            (
                "one channel per group",
                [8, 1, 3, 3],
                true,
                opts([1, 1], [0, 0], [0, 0], [1, 1], 8),
            ),
            // Channel counts that are not multiples of a vector width: 6 in, 3 out.
            (
                "ragged channels",
                [6, 3, 3, 3],
                true,
                opts([1, 1], [1, 1], [0, 0], [1, 1], 1),
            ),
        ];

        for (case, weight, bias, options) in cases {
            // Weight dim 0 is the input channel count, so the input follows it.
            assert_matches_definition(case, [2, weight[0], 5, 7], weight, bias, options);
        }
    }

    /// The shape a 1D data gradient reshapes to: the unit axis is no longer the fastest-varying
    /// one under NHWC.
    #[test]
    fn matches_the_definition_with_a_unit_axis() {
        assert_matches_definition(
            "unit width",
            [2, 6, 9, 1],
            [6, 4, 5, 1],
            false,
            ConvTransposeOptions::new([1, 1], [4, 0], [0, 0], [1, 1], 1),
        );
    }
}
