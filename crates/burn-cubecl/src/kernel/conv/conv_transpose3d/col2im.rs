use crate::{
    kernel::{
        conv::{batches_per_run, index},
        into_contiguous_aligned,
        matmul::{MatmulStrategy, matmul},
        utils::{address_type, decompose_linear, shape_divmod},
    },
    ops::{numeric::empty_device_dtype, reshape, swap_dims},
    tensor::CubeTensor,
};
use burn_backend::cubecl::dtype_to_storage_type;
use burn_backend::{
    Shape,
    ops::{ConvTransposeOptions, conv::calculate_conv_transpose_output_size},
};
use cubecl::{
    calculate_cube_count_elemwise,
    prelude::*,
    std::{FastDivmod, tensor::layout::linear::LinearViewMut},
};
use cubek::convolution::components::ConvSetupError;

/// Perform a 3D convolution transposition using the GEMM (col2im) algorithm.
///
/// The transposition is one matrix multiplication of the weight against the input, laid out as
/// columns, followed by a gather that sums each output voxel's contributions back from them. It
/// is the 2D algorithm with a depth axis.
///
/// * `input` - The input feature map
/// * `weight` - The weights (filter) applied to each kernel
/// * `bias` - The bias added to each channel
/// * `options` - The options to use for the convolution
pub fn conv_transpose3d_col2im(
    input: CubeTensor,
    weight: CubeTensor,
    bias: Option<CubeTensor>,
    options: ConvTransposeOptions<3>,
) -> Result<CubeTensor, ConvSetupError> {
    let [
        input_channels,
        im_ch_per_group,
        kernel_d,
        kernel_h,
        kernel_w,
    ] = weight.meta.shape().dims();
    let [batch_size, _, input_d, input_h, input_w] = input.meta.shape().dims();
    let groups = options.groups;
    let input_ch_per_group = input_channels / groups;
    let ConvTransposeOptions {
        padding: [padding_d, padding_h, padding_w],
        padding_out: [padding_out_d, padding_out_h, padding_out_w],
        dilation: [dilation_d, dilation_h, dilation_w],
        stride: [stride_d, stride_h, stride_w],
        ..
    } = options.clone();

    let im_d = calculate_conv_transpose_output_size(
        kernel_d,
        stride_d,
        padding_d,
        padding_out_d,
        dilation_d,
        input_d,
    );
    let im_h = calculate_conv_transpose_output_size(
        kernel_h,
        stride_h,
        padding_h,
        padding_out_h,
        dilation_h,
        input_h,
    );
    let im_w = calculate_conv_transpose_output_size(
        kernel_w,
        stride_w,
        padding_w,
        padding_out_w,
        dilation_w,
        input_w,
    );
    let im_channels = im_ch_per_group * groups;
    let kernel = [kernel_d, kernel_h, kernel_w];

    let batches_per_run = batches_per_run(
        batch_size,
        input_d * input_h * input_w,
        input.client.properties().hardware.plane_size_max as usize,
    )?;
    let col_shape_0 = im_ch_per_group * kernel_d * kernel_h * kernel_w;

    let weight = reshape(
        weight.clone(),
        Shape::new([groups, input_ch_per_group, col_shape_0]),
    );
    let weight = into_contiguous_aligned(swap_dims(weight, 1, 2));

    if batches_per_run != batch_size {
        let runs = batch_size / batches_per_run;

        let im_shape = Shape::new([runs, batches_per_run, im_channels, im_d, im_h, im_w]);
        let image = empty_device_dtype(
            input.client.clone(),
            input.device.clone(),
            im_shape,
            input.dtype,
        );

        let input_shape = Shape::new([
            runs,
            batches_per_run,
            input_channels,
            input_d,
            input_h,
            input_w,
        ]);
        let input = reshape(input, input_shape);
        let input_shape_run =
            Shape::new([batches_per_run, input_channels, input_d, input_h, input_w]);

        for run in 0..runs {
            let input = index(input.clone(), run);
            let input = reshape(input, input_shape_run.clone());
            let im_shape = Shape::new([batches_per_run, im_channels, im_d, im_h, im_w]);
            let image_slice = index(image.clone(), run);
            let image_slice = reshape(image_slice, im_shape);
            execute(
                input,
                weight.clone(),
                bias.clone(),
                image_slice,
                options.clone(),
                kernel,
            )?;
        }
        Ok(reshape(
            image,
            Shape::new([batch_size, im_channels, im_d, im_h, im_w]),
        ))
    } else {
        let im_shape = Shape::new([batches_per_run, im_channels, im_d, im_h, im_w]);
        let image = empty_device_dtype(
            input.client.clone(),
            input.device.clone(),
            im_shape,
            input.dtype,
        );
        execute(input, weight, bias, image.clone(), options, kernel)?;
        Ok(image)
    }
}

fn execute(
    input: CubeTensor,
    weight: CubeTensor,
    bias: Option<CubeTensor>,
    image: CubeTensor,
    options: ConvTransposeOptions<3>,
    kernel: [usize; 3],
) -> Result<(), ConvSetupError> {
    let [batch_size, _, input_d, input_h, input_w] = input.meta.shape().dims();
    let [groups, col_shape_0, input_ch_per_group] = weight.meta.shape().dims();

    let col_shape_1 = batch_size * input_d * input_h * input_w;

    let input = swap_dims(input, 0, 1);
    let input_shape = Shape::new([groups, input_ch_per_group, col_shape_1]);
    let input = reshape(input, input_shape);

    let dtype = input.dtype;
    let columns = matmul(weight, input, None, MatmulStrategy::default(), dtype)?;
    let columns = reshape(columns, Shape::new([col_shape_0 * groups, col_shape_1]));

    col2im(
        columns,
        bias,
        image,
        kernel,
        [input_d, input_h, input_w],
        options,
    )?;

    Ok(())
}

fn col2im(
    columns: CubeTensor,
    bias: Option<CubeTensor>,
    out: CubeTensor,
    kernel: [usize; 3],
    grid: [usize; 3],
    options: ConvTransposeOptions<3>,
) -> Result<(), LaunchError> {
    let dtype = columns.dtype;

    let columns = into_contiguous_aligned(columns);
    let bias = bias.map(into_contiguous_aligned);

    let num_elems = out.meta.num_elements();

    let cube_dim = CubeDim::new(&columns.client, num_elems);
    let cube_count = calculate_cube_count_elemwise(&columns.client, num_elems, cube_dim);

    let shape = shape_divmod(&out);
    unsafe {
        col2im_3d_kernel::launch_unchecked(
            &columns.client.clone(),
            cube_count,
            cube_dim,
            address_type!(columns, bias, out),
            columns.into_tensor_arg(),
            bias.map(|bias| bias.into_buffer_arg()).into(),
            out.into_linear_view(),
            shape,
            Col2Im3dArgsLaunch::new(
                grid[0],
                grid[1],
                grid[2],
                kernel[0],
                kernel[1],
                kernel[2],
                options.padding[0],
                options.padding[1],
                options.padding[2],
                options.dilation[0],
                options.dilation[1],
                options.dilation[2],
                options.stride[0],
                options.stride[1],
                options.stride[2],
            ),
            dtype_to_storage_type(dtype),
        )
    };

    Ok(())
}

/// The columns grid is the transposition's input grid, so `out_*` is the input extent.
#[derive(CubeLaunch, CubeType)]
struct Col2Im3dArgs {
    out_d: usize,
    out_h: usize,
    out_w: usize,

    kernel_d: usize,
    kernel_h: usize,
    kernel_w: usize,

    pad_d: usize,
    pad_h: usize,
    pad_w: usize,
    dilation_d: usize,
    dilation_h: usize,
    dilation_w: usize,
    stride_d: usize,
    stride_h: usize,
    stride_w: usize,
}

#[cube(launch_unchecked, address_type = "dynamic")]
fn col2im_3d_kernel<E: Numeric>(
    columns: &Tensor<E>,
    bias: ComptimeOption<&[E]>,
    mut image: LinearViewMut<'_, E>,
    image_shape: Sequence<FastDivmod<usize>>,
    args: &Col2Im3dArgs,
    #[define(E)] _dtype: ElemType,
) {
    if ABSOLUTE_POS >= image.shape() {
        terminate!();
    }

    let (_, pos) = decompose_linear(ABSOLUTE_POS, &image_shape);
    let [batch, ch_im, im_z, im_y, im_x] = *pos else {
        unreachable!()
    };

    let im_x = im_x + args.pad_w;
    let im_y = im_y + args.pad_h;
    let im_z = im_z + args.pad_d;

    let kernel_extent_w = (args.kernel_w - 1) * args.dilation_w + 1;
    let kernel_extent_h = (args.kernel_h - 1) * args.dilation_h + 1;
    let kernel_extent_d = (args.kernel_d - 1) * args.dilation_d + 1;

    let mut val = E::zero();

    let x_col_start = if im_x >= kernel_extent_w {
        (im_x - kernel_extent_w) / args.stride_w + 1
    } else {
        0usize.runtime()
    };
    let x_col_end = clamp_max(im_x / args.stride_w + 1, args.out_w);
    let y_col_start = if im_y >= kernel_extent_h {
        (im_y - kernel_extent_h) / args.stride_h + 1
    } else {
        0usize.runtime()
    };
    let y_col_end = clamp_max(im_y / args.stride_h + 1, args.out_h);
    let z_col_start = if im_z >= kernel_extent_d {
        (im_z - kernel_extent_d) / args.stride_d + 1
    } else {
        0usize.runtime()
    };
    let z_col_end = clamp_max(im_z / args.stride_d + 1, args.out_d);

    for col_z in z_col_start..z_col_end {
        let kernel_z = im_z - col_z * args.stride_d;
        for col_y in y_col_start..y_col_end {
            let kernel_y = im_y - col_y * args.stride_h;
            for col_x in x_col_start..x_col_end {
                let kernel_x = im_x - col_x * args.stride_w;

                if kernel_z.is_multiple_of(args.dilation_d)
                    && kernel_y.is_multiple_of(args.dilation_h)
                    && kernel_x.is_multiple_of(args.dilation_w)
                {
                    let kernel_z = kernel_z / args.dilation_d;
                    let kernel_y = kernel_y / args.dilation_h;
                    let kernel_x = kernel_x / args.dilation_w;

                    let col_k = ((ch_im * args.kernel_d + kernel_z) * args.kernel_h + kernel_y)
                        * args.kernel_w
                        + kernel_x;
                    let col_n =
                        ((batch * args.out_d + col_z) * args.out_h + col_y) * args.out_w + col_x;
                    let col_pos = col_k * columns.stride(0) + col_n * columns.stride(1);
                    val += columns[col_pos];
                }
            }
        }
    }

    #[comptime]
    match bias {
        ComptimeOption::Some(bias) => image.write(ABSOLUTE_POS, val + bias[ch_im]),
        ComptimeOption::None => image.write(ABSOLUTE_POS, val),
    }
}
