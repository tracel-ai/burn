use burn_compute::client::ComputeClient;
use burn_cube::{branch::*, dialect::ComputeShader, LaunchArg, *};
use burn_tensor::{
    ops::{conv::calculate_conv_output_size, ConvOptions},
    Shape,
};

use crate::{
    kernel::into_contiguous,
    ops::{
        numeric::{empty_device, zeros_device},
        reshape,
    },
    tensor::JitTensor,
    FloatElement, JitRuntime,
};

#[cube(launch)]
fn kernel<F: Float>(
    input: Tensor<F>,
    weight: Tensor<F>,
    bias: Tensor<F>,
    mut output: Tensor<F>,
    conv_stride_0: UInt,
    conv_stride_1: UInt,
    dilation_0: UInt,
    dilation_1: UInt,
    padding_0: UInt,
    padding_1: UInt,
    groups: UInt,
    kernel_size_0_unroll: Comptime<Option<UInt>>,
    kernel_size_1_unroll: Comptime<Option<UInt>>,
) {
    if AbsoluteIndex::get() >= Tensor::<F>::len(output) {
        return;
    }

    let in_channels = Tensor::<F>::shape(weight, 1u32);

    let kernel_size_0 =
        Comptime::unwrap_or_else(kernel_size_0_unroll, || Tensor::<F>::shape(weight, 2u32));
    let unroll_0 = Comptime::is_some(kernel_size_0_unroll);
    let kernel_size_1 =
        Comptime::unwrap_or_else(kernel_size_1_unroll, || Tensor::<F>::shape(weight, 3u32));
    let unroll_1 = Comptime::is_some(kernel_size_1_unroll);

    let b =
        AbsoluteIndex::get() / Tensor::<F>::stride(output, 0u32) % Tensor::<F>::shape(output, 0u32);
    let oc =
        AbsoluteIndex::get() / Tensor::<F>::stride(output, 1u32) % Tensor::<F>::shape(output, 1u32);
    let oh =
        AbsoluteIndex::get() / Tensor::<F>::stride(output, 2u32) % Tensor::<F>::shape(output, 2u32);
    let ow =
        AbsoluteIndex::get() / Tensor::<F>::stride(output, 3u32) % Tensor::<F>::shape(output, 3u32);

    let g = (Tensor::<F>::shape(weight, 0u32) + oc) % groups;
    let ic_start = in_channels * g;
    let ic_end = ic_start + in_channels;
    let mut sum = bias[oc];

    let ih_base = oh * conv_stride_0;
    let iw_base = ow * conv_stride_1;

    let weight_stride_1 = Tensor::<F>::stride(weight, 1u32);
    let weight_stride_2 = Tensor::<F>::stride(weight, 2u32);
    let weight_stride_3 = Tensor::<F>::stride(weight, 3u32);

    let input_stride_1 = Tensor::<F>::stride(input, 1u32);
    let input_stride_2 = Tensor::<F>::stride(input, 2u32);
    let input_stride_3 = Tensor::<F>::stride(input, 3u32);
    let input_shape_2 = Tensor::<F>::shape(input, 2u32);
    let input_shape_3 = Tensor::<F>::shape(input, 3u32);

    let border_top = padding_0;
    let border_left = padding_1;
    let border_bottom = input_shape_2 + padding_0;
    let border_right = input_shape_3 + padding_1;

    let index_input_0 = b * Tensor::<F>::stride(input, 0u32);
    let index_weight_0 = oc * Tensor::<F>::stride(weight, 0u32);

    for ic in range(ic_start, ic_end, Comptime::new(false)) {
        let index_input_1 = ic * input_stride_1;
        let index_weight_1 = (ic - ic_start) * weight_stride_1;

        for kh in range(0u32, kernel_size_0, unroll_0) {
            for kw in range(0u32, kernel_size_1, unroll_1) {
                let ih = kh * dilation_0 + ih_base;
                let iw = kw * dilation_1 + iw_base;

                let within_padding = ih >= border_top
                    && ih < border_bottom
                    && iw >= border_left
                    && iw < border_right;

                if within_padding {
                    let ih_pad = ih - padding_0;
                    let iw_pad = iw - padding_1;

                    let index_input = index_input_0
                        + index_input_1
                        + ih_pad * input_stride_2
                        + iw_pad * input_stride_3;

                    let index_weight = index_weight_0
                        + index_weight_1
                        + kh * weight_stride_2
                        + kw * weight_stride_3;

                    sum += input[index_input] * weight[index_weight];
                }
            }
        }
    }

    output[AbsoluteIndex::get()] = sum;
}

pub(crate) fn conv2d<R: JitRuntime, E: FloatElement>(
    input: JitTensor<R, E, 4>,
    weight: JitTensor<R, E, 4>,
    bias: Option<JitTensor<R, E, 1>>,
    options: ConvOptions<2>,
) -> JitTensor<R, E, 4> {
    let input = into_contiguous(input);
    let weight = into_contiguous(weight);
    let [batch_size, _, in_height, in_width] = input.shape.dims;
    let [out_channels, _, kernel_0, kernel_1] = weight.shape.dims;

    let out_0 = calculate_conv_output_size(
        kernel_0,
        options.stride[0],
        options.padding[0],
        options.dilation[0],
        in_height,
    );
    let out_1 = calculate_conv_output_size(
        kernel_1,
        options.stride[1],
        options.padding[1],
        options.dilation[1],
        in_width,
    );

    let shape_out = Shape::new([batch_size, out_channels, out_0, out_1]);

    let output = empty_device(
        input.client.clone(),
        input.device.clone(),
        shape_out.clone(),
    );

    let bias = match bias {
        Some(bias) => {
            let shape = Shape::from([bias.shape.dims[0], 1, 1, 1]);
            reshape(bias, shape)
        }
        None => {
            let shape = Shape::from([output.shape.dims[0], 1, 1, 1]);
            zeros_device(input.client.clone(), input.device.clone(), shape)
        }
    };

    let num_elems_output = output.shape.num_elements();
    let workgroup = elemwise_workgroup(num_elems_output, WORKGROUP_DEFAULT);
    let settings = CompilationSettings::default();

    kernel_launch::<E::CubeElement, R>(
        input.client,
        workgroup,
        settings,
        TensorHandle::new(&input.handle, &input.strides, &input.shape.dims),
        TensorHandle::new(&weight.handle, &weight.strides, &weight.shape.dims),
        TensorHandle::new(&bias.handle, &bias.strides, &bias.shape.dims),
        TensorHandle::new(&output.handle, &output.strides, &output.shape.dims),
        options.stride[0] as u32,
        options.stride[1] as u32,
        options.dilation[0] as u32,
        options.dilation[1] as u32,
        options.padding[0] as u32,
        options.padding[1] as u32,
        options.groups as u32,
        Some(kernel_0.into()),
        Some(kernel_1.into()),
    );

    output
}
