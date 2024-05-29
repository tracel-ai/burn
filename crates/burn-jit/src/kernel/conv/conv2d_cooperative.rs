use burn_compute::client::ComputeClient;
use burn_cube::{branch::*, dialect::ComputeShader, LaunchArg, *};
use burn_tensor::{
    ops::{conv::calculate_conv_output_size, ConvOptions},
    Shape,
};
use dialect::WorkgroupSize;
use synchronization::*;

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
    // _conv_stride_0: UInt,
    // _conv_stride_1: UInt,
    // _dilation_0: UInt,
    // _dilation_1: UInt,
    // _padding_0: UInt,
    // _padding_1: UInt,
    // _groups: UInt,
    cube_dim: Comptime<u32>,
) {
    // We assume number of units is exact

    // Reading
    let mut shared_memory = SharedMemory::<F>::new(cube_dim);

    // Add kernel / 2 for offsetting and reach output pos, but subtract kernel / 2 for starting left corner of kernel
    let input_index = (CUBE_POS_X + UNIT_POS_X) * input.stride(2)
        + (CUBE_POS_Y + UNIT_POS_Y) * input.stride(3)
        + (CUBE_POS_Z + UNIT_POS_Z) * (input.stride(0) + input.stride(1));

    let weight_index = UNIT_POS_X * weight.stride(2)
        + UNIT_POS_Y * weight.stride(3)
        + UNIT_POS_Z * (weight.stride(0) + weight.stride(1));

    shared_memory[UNIT_POS] = input[input_index] * weight[weight_index];

    sync_units();

    // Writing (naive version)
    if UNIT_POS == UInt::new(0) {
        let cube_pos =
            CUBE_DIM_X * (CUBE_COUNT_X * CUBE_COUNT_Y) + CUBE_DIM_Y * CUBE_COUNT_Y + CUBE_DIM_Z;

        let mut sum = bias[cube_pos / output.stride(1) % output.shape(1)];

        for i in range(0u32, cube_dim, Comptime::new(false)) {
            sum += input[i];
        }

        output[block_pos] = sum;
    }
}

pub(crate) fn conv2d<R: JitRuntime, E: FloatElement>(
    input: JitTensor<R, E, 4>,
    weight: JitTensor<R, E, 4>,
    bias: Option<JitTensor<R, E, 1>>,
    options: ConvOptions<2>,
) -> JitTensor<R, E, 4> {
    let input = into_contiguous(input);
    let weight = into_contiguous(weight);
    let [batch_size, in_channels, in_height, in_width] = input.shape.dims;
    let [out_channels, _weight_in_channels, kernel_0, kernel_1] = weight.shape.dims;

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

    // Maybe better to round its total product up to a power of 2 and have idle threads
    let cube_dim = WorkgroupSize {
        x: kernel_0 as u32,
        y: kernel_1 as u32,
        z: in_channels as u32,
    };

    let block_count = WorkGroup {
        x: in_height as u32,
        y: in_width as u32,
        z: (batch_size * out_channels) as u32,
    };

    let settings = CompilationSettings::default()
        .workgroup_size(cube_dim)
        .vectorize_input(0, 1)
        .vectorize_output(0, 1);

    kernel_launch::<E::CubeElement, R>(
        input.client,
        block_count,
        settings,
        TensorHandle::new(&input.handle, &input.strides, &input.shape.dims),
        TensorHandle::new(&weight.handle, &weight.strides, &weight.shape.dims),
        TensorHandle::new(&bias.handle, &bias.strides, &bias.shape.dims),
        TensorHandle::new(&output.handle, &output.strides, &output.shape.dims),
        // options.stride[0] as u32,
        // options.stride[1] as u32,
        // options.dilation[0] as u32,
        // options.dilation[1] as u32,
        // options.padding[0] as u32,
        // options.padding[1] as u32,
        // options.groups as u32,
        cube_dim.x * cube_dim.y * cube_dim.z,
    );

    output
}
