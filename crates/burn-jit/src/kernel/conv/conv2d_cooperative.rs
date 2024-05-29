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

    let kh = CUBE_DIM_X;
    let kw = CUBE_DIM_Y;
    let oh = CUBE_POS_X;
    let ow = CUBE_POS_Y;
    let ih = kh + oh;
    let iw = kw + ow;
    let b = CUBE_POS / output.stride(0) % output.shape(0);
    let oc = CUBE_POS / output.stride(1) % output.shape(1);
    let ic = CUBE_POS_Z;
    let index_input_0 = b * input.stride(0);
    let index_weight_0 = oc * weight.stride(0);

    let input_index =
        index_input_0 + ic * input.stride(1) + ih * input.stride(2) + iw * input.stride(3);

    let weight_index =
        index_weight_0 + ic * weight.stride(1) + kh * weight.stride(2) + kw * weight.stride(3);

    // let x = weight[weight_index];
    // let y = input[input_index];
    shared_memory[UNIT_POS] = input[input_index] * weight[weight_index];

    sync_units();

    // Writing (naive version)
    if UNIT_POS == UInt::new(0) {
        let mut sum = bias[0];

        for i in range(0u32, CUBE_DIM, Comptime::new(false)) {
            sum += shared_memory[i];
        }

        // let x = input_index;
        // output[CUBE_POS] = F::cast_from(x);
        output[CUBE_POS] = sum;
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
        x: out_0 as u32,
        y: out_1 as u32,
        z: (batch_size * out_channels) as u32,
    };

    let settings = CompilationSettings::default().workgroup_size(cube_dim);

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
