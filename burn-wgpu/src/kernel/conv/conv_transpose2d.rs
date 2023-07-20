use crate::{
    element::WgpuElement,
    kernel::{self, build_info_no_dim, elemwise_workgroup, KernelSettings},
    kernel_wgsl,
    tensor::WgpuTensor,
};
use burn_tensor::{ops::ConvTransposeOptions, Shape};

kernel_wgsl!(ConvTranspose2d, "../../template/conv/conv_transpose2d.wgsl");

pub(crate) fn conv_transpose2d<E: WgpuElement>(
    input: WgpuTensor<E, 4>,
    weight: WgpuTensor<E, 4>,
    bias: Option<WgpuTensor<E, 1>>,
    options: ConvTransposeOptions<2>,
) -> WgpuTensor<E, 4> {
    const WORKGROUP: usize = 32;

    let input = kernel::into_continuous(input);
    let weight = kernel::into_continuous(weight);
    let [batch_size, _, in_height, in_width] = input.shape.dims;
    let [_, out_channels, kernel_0, kernel_1] = weight.shape.dims;

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

    let shape_out = Shape::new([batch_size, out_channels * options.groups, out_0, out_1]);
    let num_elems = shape_out.num_elements();

    let buffer = input
        .context
        .create_buffer(num_elems * core::mem::size_of::<E>());
    let output = WgpuTensor::new(input.context.clone(), shape_out, buffer);

    let mut info = build_info_no_dim(&[&input, &output, &weight]);
    info.push(options.stride[0] as u32);
    info.push(options.stride[1] as u32);
    info.push(options.padding[0] as u32);
    info.push(options.padding[1] as u32);
    info.push(options.dilation[0] as u32);
    info.push(options.dilation[1] as u32);
    info.push(options.groups as u32);

    let bias_buffer = bias
        .map(|bias| bias.buffer)
        .unwrap_or_else(|| input.context.create_buffer(core::mem::size_of::<E>()));

    for id in 0..num_elems {
        let id = id as u32;
        let bias = input.context.read_buffer(bias_buffer.clone());
        let weight = input.context.read_buffer(weight.buffer.clone());
        let input = input.context.read_buffer(input.buffer.clone());
        let bias: &[f32] = bytemuck::cast_slice(&bias);
        let weight: &[f32] = bytemuck::cast_slice(&weight);
        let input: &[f32] = bytemuck::cast_slice(&input);

        let input_stride_0 = info[0];
        let input_stride_1 = info[1];
        let input_stride_2 = info[2];
        let input_stride_3 = info[3];
        let output_stride_0 = info[4];
        let output_stride_1 = info[5];
        let output_stride_2 = info[6];
        let output_stride_3 = info[7];
        let weight_stride_0 = info[8];
        let weight_stride_1 = info[9];
        let weight_stride_2 = info[10];
        let weight_stride_3 = info[11];

        let input_shape_0 = info[12];
        let input_shape_1 = info[13];
        let input_shape_2 = info[14];
        let input_shape_3 = info[15];
        let output_shape_0 = info[16];
        let output_shape_1 = info[17];
        let output_shape_2 = info[18];
        let output_shape_3 = info[19];
        let weight_shape_0 = info[20];
        let weight_shape_1 = info[21];
        let weight_shape_2 = info[22];
        let weight_shape_3 = info[23];

        let stride_0 = info[24];
        let stride_1 = info[25];
        let padding_0 = info[26];
        let padding_1 = info[27];
        let dilation_0 = info[28];
        let dilation_1 = info[29];
        let groups = info[30];

        let in_channels = weight_shape_0;
        let kernel_size_0 = weight_shape_2;
        let kernel_size_1 = weight_shape_3;

        let b = id / output_stride_0 % output_shape_0;
        let oc_out = id / output_stride_1 % output_shape_1;
        let oh = id / output_stride_2 % output_shape_2;
        let ow = id / output_stride_3 % output_shape_3;

        let k = oc_out / weight_shape_1;
        let g = k % groups;
        println!("G {g}/{groups} OC Out {oc_out} K {k}");
        let oc = oc_out - (weight_shape_1 * g);

        let mut sum = bias[0];

        let ic_start = g * (in_channels / groups);
        let ic_end = ic_start + in_channels / groups;

        let ih_start = 0;
        let ih_end = input_shape_2;

        let iw_start = 0;
        let iw_end = input_shape_3;

        for ic in ic_start..ic_end {
            for ih in ih_start..ih_end {
                for iw in iw_start..iw_end {
                    for kh in 0..kernel_size_0 {
                        for kw in 0..kernel_size_1 {
                            let oh_tmp = ih * stride_0 + kh * dilation_0;
                            let ow_tmp = iw * stride_1 + kw * dilation_1;

                            if oh_tmp >= padding_0 && ow_tmp >= padding_1 {
                                let oh_tmp_pad = oh_tmp - padding_0;
                                let ow_tmp_pad = ow_tmp - padding_1;

                                if oh_tmp_pad == oh && ow_tmp_pad == ow {
                                    // println!("Sum += ic {ic} oc {oc} ih {ih} iw {iw} oh {oh} ow {ow}");
                                    let index_input = b * input_stride_0
                                        + ic * input_stride_1
                                        + ih * input_stride_2
                                        + iw * input_stride_3;
                                    let index_weight = ic * weight_stride_0
                                        + oc * weight_stride_1
                                        + kh * weight_stride_2
                                        + kw * weight_stride_3;

                                    sum +=
                                        input[index_input as usize] * weight[index_weight as usize];
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    let info_buffer = input
        .context
        .create_buffer_with_data(bytemuck::cast_slice(&info));

    let kernel = input
        .context
        .compile_static::<KernelSettings<ConvTranspose2d, E, i32, WORKGROUP, WORKGROUP, 1>>();

    input.context.execute(
        elemwise_workgroup(output.shape.num_elements(), WORKGROUP),
        kernel,
        &[
            &input.buffer,
            &weight.buffer,
            &bias_buffer,
            &output.buffer,
            &info_buffer,
        ],
    );

    output
}

#[cfg(test)]
mod tests {
    use crate::tests::{ReferenceBackend, TestBackend};
    use burn_tensor::{module, Distribution, Tensor};

    // #[test]
    // fn conv_transpose2d_should_work_with_multiple_invocations() {
    //     let input = Tensor::<TestBackend, 4>::random([6, 16, 32, 32], Distribution::Default);
    //     let weight = Tensor::<TestBackend, 4>::random([12, 8, 3, 3], Distribution::Default);
    //     let bias = Tensor::<TestBackend, 1>::random([12], Distribution::Default);
    //     let input_ref = Tensor::<ReferenceBackend, 4>::from_data(input.to_data());
    //     let weight_ref = Tensor::<ReferenceBackend, 4>::from_data(weight.to_data());
    //     let bias_ref = Tensor::<ReferenceBackend, 1>::from_data(bias.to_data());
    //     let options =
    //         burn_tensor::ops::ConvTransposeOptions::new([2, 3], [2, 3], [2, 3], [2, 3], 2);

    //     let output = module::conv_transpose2d(input, weight, Some(bias), options.clone());
    //     let output_ref = module::conv_transpose2d(input_ref, weight_ref, Some(bias_ref), options);

    //     output
    //         .into_data()
    //         .assert_approx_eq(&output_ref.into_data(), 3);
    // }
}
