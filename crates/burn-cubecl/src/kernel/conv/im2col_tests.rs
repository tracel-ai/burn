use super::im2col::conv_im2col;
use crate::{CubeDevice, ops::into_data_sync, tensor::CubeTensor};
use burn_backend::{
    DType,
    ops::{ConvOptions, conv::calculate_conv_output_sizes},
};
use burn_std::{Metadata, Shape};
use cubecl::prelude::CubeElement;

fn upload(device: &CubeDevice, shape: &[usize], values: &[f32], permuted: bool) -> CubeTensor {
    let rank = shape.len();
    let mut order = (0..rank).collect::<Vec<_>>();
    if permuted {
        order.swap(1, rank - 1);
    }
    let mut strides = vec![0; rank];
    let mut size = 1;
    for &axis in order.iter().rev() {
        strides[axis] = size;
        size *= shape[axis];
    }
    let mut data = vec![0.0; size];
    for (i, &value) in values.iter().enumerate() {
        let mut rem = i;
        let mut offset = 0;
        for axis in (0..rank).rev() {
            offset += (rem % shape[axis]) * strides[axis];
            rem /= shape[axis];
        }
        data[offset] = value;
    }
    let client = device.client();
    CubeTensor::new(
        client.clone(),
        client.create_from_slice(f32::as_bytes(&data)),
        Metadata::new(Shape::from(shape), strides),
        device.clone(),
        DType::F32,
    )
}

fn check<const N: usize>(
    device: &CubeDevice,
    spatial: [usize; N],
    kernel: [usize; N],
    stride: [usize; N],
    dilation: [usize; N],
    padding: [(usize, usize); N],
    permuted: bool,
    with_bias: bool,
) {
    let (batch, ci, co) = (2, 3, 5);
    let options = ConvOptions::new_with_padding(stride, padding, dilation, 1);
    let out_spatial = calculate_conv_output_sizes(&kernel, &stride, &padding, &dilation, &spatial);
    let mut in_shape = vec![batch];
    in_shape.extend(spatial);
    in_shape.push(ci);
    let mut weight_shape = vec![co];
    weight_shape.extend(kernel);
    weight_shape.push(ci);
    let input = (0..in_shape.iter().product())
        .map(|i| ((i * 7 % 17) as f32 - 8.0) / 8.0)
        .collect::<Vec<_>>();
    let weight = (0..weight_shape.iter().product())
        .map(|i| ((i * 3 % 11) as f32 - 5.0) / 8.0)
        .collect::<Vec<_>>();
    let bias = (0..co).map(|i| i as f32 / 4.0 - 0.5).collect::<Vec<_>>();
    let out_size = out_spatial.iter().product::<usize>();
    let taps = kernel.iter().product::<usize>();
    let mut expected = vec![0.0; batch * out_size * co];
    for b in 0..batch {
        for o in 0..out_size {
            let mut out_pos = [0; N];
            let mut rem = o;
            for axis in (0..N).rev() {
                out_pos[axis] = rem % out_spatial[axis];
                rem /= out_spatial[axis];
            }
            for oc in 0..co {
                let mut value = if with_bias { bias[oc] } else { 0.0 };
                for tap in 0..taps {
                    let mut rem = tap;
                    let mut pos = [0; N];
                    let mut valid = true;
                    for axis in (0..N).rev() {
                        let k = rem % kernel[axis];
                        rem /= kernel[axis];
                        let p = (out_pos[axis] * stride[axis] + k * dilation[axis]) as isize
                            - padding[axis].0 as isize;
                        valid &= p >= 0 && p < spatial[axis] as isize;
                        pos[axis] = p.max(0) as usize;
                    }
                    if valid {
                        let mut index = b;
                        for axis in 0..N {
                            index = index * spatial[axis] + pos[axis];
                        }
                        for ic in 0..ci {
                            value += input[index * ci + ic] * weight[(oc * taps + tap) * ci + ic];
                        }
                    }
                }
                expected[(b * out_size + o) * co + oc] = value;
            }
        }
    }
    let output = conv_im2col::<N>(
        upload(device, &in_shape, &input, permuted),
        upload(device, &weight_shape, &weight, permuted),
        with_bias.then(|| upload(device, &[co], &bias, false)),
        options,
    )
    .expect("explicit lowering must accept a small dense F32 filter");
    let mut out_shape = vec![batch];
    out_shape.extend(out_spatial);
    out_shape.push(co);
    assert_eq!(output.meta.shape().as_slice(), out_shape);
    let actual = into_data_sync(output).try_to_vec::<f32>().unwrap();
    for (i, (&a, &e)) in actual.iter().zip(&expected).enumerate() {
        assert!(
            (a - e).abs() < 1e-5,
            "N={N}, permuted={permuted}, bias={with_bias}, index={i}, got={a}, expected={e}"
        );
    }
}

fn suite(device: CubeDevice) {
    for permuted in [false, true] {
        for bias in [false, true] {
            check(&device, [17], [3], [2], [2], [(1, 2)], permuted, bias);
            check(
                &device,
                [13, 17],
                [3, 3],
                [1, 1],
                [1, 1],
                [(1, 1), (1, 1)],
                permuted,
                bias,
            );
            check(
                &device,
                [9, 11],
                [2, 3],
                [2, 1],
                [2, 2],
                [(0, 1), (3, 0)],
                permuted,
                bias,
            );
            check(
                &device,
                [5, 7, 9],
                [2, 2, 3],
                [1, 2, 1],
                [1, 1, 2],
                [(1, 0), (0, 1), (2, 0)],
                permuted,
                bias,
            );
        }
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_dense_im2col_matches_reference() {
    suite(CubeDevice::Cuda(Default::default()));
}

#[cfg(feature = "vulkan")]
#[test]
fn vulkan_dense_im2col_matches_reference() {
    suite(CubeDevice::Wgpu(
        cubecl::wgpu::WgpuDevice::default().on(cubecl::wgpu::WgpuBackend::Vulkan),
    ));
}

#[cfg(feature = "metal")]
#[test]
fn metal_dense_im2col_matches_reference() {
    suite(CubeDevice::Wgpu(
        cubecl::wgpu::WgpuDevice::default().on(cubecl::wgpu::WgpuBackend::Metal),
    ));
}
