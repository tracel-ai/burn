use super::{conv_im2col, flatten_columns, forward_columns, im2col};
use crate::{
    CubeDevice,
    kernel::into_contiguous,
    ops::{into_data_sync, swap_dims},
    tensor::CubeTensor,
};
use burn_backend::{
    DType,
    ops::{ConvOptions, conv::calculate_conv_output_sizes},
};
use burn_std::{Metadata, Shape, tensor::contiguous_strides};
use cubecl::prelude::CubeElement;

#[test]
fn forward_workspace_is_bounded_without_row_padding() {
    let device = CubeDevice::default();
    let input = upload(&device, &[1], &[0.0], false);

    // A 1x2 filter over a 4096x4097 image: pitched 16-byte rows would
    // allocate 256 MiB even though the logical workspace is only 128 MiB.
    let rows = 4096 * 4096;
    let columns = forward_columns(&input, rows, 2).unwrap();
    assert_eq!(&columns.meta.strides()[..], &[2, 1]);
    assert_eq!(columns.handle.size(), 128 * 1024 * 1024);
    assert!(forward_columns(&input, rows + 1, 2).is_none());
}

#[test]
fn slice_lowering_preserves_workspace_and_zeros_padding() {
    let device = CubeDevice::default();
    let input = upload(&device, &[1, 4, 1], &[1.0, -2.0, 3.0, 5.0], false);
    // Nonzero contents ensure omitted padding positions are actually cleared.
    let columns = upload(&device, &[4, 3], &[99.0; 12], false);
    let workspace_bytes = columns.handle.size();
    let columns = flatten_columns(im2col(
        input,
        columns,
        &[4],
        &[3],
        &ConvOptions::new([1], [1], [1], 1),
    ));
    assert_eq!(&columns.meta.strides()[..], &[3, 1]);
    assert_eq!(columns.handle.size(), workspace_bytes);
    assert_eq!(
        into_data_sync(columns).try_to_vec::<f32>().unwrap(),
        [
            0.0, 1.0, -2.0, 1.0, -2.0, 3.0, -2.0, 3.0, 5.0, 3.0, 5.0, 0.0
        ]
    );
}

fn upload(device: &CubeDevice, shape: &[usize], values: &[f32], permuted: bool) -> CubeTensor {
    let client = device.client();
    let tensor = CubeTensor::new(
        client.clone(),
        client.create_from_slice(f32::as_bytes(values)),
        Metadata::new(Shape::from(shape), contiguous_strides(shape)),
        device.clone(),
        DType::F32,
    );
    if permuted {
        let last = shape.len() - 1;
        // Preserve logical values while making channels non-contiguous.
        swap_dims(into_contiguous(swap_dims(tensor, 1, last)), 1, last)
    } else {
        tensor
    }
}

fn check<const N: usize>(
    device: &CubeDevice,
    spatial: [usize; N],
    kernel: [usize; N],
    options: ConvOptions<N>,
    permuted: bool,
    with_bias: bool,
) {
    let (batch, ci, co) = (2, 3, 5);
    let out_spatial = calculate_conv_output_sizes(
        &kernel,
        &options.stride,
        &options.padding,
        &options.dilation,
        &spatial,
    );
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
                        let p = (out_pos[axis] * options.stride[axis] + k * options.dilation[axis])
                            as isize
                            - options.padding[axis].0 as isize;
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
    // Cover each layout/bias combination once across the distinct geometries.
    check(
        &device,
        [17],
        [3],
        ConvOptions::new_with_padding([2], [(1, 2)], [2], 1),
        false,
        false,
    );
    check(
        &device,
        [13, 17],
        [3, 3],
        ConvOptions::new([1, 1], [1, 1], [1, 1], 1),
        false,
        true,
    );
    check(
        &device,
        [9, 11],
        [2, 3],
        ConvOptions::new_with_padding([2, 1], [(0, 1), (3, 0)], [2, 2], 1),
        true,
        false,
    );
    check(
        &device,
        [5, 7, 9],
        [2, 2, 3],
        ConvOptions::new_with_padding([1, 2, 1], [(1, 0), (0, 1), (2, 0)], [1, 1, 2], 1),
        true,
        true,
    );
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
