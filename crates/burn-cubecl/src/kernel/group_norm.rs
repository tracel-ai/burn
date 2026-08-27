use burn_backend::cubecl::dtype_to_storage_type;
use burn_backend::{Shape, TensorMetadata};
use burn_std::Metadata;
use cubecl::prelude::*;

use crate::{
    CubeRuntime, kernel::utils::address_type, ops::numeric::empty_device_contiguous_dtype,
    tensor::CubeTensor,
};

#[cube]
fn group_norm_offset<F: Float>(
    tensor: &Tensor<F>,
    batch: usize,
    channel: usize,
    spatial_index: usize,
    num_channels: usize,
    spatial_size: usize,
    #[comptime] dense_nchw: bool,
    #[comptime] dense_channels_innermost: bool,
) -> usize {
    if dense_nchw {
        (batch * num_channels + channel) * spatial_size + spatial_index
    } else if dense_channels_innermost {
        (batch * spatial_size + spatial_index) * num_channels + channel
    } else {
        let mut offset = batch * tensor.stride(0) + channel * tensor.stride(1);
        let mut spatial_index = spatial_index;
        let mut dim = tensor.rank();
        while dim > 2 {
            dim -= 1;
            let shape = tensor.shape(dim);
            offset += spatial_index % shape * tensor.stride(dim);
            spatial_index /= shape;
        }
        offset
    }
}

#[cube(launch, address_type = "dynamic")]
fn group_norm_kernel<F: Float>(
    input: &Tensor<F>,
    gamma: &Tensor<F>,
    beta: &Tensor<F>,
    output: &mut Tensor<F>,
    num_groups: usize,
    spatial_size: usize,
    channels_per_group: usize,
    num_channels: usize,
    epsilon: InputScalar,
    #[comptime] scale: bool,
    #[comptime] shift: bool,
    #[comptime] channels_innermost: bool,
    #[comptime] dense_nchw: bool,
    #[comptime] dense_channels_innermost: bool,
    #[define(F)] _dtype: ElemType,
) {
    let batch_group = CUBE_POS_X as usize;
    let group = batch_group % num_groups;
    let batch = batch_group / num_groups;
    let unit = UNIT_POS_X as usize;
    let cube_stride = CUBE_DIM_X as usize;
    let group_size = channels_per_group * spatial_size;

    let mut count = f32::new(0.0_f32);
    let mut mean = f32::new(0.0_f32);
    let mut m2 = f32::new(0.0_f32);
    let mut index = unit;
    while index < group_size {
        let input_offset = if dense_nchw {
            (batch * num_channels + group * channels_per_group) * spatial_size + index
        } else {
            let channel_in_group = if channels_innermost {
                index % channels_per_group
            } else {
                index / spatial_size
            };
            let spatial_index = if channels_innermost {
                index / channels_per_group
            } else {
                index % spatial_size
            };
            let channel = group * channels_per_group + channel_in_group;
            group_norm_offset(
                input,
                batch,
                channel,
                spatial_index,
                num_channels,
                spatial_size,
                dense_nchw,
                dense_channels_innermost,
            )
        };
        let value = f32::cast_from(input[input_offset]);
        count += f32::new(1.0_f32);
        let delta = value - mean;
        mean += delta / count;
        let delta_next = value - mean;
        m2 += delta * delta_next;
        index += cube_stride;
    }

    let mut counts = Shared::new_slice(256usize);
    let mut means = Shared::new_slice(256usize);
    let mut m2s = Shared::new_slice(256usize);
    counts[unit] = count;
    means[unit] = mean;
    m2s[unit] = m2;
    sync_cube();

    // Pairwise Welford merge over the fixed power-of-two workgroup.
    let mut offset = 128usize;
    while offset > 0 {
        if unit < offset {
            let count_a = counts[unit];
            let count_b = counts[unit + offset];
            if count_b > f32::new(0.0_f32) {
                let count_total = count_a + count_b;
                let delta = means[unit + offset] - means[unit];
                means[unit] += delta * count_b / count_total;
                m2s[unit] += m2s[unit + offset] + delta * delta * count_a * count_b / count_total;
                counts[unit] = count_total;
            }
        }
        sync_cube();
        offset /= 2;
    }

    let group_mean = means[0];
    let variance = m2s[0] / counts[0];
    let inv_std = f32::new(1.0_f32) / (variance + epsilon.get::<f32>()).sqrt();

    let mut index = unit;
    while index < group_size {
        let channel_in_group = if dense_nchw && !scale && !shift {
            usize::cast_from(u32::new(0))
        } else if channels_innermost {
            index % channels_per_group
        } else {
            index / spatial_size
        };
        let spatial_index = if dense_nchw {
            usize::cast_from(u32::new(0))
        } else if channels_innermost {
            index / channels_per_group
        } else {
            index % spatial_size
        };
        let channel = group * channels_per_group + channel_in_group;
        let dense_nchw_offset =
            (batch * num_channels + group * channels_per_group) * spatial_size + index;
        let input_offset = if dense_nchw {
            dense_nchw_offset
        } else {
            group_norm_offset(
                input,
                batch,
                channel,
                spatial_index,
                num_channels,
                spatial_size,
                dense_nchw,
                dense_channels_innermost,
            )
        };
        let output_offset = if dense_nchw {
            dense_nchw_offset
        } else if dense_channels_innermost {
            (batch * spatial_size + spatial_index) * num_channels + channel
        } else {
            let mut output_offset = batch * output.stride(0) + channel * output.stride(1);
            let mut output_spatial_index = spatial_index;
            let mut dim = output.rank();
            while dim > 2 {
                dim -= 1;
                let shape = output.shape(dim);
                output_offset += output_spatial_index % shape * output.stride(dim);
                output_spatial_index /= shape;
            }
            output_offset
        };
        let mut value = (f32::cast_from(input[input_offset]) - group_mean) * inv_std;
        if scale {
            value *= f32::cast_from(gamma[channel * gamma.stride(0)]);
        }
        if shift {
            value += f32::cast_from(beta[channel * beta.stride(0)]);
        }
        output[output_offset] = F::cast_from(value);
        index += cube_stride;
    }
}

fn is_dense_channels_innermost(shape: &Shape, strides: &[usize]) -> bool {
    if shape.num_elements() == 0 || strides[1] != 1 {
        return false;
    }

    let mut expected_stride = shape[1];
    for dim in (2..shape.num_dims()).rev() {
        if shape[dim] > 1 && strides[dim] != expected_stride {
            return false;
        }
        expected_stride *= shape[dim];
    }

    shape[0] == 1 || strides[0] == expected_stride
}

fn empty_group_norm_output<R: CubeRuntime>(
    input: &CubeTensor<R>,
    preserve_layout: bool,
) -> CubeTensor<R> {
    let shape = input.shape();
    // A unique address per element plus an exact-size handle makes the input strides a dense
    // permutation, so the same metadata is valid for a fresh element-count-sized allocation.
    if preserve_layout {
        let strides = input.meta.strides().clone();
        let handle = input
            .client
            .empty(shape.num_elements() * input.dtype.size());
        CubeTensor::new(
            input.client.clone(),
            handle,
            Metadata::new(shape, strides),
            input.device.clone(),
            input.dtype,
        )
    } else {
        empty_device_contiguous_dtype(
            input.client.clone(),
            input.device.clone(),
            shape,
            input.dtype,
        )
    }
}

pub(crate) fn group_norm<R: CubeRuntime>(
    input: CubeTensor<R>,
    gamma: Option<CubeTensor<R>>,
    beta: Option<CubeTensor<R>>,
    num_groups: usize,
    epsilon: f64,
) -> CubeTensor<R> {
    let shape = input.shape();
    let rank = shape.num_dims();
    assert!(rank >= 3, "group_norm: input rank must be at least 3");
    assert!(num_groups > 0, "group_norm: num_groups must be positive");

    let batch_size = shape[0];
    let num_channels = shape[1];
    assert_eq!(
        num_channels % num_groups,
        0,
        "group_norm: number of channels must be divisible by number of groups"
    );
    if let Some(gamma) = &gamma {
        assert_eq!(
            gamma.shape(),
            Shape::new([num_channels]),
            "group_norm: gamma must have shape [num_channels]"
        );
        assert_eq!(
            gamma.dtype, input.dtype,
            "group_norm: gamma must have the same dtype as the input"
        );
        input.assert_is_on_same_device(gamma);
    }
    if let Some(beta) = &beta {
        assert_eq!(
            beta.shape(),
            Shape::new([num_channels]),
            "group_norm: beta must have shape [num_channels]"
        );
        assert_eq!(
            beta.dtype, input.dtype,
            "group_norm: beta must have the same dtype as the input"
        );
        input.assert_is_on_same_device(beta);
    }

    let client = input.client.clone();
    let dtype = input.dtype;
    let preserve_layout =
        shape.num_elements() != 0 && input.is_nonoverlapping() && input.is_contiguous_buffer();
    let dense_nchw = preserve_layout && input.is_contiguous();
    let dense_channels_innermost =
        preserve_layout && is_dense_channels_innermost(&shape, input.meta.strides());
    let output = empty_group_norm_output(&input, preserve_layout);
    if shape.num_elements() == 0 {
        return output;
    }

    let scale = gamma.is_some();
    let shift = beta.is_some();
    let channels_innermost = input.meta.strides()[1] == 1;
    let gamma = gamma.unwrap_or_else(|| input.clone());
    let beta = beta.unwrap_or_else(|| input.clone());
    let spatial_size = shape[2..].iter().product::<usize>();
    let channels_per_group = num_channels / num_groups;

    assert!(
        client.properties().hardware.max_cube_dim.0 >= 256,
        "group_norm: runtime must support 256 threads per workgroup"
    );
    let workgroups = batch_size
        .checked_mul(num_groups)
        .and_then(|count| u32::try_from(count).ok())
        .expect("group_norm: batch_size * num_groups exceeds the dispatch limit");

    group_norm_kernel::launch::<R>(
        &client,
        CubeCount::Static(workgroups, 1, 1),
        CubeDim::new_1d(256),
        address_type!(input, gamma, beta, output),
        input.into_tensor_arg(),
        gamma.into_tensor_arg(),
        beta.into_tensor_arg(),
        output.clone().into_tensor_arg(),
        num_groups,
        spatial_size,
        channels_per_group,
        num_channels,
        InputScalar::new(
            epsilon as f32,
            dtype_to_storage_type(burn_backend::DType::F32),
        ),
        scale,
        shift,
        channels_innermost,
        dense_nchw,
        dense_channels_innermost,
        dtype_to_storage_type(dtype),
    );

    output
}
