use crate::{Backend, DType, TensorMetadata, tensor::FloatTensor};
use burn_std::{FloatDType, IntDType, Shape};

/// Portable group normalization implementation used by backend fallbacks.
pub fn group_norm_fallback<B: Backend>(
    mut tensor: FloatTensor<B>,
    mut gamma: Option<FloatTensor<B>>,
    mut beta: Option<FloatTensor<B>>,
    num_groups: usize,
    epsilon: f64,
) -> FloatTensor<B> {
    let shape = tensor.shape();
    let rank = shape.num_dims();
    assert!(rank >= 3, "group_norm: input rank must be at least 3");
    assert!(num_groups > 0, "group_norm: num_groups must be positive");

    let num_channels = shape[1];
    if let Some(gamma) = &gamma {
        assert_eq!(
            gamma.shape(),
            Shape::new([num_channels]),
            "group_norm: gamma must have shape [num_channels]"
        );
        assert_eq!(
            gamma.dtype(),
            tensor.dtype(),
            "group_norm: gamma must have the same dtype as the input"
        );
        assert_eq!(
            gamma.device(),
            tensor.device(),
            "group_norm: gamma must be on the same device as the input"
        );
    }
    if let Some(beta) = &beta {
        assert_eq!(
            beta.shape(),
            Shape::new([num_channels]),
            "group_norm: beta must have shape [num_channels]"
        );
        assert_eq!(
            beta.dtype(),
            tensor.dtype(),
            "group_norm: beta must have the same dtype as the input"
        );
        assert_eq!(
            beta.device(),
            tensor.device(),
            "group_norm: beta must be on the same device as the input"
        );
    }

    let batch_size = shape[0];
    assert_eq!(
        num_channels % num_groups,
        0,
        "group_norm: number of channels must be divisible by number of groups"
    );

    // Materialize strided views before folding channels and spatial dimensions together.
    let device = tensor.device();
    if shape.num_elements() == 0 {
        let tensor_zeros = B::float_zeros(shape.clone(), &device, tensor.dtype().into());
        let mut output = B::float_add(tensor_zeros, tensor);
        let mut broadcast_dims = alloc::vec![1; rank];
        broadcast_dims[1] = num_channels;
        if let Some(gamma) = gamma {
            output = B::float_mul(
                output,
                B::float_reshape(gamma, Shape::from(broadcast_dims.clone())),
            );
        }
        if let Some(beta) = beta {
            output = B::float_add(output, B::float_reshape(beta, Shape::from(broadcast_dims)));
        }
        return output;
    }
    let channel_indices = B::int_arange(0..num_channels as i64, &device, IntDType::I64);
    tensor = B::float_select(tensor, 1, channel_indices);

    let widened = tensor.dtype() == DType::F16;
    if widened {
        tensor = B::float_cast(tensor, FloatDType::F32);
        gamma = gamma.map(|gamma| B::float_cast(gamma, FloatDType::F32));
        beta = beta.map(|beta| B::float_cast(beta, FloatDType::F32));
    }

    let hidden_size = shape[2..].iter().product::<usize>() * num_channels / num_groups;
    let tensor = B::float_reshape(tensor, Shape::new([batch_size, num_groups, hidden_size]));
    let mean = B::float_mean_dim(tensor.clone(), 2);
    let centered = B::float_sub(tensor, mean);
    let var = B::float_mean_dim(B::float_mul(centered.clone(), centered.clone()), 2);
    let denom = B::float_sqrt(B::float_add_scalar(var, epsilon.into()));
    let normalized = B::float_reshape(B::float_div(centered, denom), shape);

    let mut broadcast_dims = alloc::vec![1; rank];
    broadcast_dims[1] = num_channels;
    let output = match gamma {
        Some(gamma) => B::float_mul(
            normalized,
            B::float_reshape(gamma, Shape::from(broadcast_dims.clone())),
        ),
        None => normalized,
    };
    let output = match beta {
        Some(beta) => B::float_add(output, B::float_reshape(beta, Shape::from(broadcast_dims))),
        None => output,
    };

    if widened {
        B::float_cast(output, FloatDType::F16)
    } else {
        output
    }
}
