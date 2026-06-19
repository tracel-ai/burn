use super::data::NestedValue;

/// A trait that defines the adapter for a Burn module.
///
/// This is used to adapt an incoming module to a Burn module.
pub trait BurnModuleAdapter: Sized {
    /// Adapts a module.
    fn adapt(name: &str, data: NestedValue) -> NestedValue {
        match name {
            "BatchNorm" => Self::adapt_batch_norm(data),
            "Conv1d" => Self::adapt_conv1d(data),
            "Conv2d" => Self::adapt_conv2d(data),
            "Conv3d" => Self::adapt_conv3d(data),
            "ConvTranspose1d" => Self::adapt_conv_transpose_1d(data),
            "ConvTranspose2d" => Self::adapt_conv_transpose_2d(data),
            "ConvTranspose3d" => Self::adapt_conv_transpose_3d(data),
            "Embedding" => Self::adapt_embedding(data),
            "GroupNorm" => Self::adapt_group_norm(data),
            "LayerNorm" => Self::adapt_layer_norm(data),
            "Linear" => Self::adapt_linear(data),
            _ => data,
        }
    }

    /// Adapts a linear module.
    fn adapt_linear(data: NestedValue) -> NestedValue {
        data
    }

    /// Adapts a Convolution 1D module.
    fn adapt_conv1d(data: NestedValue) -> NestedValue {
        data
    }

    /// Adapts a Convolution 2D module.
    fn adapt_conv2d(data: NestedValue) -> NestedValue {
        data
    }

    /// Adapts a Convolution 3D module.
    fn adapt_conv3d(data: NestedValue) -> NestedValue {
        data
    }

    /// Adapts convolution transpose 1D module.
    fn adapt_conv_transpose_1d(data: NestedValue) -> NestedValue {
        data
    }

    /// Adapts convolution transpose 2D module.
    fn adapt_conv_transpose_2d(data: NestedValue) -> NestedValue {
        data
    }

    /// Adapts convolution transpose 2D module.
    fn adapt_conv_transpose_3d(data: NestedValue) -> NestedValue {
        data
    }

    /// Adapts embedding module.
    fn adapt_embedding(data: NestedValue) -> NestedValue {
        data
    }

    /// Adapts group normalization module.
    fn adapt_group_norm(data: NestedValue) -> NestedValue {
        data
    }

    /// Adapts layer normalization module.
    fn adapt_layer_norm(data: NestedValue) -> NestedValue {
        data
    }

    /// Adapts batch normalization module.
    fn adapt_batch_norm(data: NestedValue) -> NestedValue {
        data
    }
}

/// Default adapter that takes no action.
pub struct DefaultAdapter;
impl BurnModuleAdapter for DefaultAdapter {}
