use crate::{element::JitElement, ops::numeric::empty_device, tensor::JitTensor, JitRuntime};

use super::autotune_reduce;

pub use cubecl::reduce::instructions::{ArgMax, ArgMin, Mean, Prod, Sum};

/// Reduce all elements of the `input` tensor using the instruction `Rd` and the given [Strategy](ReduceStrategy).
///
/// Return an error if `strategy` is `Specific(strategy)` and the specified strategy is not supported by the `client`.
/// Also returns an error if the `axis` is larger than the `input` rank or if the shape of `output` is invalid.
/// The shape of `output` must be the same as input except with a value of 1 for the given `axis`.
///
/// If there is no error, the output is a tensor with decreasing strides
/// where the shape of reduced dim is set to 1 but all shape are similar to the input.
pub fn reduce<Run: JitRuntime, In: JitElement, Out: JitElement, Rd: cubecl::reduce::Reduce>(
    mut input: JitTensor<Run>,
    strategy: ReduceStrategy,
) -> Result<JitTensor<Run>, cubecl::reduce::ReduceError> {
    input.shape = input.shape.flatten();
    input.strides = vec![1];
    reduce_dim::<Run, In, Out, Rd>(input, 0, strategy)
}

/// Reduce the given `axis` of the `input` tensor using the instruction `Rd` and the given [Strategy](ReduceStrategy).
///
/// Return an error if `strategy` is `Specific(strategy)` and the specified strategy is not supported by the `client`.
/// Also returns an error if the `axis` is larger than the `input` rank or if the shape of `output` is invalid.
/// The shape of `output` must be the same as input except with a value of 1 for the given `axis`.
///
/// If there is no error, the output is a tensor with decreasing strides
/// where the shape of reduced dim is set to 1 but all shape are similar to the input.
pub fn reduce_dim<Run: JitRuntime, In: JitElement, Out: JitElement, Rd: cubecl::reduce::Reduce>(
    input: JitTensor<Run>,
    dim: usize,
    strategy: ReduceStrategy,
) -> Result<JitTensor<Run>, cubecl::reduce::ReduceError> {
    let client = input.client.clone();
    let output = init_reduce_output::<Run, In, Out>(&input, dim).ok_or(
        cubecl::reduce::ReduceError::InvalidAxis {
            axis: dim,
            rank: input.shape.num_dims(),
        },
    )?;
    let result = match strategy {
        ReduceStrategy::Unspecified => cubecl::reduce::reduce::<Run, In, Out, Rd>(
            &client,
            input.as_handle_ref(),
            output.as_handle_ref(),
            dim,
            None,
        ),
        ReduceStrategy::Specific(strategy) => cubecl::reduce::reduce::<Run, In, Out, Rd>(
            &client,
            input.as_handle_ref(),
            output.as_handle_ref(),
            dim,
            Some(strategy),
        ),
        #[cfg(feature = "autotune")]
        ReduceStrategy::Autotune => {
            autotune_reduce::<Run, In, Out, Rd>(&client, input, output.clone(), dim)
        }
    };
    result.map(|_| output)
}

/// Creates an empty output tensor with the proper shape and decreasing strides to reduce the given `axis` of `input`
/// or return `None` if `axis` is out-of-bound.
pub fn init_reduce_output<Run: JitRuntime, In: JitElement, Out: JitElement>(
    input: &JitTensor<Run>,
    dim: usize,
) -> Option<JitTensor<Run>> {
    (dim < input.shape.num_dims()).then(|| {
        let mut shape_out = input.shape.clone();
        shape_out.dims[dim] = 1;
        empty_device::<Run, Out>(input.client.clone(), input.device.clone(), shape_out)
    })
}

/// Select a strategy to perform a reduction.
#[derive(Copy, Clone, Debug)]
pub enum ReduceStrategy {
    /// Use a best-effort strategy based on the hardware capacity.
    /// This differs from Autotune as it doesn't try and compare many strategies to select the best.
    Unspecified,
    /// Fix the exact strategy for the reduction.
    Specific(cubecl::reduce::ReduceStrategy),
    /// Use autotune to find the best strategy given the hardware and the inputs.
    #[cfg(feature = "autotune")]
    Autotune,
}

impl Default for ReduceStrategy {
    fn default() -> Self {
        Self::Unspecified
    }
}
