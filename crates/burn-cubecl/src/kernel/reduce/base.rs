#[cfg(feature = "autotune")]
use super::{autotune_reduce, autotune_sum};
use crate::{element::CubeElement, ops::numeric::empty_device, tensor::CubeTensor, CubeRuntime};
use burn_tensor::Shape;
pub use cubecl::reduce::instructions::{ArgMax, ArgMin, Mean, Prod, Sum};
use cubecl::reduce::shared_sum;

/// Specialize reduce function to compute the sum of all elements of the `input` tensor and return
/// the value into a single-element tensor of shape `1 x 1 x 1 x ...` with the same rank as `input`.
///
/// This is expected to be faster for larger tensors than calling [reduce] with the `Sum` instruction.
///
/// Return an error if the `client` doesn't support atomic add for the type `E`.
pub fn sum<Run: CubeRuntime, E: CubeElement>(
    tensor: CubeTensor<Run>,
    cube_count: SumStrategy,
) -> Result<CubeTensor<Run>, cubecl::reduce::ReduceError> {
    let client = tensor.client.clone();
    let device = tensor.device.clone();

    match cube_count {
        SumStrategy::OneShot(cube_count) => {
            let handle = client.create(E::as_bytes(&[E::from_int(0)]));
            let output =
                CubeTensor::new_contiguous(client.clone(), device, [1].into(), handle, E::dtype());
            shared_sum::<Run, E>(
                &client,
                tensor.as_handle_ref(),
                output.as_handle_ref(),
                cube_count,
            )?;

            Ok(output)
        }
        SumStrategy::Chained(strategy) => reduce::<Run, E, E, Sum>(tensor, strategy),
        #[cfg(feature = "autotune")]
        SumStrategy::Autotune => Ok(autotune_sum::<Run, E>(&client, tensor)),
    }
}

/// Select a strategy to perform a sum.
pub enum SumStrategy {
    /// Run a single kernel with many cubes working in parallel to sum all elements.
    /// The provided value is the number of elements summed per unit (up-to-rounding )
    OneShot(u32),
    /// Use multiple kernels
    Chained(ReduceStrategy),
    /// Use autotune to find the best cube count given the hardware and the input.
    #[cfg(feature = "autotune")]
    Autotune,
}

impl Default for SumStrategy {
    fn default() -> Self {
        #[cfg(feature = "autotune")]
        return Self::Autotune;

        #[cfg(not(feature = "autotune"))]
        return Self::OneShot(4);
    }
}

/// Reduce all elements of the `input` tensor using the instruction `Rd` and the given [Strategy](ReduceStrategy).
///
/// Return an error if `strategy` is `Specific(strategy)` and the specified strategy is not supported by the `client`.
///
/// If there is no error, the output is a tensor with decreasing strides
/// where the shape of reduced dim is set to 1 but all shape are similar to the input.
pub fn reduce<Run: CubeRuntime, In: CubeElement, Out: CubeElement, Rd: cubecl::reduce::Reduce>(
    mut tensor: CubeTensor<Run>,
    strategy: ReduceStrategy,
) -> Result<CubeTensor<Run>, cubecl::reduce::ReduceError> {
    // In practice, it looks like starting by the axis with the smallest shape
    // and going in increasing order lead to the fastest calculation.
    let sorted_axis = argsort(&tensor.shape.dims);
    for axis in sorted_axis {
        tensor = reduce_dim::<Run, In, Out, Rd>(tensor, axis, strategy)?;
    }
    // reshape to scalar tensor
    tensor.shape = Shape::new([1]);
    tensor.strides = vec![1];
    Ok(tensor)
}

fn argsort(shape: &[usize]) -> Vec<usize> {
    let mut indices = (0..shape.len()).collect::<Vec<_>>();
    indices.sort_by_key(|&i| &shape[i]);
    indices
}

/// Reduce the given `axis` of the `input` tensor using the instruction `Rd` and the given [Strategy](ReduceStrategy).
///
/// Return an error if `strategy` is `Specific(strategy)` and the specified strategy is not supported by the `client`.
/// Also returns an error if the `axis` is larger than the `input` rank or if the shape of `output` is invalid.
///
/// If there is no error, the output is a tensor with decreasing strides
/// where the shape of reduced dim is set to 1 but all shape are similar to the input.
pub fn reduce_dim<
    Run: CubeRuntime,
    In: CubeElement,
    Out: CubeElement,
    Rd: cubecl::reduce::Reduce,
>(
    input: CubeTensor<Run>,
    dim: usize,
    strategy: ReduceStrategy,
) -> Result<CubeTensor<Run>, cubecl::reduce::ReduceError> {
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
            autotune_reduce::<Run, In, Out, Rd>(&client, input, output.clone(), dim);
            Ok(())
        }
    };
    result.map(|_| output)
}

/// Creates an empty output tensor with the proper shape and decreasing strides to reduce the given `axis` of `input`
/// or return `None` if `axis` is out-of-bound.
pub fn init_reduce_output<Run: CubeRuntime, In: CubeElement, Out: CubeElement>(
    input: &CubeTensor<Run>,
    dim: usize,
) -> Option<CubeTensor<Run>> {
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
        #[cfg(feature = "autotune")]
        return Self::Autotune;

        #[cfg(not(feature = "autotune"))]
        return Self::Unspecified;
    }
}
