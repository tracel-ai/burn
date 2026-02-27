#[cfg(feature = "autotune")]
use super::{autotune_reduce, autotune_sum};
use crate::{
    CubeRuntime,
    ops::numeric::{empty_device_contiguous_dtype, zeros_client},
    tensor::CubeTensor,
};
use burn_backend::{DType, TensorMetadata};
use burn_std::Metadata;
use cubecl::{AutotuneKey, client::ComputeClient, features::TypeUsage, ir::StorageType};
use cubek::reduce::{
    ReduceDtypes, ReduceError, ReduceStrategy,
    components::instructions::ReduceOperationConfig,
    launch::{LineSizeStrategy, RoutineStrategy},
    routines::{BlueprintStrategy, unit::UnitStrategy},
    shared_sum,
};
use serde::{Deserialize, Serialize};

#[derive(Hash, Eq, PartialEq, Debug, Clone, Serialize, Deserialize, AutotuneKey)]
/// Autotune key representative of sum versions
pub struct SumAutotuneKey {
    /// The type of the tensor
    dtype: burn_backend::DType,
    /// The anchored length of the tensor
    #[autotune(anchor)]
    length: usize,
}

/// Check if the client supports atomic add for the given element type.
fn supports_atomic_add<R: CubeRuntime>(client: &ComputeClient<R>, dtype: DType) -> bool {
    client
        .properties()
        .type_usage(StorageType::Atomic(dtype.into()))
        .contains(TypeUsage::AtomicAdd)
}

/// [Sum](sum) with fallback when `client` doesn't support atomic add for the type `E`.
pub fn sum_fallback<R: CubeRuntime>(
    tensor: CubeTensor<R>,
    mut strategy: SumStrategy,
) -> Result<CubeTensor<R>, ReduceError> {
    // Early check before creating output and fallback
    if matches!(strategy, SumStrategy::OneShot(_))
        && !supports_atomic_add(&tensor.client, tensor.dtype)
    {
        strategy = SumStrategy::Chained(Default::default());
    }
    sum(tensor, strategy)
}

/// Specialize reduce function to compute the sum of all elements of the `input` tensor and return
/// the value into a single-element tensor of shape `1 x 1 x 1 x ...` with the same rank as `input`.
///
/// This is expected to be faster for larger tensors than calling [reduce] with the `Sum` instruction.
///
/// Return an error if the `client` doesn't support atomic add for the type `E`.
pub fn sum<Run: CubeRuntime>(
    tensor: CubeTensor<Run>,
    strategy: SumStrategy,
) -> Result<CubeTensor<Run>, ReduceError> {
    let client = tensor.client.clone();
    let device = tensor.device.clone();

    match strategy {
        SumStrategy::OneShot(cube_count) => {
            let output = zeros_client(client.clone(), device, [1].into(), tensor.dtype);
            let dtype = tensor.dtype;

            shared_sum::<Run>(
                &client,
                tensor.binding(),
                output.clone().binding(),
                cube_count,
                dtype.into(),
            )?;

            Ok(output)
        }
        SumStrategy::Chained(strategy) => {
            reduce::<Run>(tensor, None, strategy, ReduceOperationConfig::Sum)
        }
        #[cfg(feature = "autotune")]
        SumStrategy::Autotune => Ok(autotune_sum::<Run>(&client, tensor)),
    }
}

/// Select a strategy to perform a sum.
pub enum SumStrategy {
    /// Run a single kernel with many cubes working in parallel to sum all elements.
    /// The provided value is the number of elements summed per unit (up-to-rounding )
    OneShot(u32),
    /// Use multiple kernels
    Chained(KernelReduceStrategy),
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
pub fn reduce<Run: CubeRuntime>(
    mut tensor: CubeTensor<Run>,
    output_dtype: Option<DType>,
    strategy: KernelReduceStrategy,
    config: ReduceOperationConfig,
) -> Result<CubeTensor<Run>, cubek::reduce::ReduceError> {
    // In practice, it looks like starting by the axis with the smallest shape
    // and going in increasing order lead to the fastest calculation.
    let sorted_axis = argsort(tensor.meta.shape());
    for axis in sorted_axis {
        tensor = reduce_dim::<Run>(tensor, output_dtype, axis, strategy.clone(), config)?;
    }
    // reshape to scalar tensor
    *tensor.meta = Metadata::new([1], [1]);
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
pub fn reduce_dim<Run: CubeRuntime>(
    input: CubeTensor<Run>,
    output_dtype: Option<DType>,
    dim: usize,
    strategy: KernelReduceStrategy,
    config: ReduceOperationConfig,
) -> Result<CubeTensor<Run>, cubek::reduce::ReduceError> {
    debug_assert!(
        !matches!(
            config,
            ReduceOperationConfig::ArgMax | ReduceOperationConfig::ArgMin
        ) || output_dtype.is_some(),
        "The `output_dtype` has to be `Some` only when the `config` is `ArgMax` or `ArgMin`.
        "
    );

    let dtypes = config.precision(input.dtype.into(), output_dtype.map(Into::into));
    let client = input.client.clone();
    let output = init_reduce_output::<Run>(&input, dim, &dtypes).ok_or(
        cubek::reduce::ReduceError::InvalidAxis {
            axis: dim,
            rank: input.meta.num_dims(),
        },
    )?;

    let result = match strategy {
        KernelReduceStrategy::Unspecified => cubek::reduce::reduce::<Run>(
            &client,
            input.binding(),
            output.clone().binding(),
            dim,
            ReduceStrategy {
                routine: RoutineStrategy::Unit(BlueprintStrategy::Inferred(UnitStrategy)),
                line_size: LineSizeStrategy {
                    parallel_output_vectorization: false,
                },
            },
            config,
            dtypes,
        ),
        KernelReduceStrategy::Specific(strategy) => cubek::reduce::reduce::<Run>(
            &client,
            input.binding(),
            output.clone().binding(),
            dim,
            strategy,
            config,
            dtypes,
        ),
        #[cfg(feature = "autotune")]
        KernelReduceStrategy::Autotune => {
            autotune_reduce::<Run>(&client, input, output.clone(), dim, config, dtypes);
            Ok(())
        }
    };
    result.map(|_| output)
}

/// Creates an empty output tensor with the proper shape and decreasing strides to reduce the given `axis` of `input`
/// or return `None` if `axis` is out-of-bound.
pub fn init_reduce_output<Run: CubeRuntime>(
    input: &CubeTensor<Run>,
    dim: usize,
    dtypes: &ReduceDtypes,
) -> Option<CubeTensor<Run>> {
    (dim < input.meta.num_dims()).then(|| {
        let mut shape_out = input.shape();
        shape_out[dim] = 1;
        empty_device_contiguous_dtype(
            input.client.clone(),
            input.device.clone(),
            shape_out,
            dtypes.output.elem_type().into(),
        )
    })
}

/// Select a strategy to perform a reduction.
#[derive(Clone, Debug)]
pub enum KernelReduceStrategy {
    /// Use a best-effort strategy based on the hardware capacity.
    /// This differs from Autotune as it doesn't try and compare many strategies to select the best.
    Unspecified,
    /// Fix the exact strategy for the reduction.
    Specific(cubek::reduce::launch::ReduceStrategy),
    /// Use autotune to find the best strategy given the hardware and the inputs.
    #[cfg(feature = "autotune")]
    Autotune,
}

impl Default for KernelReduceStrategy {
    fn default() -> Self {
        #[cfg(feature = "autotune")]
        return Self::Autotune;

        #[cfg(not(feature = "autotune"))]
        return Self::Unspecified;
    }
}
