use burn_tensor::ElementConversion;
use cubecl::{
    client::ComputeClient,
    tune,
    tune::{local_tuner, tune_with, LocalTuner},
    AutotuneKey,
};
use serde::{Deserialize, Serialize};

use crate::{
    element::JitElement, kernel::prng::random_like_uniform, ops::numeric::empty_device,
    tensor::JitTensor, JitAutotuneKey, JitRuntime, JitTuneId,
};

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

/// Executes autotune on reduce operations.
pub fn autotune_reduce<
    Run: JitRuntime,
    In: JitElement,
    Out: JitElement,
    Rd: cubecl::reduce::Reduce,
>(
    client: &ComputeClient<Run::Server, Run::Channel>,
    input: JitTensor<Run>,
    output: JitTensor<Run>,
    dim: usize,
) -> Result<(), cubecl::reduce::ReduceError> {
    static TUNER: LocalTuner<JitAutotuneKey, JitTuneId> = local_tuner!();

    TUNER.execute(
        &JitTuneId::new::<Run>(&input.device),
        client,
        Box::new(ReduceOps::<Run, In, Out, Rd>::new(input, output, dim)),
    );

    Ok(())
}

#[derive(Hash, Eq, PartialEq, Debug, Clone, Serialize, Deserialize, AutotuneKey)]
/// Autotune key representative of redue versions
pub struct ReduceAutotuneKey {
    dtype: burn_tensor::DType,
    #[autotune(anchor)]
    reduce_axis_shape: usize,
    #[autotune(anchor)]
    reduce_axis_stride: usize,
    #[autotune(anchor)]
    outer_axes_product: usize, // The product of the shapes of all axes with greater strides.
}

impl ReduceAutotuneKey {
    pub(crate) fn generate<Run: JitRuntime>(input: &JitTensor<Run>, axis: usize) -> Self {
        let rank = input.shape.num_dims();

        if axis > rank {
            panic!("axis {axis} is out-of-bound for a rank of {rank}");
        }

        let dtype = input.dtype;
        let reduce_axis_shape = input.shape.dims[axis];
        let reduce_axis_stride = input.strides[axis];

        let outer_axes_product = input
            .strides
            .iter()
            .zip(input.shape.dims.iter())
            .filter_map(|(stride, shape)| (*stride > reduce_axis_stride).then_some(shape))
            .product();

        Self {
            dtype,
            reduce_axis_shape,
            reduce_axis_stride,
            outer_axes_product,
        }
    }
}

pub(crate) fn create_key<Run: JitRuntime>(
    input: &JitTensor<Run>,
    _output: &JitTensor<Run>,
    dim: &usize,
) -> JitAutotuneKey {
    JitAutotuneKey::Reduce(ReduceAutotuneKey::generate(input, *dim))
}

#[tune(
    operations(reduce_shared_false_plane_false, reduce_shared_true_plane_false, reduce_shared_false_plane_true, reduce_shared_true_plane_true),
    create_key = create_key::<Run>,
    should_run = should_run
)]
fn reduce_ops<Run: JitRuntime, In: JitElement, Out: JitElement, Rd: cubecl::reduce::Reduce>(
    key: JitAutotuneKey,
    input: JitTensor<Run>,
    output: JitTensor<Run>,
    dim: usize,
) {
    let random_bounds: (In, In) = ((-10.0_f32).elem::<In>(), (10.0_f32).elem::<In>());
    let input = random_like_uniform(input, random_bounds.0, random_bounds.1);

    let output = empty_device::<Run, Out>(
        output.client.clone(),
        output.device.clone(),
        output.shape.clone(),
    );

    tune_with!(input, output, dim)
}

fn should_run<Run: JitRuntime, In: JitElement, Out: JitElement, Rd: cubecl::reduce::Reduce>(
    op: &ReduceOps<Run, In, Out, Rd>,
    _key: &JitAutotuneKey,
    index: usize,
) -> bool {
    match index {
        // if strategy uses planes
        2 | 3 => {
            let properties = op.input.client.properties();
            properties.feature_enabled(cubecl::Feature::Plane)
                && properties
                    .hardware_properties()
                    .defined_plane_size()
                    .is_some()
        }
        _ => true,
    }
}

fn reduce_shared_false_plane_false<
    Run: JitRuntime,
    In: JitElement,
    Out: JitElement,
    Rd: cubecl::reduce::Reduce,
>(
    input: JitTensor<Run>,
    output: JitTensor<Run>,
    axis: usize,
) -> Result<(), String> {
    cubecl::reduce::reduce::<Run, In, Out, Rd>(
        &input.client,
        input.as_handle_ref(),
        output.as_handle_ref(),
        axis,
        Some(cubecl::reduce::ReduceStrategy {
            shared: false,
            use_planes: false,
        }),
    ).map_err(|e| format!("{e}"))
}

fn reduce_shared_true_plane_false<
    Run: JitRuntime,
    In: JitElement,
    Out: JitElement,
    Rd: cubecl::reduce::Reduce,
>(
    input: JitTensor<Run>,
    output: JitTensor<Run>,
    axis: usize,
) -> Result<(), String> {
    cubecl::reduce::reduce::<Run, In, Out, Rd>(
        &input.client,
        input.as_handle_ref(),
        output.as_handle_ref(),
        axis,
        Some(cubecl::reduce::ReduceStrategy {
            shared: true,
            use_planes: false,
        }),
    ).map_err(|e| format!("{e}"))
}

fn reduce_shared_false_plane_true<
    Run: JitRuntime,
    In: JitElement,
    Out: JitElement,
    Rd: cubecl::reduce::Reduce,
>(
    input: JitTensor<Run>,
    output: JitTensor<Run>,
    axis: usize,
) -> Result<(), String> {
    cubecl::reduce::reduce::<Run, In, Out, Rd>(
        &input.client,
        input.as_handle_ref(),
        output.as_handle_ref(),
        axis,
        Some(cubecl::reduce::ReduceStrategy {
            shared: false,
            use_planes: true,
        }),
    ).map_err(|e| format!("{e}"))
}

fn reduce_shared_true_plane_true<
    Run: JitRuntime,
    In: JitElement,
    Out: JitElement,
    Rd: cubecl::reduce::Reduce,
>(
    input: JitTensor<Run>,
    output: JitTensor<Run>,
    axis: usize,
) -> Result<(), String> {
    cubecl::reduce::reduce::<Run, In, Out, Rd>(
        &input.client,
        input.as_handle_ref(),
        output.as_handle_ref(),
        axis,
        Some(cubecl::reduce::ReduceStrategy {
            shared: true,
            use_planes: true,
        }),
    ).map_err(|e| format!("{e}"))
}
