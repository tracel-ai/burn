#![allow(missing_docs)]

use burn_tensor::ElementConversion;
use cubecl::{
    client::ComputeClient,
    tune,
    tune::{local_tuner, tune_with, LocalTuner},
    AutotuneKey,
};
use serde::{Deserialize, Serialize};

use crate::{
    kernel::prng::random_like_uniform, ops::numeric::empty_device, tensor::JitTensor,
    JitAutotuneKey, JitElement, JitRuntime, JitTuneId,
};

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

        Self::new(
            dtype,
            reduce_axis_shape,
            reduce_axis_stride,
            outer_axes_product,
        )
    }
}

pub(crate) fn create_key<Run: JitRuntime>(
    input: &JitTensor<Run>,
    _output: &JitTensor<Run>,
    dim: &usize,
) -> JitAutotuneKey {
    JitAutotuneKey::Reduce(ReduceAutotuneKey::generate(input, *dim))
}

pub use reduce_ops::*;
mod reduce_ops {
    #![allow(missing_docs)]

    use super::*;

    #[tune(
    operations(reduce, reduce_shared, reduce_plane, reduce_shared_plane),
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

    fn reduce<Run: JitRuntime, In: JitElement, Out: JitElement, Rd: cubecl::reduce::Reduce>(
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
        )
        .map_err(|e| format!("{e}"))
    }

    fn reduce_shared<
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
        )
        .map_err(|e| format!("{e}"))
    }

    fn reduce_plane<
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
        )
        .map_err(|e| format!("{e}"))
    }

    fn reduce_shared_plane<
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
        )
        .map_err(|e| format!("{e}"))
    }
}
