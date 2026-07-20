use alloc::sync::Arc;
use burn_backend::ops::ConvOptions;
use cubecl::{
    client::ComputeClient,
    tune::{Bounds, BoundsGenerator},
};

use crate::{CubeAutotuneKey, CubeRuntime, tensor::CubeTensor};

type Inputs<R, const N: usize> = (
    CubeTensor<R>,
    CubeTensor<R>,
    Option<CubeTensor<R>>,
    ConvOptions<N>,
);

type BoundsGen<R, const N: usize> =
    dyn BoundsGenerator<CubeAutotuneKey, Inputs<R, N>> + Send + Sync;

/// Creates a closure that calculates performance bounds for forward convolution autotuning.
pub(super) fn create_bounds<R: CubeRuntime, const N: usize>(
    client: &ComputeClient<R>,
) -> Arc<BoundsGen<R, N>> {
    let owned_client = client.clone();
    Arc::new(
        move |key: &CubeAutotuneKey, (input, weight, bias, _options): &Inputs<R, N>| {
            let CubeAutotuneKey::Conv(conv_key) = key else {
                unreachable!()
            };

            Bounds {
                bounds: crate::kernel::conv::bounds::conv_autotune_bounds(
                    &owned_client,
                    conv_key,
                    input.meta.num_elements(),
                    weight.meta.num_elements(),
                    bias.is_some(),
                ),
                launch_overhead: cubecl::std::throughput::measure_peak_throughput(
                    &owned_client,
                    cubecl::throughput::ThroughputKey {
                        mode: cubecl::throughput::ThroughputMode::Launch,
                    },
                )
                .duration,
            }
        },
    )
}
