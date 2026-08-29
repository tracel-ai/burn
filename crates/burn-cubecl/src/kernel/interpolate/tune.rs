use crate::{
    CubeRuntime, CubeTuneId, kernel::interpolate::execute_interpolate,
    kernel::interpolate::map_mode, tensor::CubeTensor,
};
use burn_backend::cubecl::dtype_to_elem_type;
use burn_backend::ops::InterpolateOptions;
use cubecl::tune::{LocalTuner, Tunable, TunableSet, local_tuner};
use cubek::interpolate::{InterpolateStrategy, tune_key::InterpolateAutotuneKey};

/// Interpolate operation with autotuning. This benchmarks multiple strategies and selects the best one at runtime.
pub fn interpolate_autotune<R: CubeRuntime>(
    input: CubeTensor<R>,
    output_size: [usize; 2],
    options: InterpolateOptions,
) -> CubeTensor<R> {
    let client = input.client.clone();

    static TUNER: LocalTuner<InterpolateAutotuneKey, CubeTuneId> = local_tuner!();

    // Both intents, and only the intents: cubek resolves each to a blueprint from the device and
    // the real extents, so the tuner measures the two choices that swing a run — cube width and
    // whether the gathered input is staged — rather than a grid of tile sizes it would have to
    // re-derive per device. A `Forced` blueprint is for a characterization sweep, not for this.
    let tunables = TUNER.init(|| {
        TunableSet::new(create_key::<R>, input_gen::<R>)
            .with(Tunable::new(
                "maximize_throughput",
                |(input, output_size, options)| {
                    execute_interpolate::<R>(
                        input,
                        output_size,
                        options,
                        InterpolateStrategy::MaximizeThroughput,
                    )
                },
            ))
            .with(Tunable::new(
                "minimize_latency",
                |(input, output_size, options)| {
                    execute_interpolate::<R>(
                        input,
                        output_size,
                        options,
                        InterpolateStrategy::MinimizeLatency,
                    )
                },
            ))
    });

    TUNER.execute(
        &CubeTuneId::new(&client, &input.device),
        &client,
        tunables,
        (input, output_size, options),
    )
}

fn create_key<R: CubeRuntime>(
    (input, output_size, options): &(CubeTensor<R>, [usize; 2], InterpolateOptions),
) -> InterpolateAutotuneKey {
    let elem_input = dtype_to_elem_type(input.dtype);
    let elem_output = dtype_to_elem_type(input.dtype);
    let mode = map_mode(options.mode.clone());

    // The tensor is still NCHW here; the permute to NHWC happens inside the launch.
    let [_batch_size, channels, input_height, input_width] = input.meta.shape().dims();
    let [output_height, output_width] = *output_size;

    InterpolateAutotuneKey::generate(
        elem_input,
        elem_output,
        mode,
        options.align_corners,
        input_height,
        input_width,
        channels,
        output_height,
        output_width,
    )
}

fn input_gen<R: CubeRuntime>(
    _key: &InterpolateAutotuneKey,
    (input, output_size, options): &(CubeTensor<R>, [usize; 2], InterpolateOptions),
) -> (CubeTensor<R>, [usize; 2], InterpolateOptions) {
    (input.clone(), *output_size, options.clone())
}
