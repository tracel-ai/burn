use crate::{
    CubeRuntime, CubeTuneId, kernel::interpolate::execute_interpolate,
    kernel::interpolate::map_mode, tensor::CubeTensor,
};
use burn_backend::cubecl::dtype_to_elem_type;
use burn_backend::ops::InterpolateOptions;
use cubecl::tune::{LocalTuner, Tunable, TunableSet, local_tuner};
use cubek::interpolate::{definition::InterpolateStrategy, tune_key::InterpolateAutotuneKey};

/// Interpolate operation with autotuning. This benchmarks multiple strategies and selects the best one at runtime.
pub fn interpolate_autotune<R: CubeRuntime>(
    input: CubeTensor<R>,
    output_size: [usize; 2],
    options: InterpolateOptions,
) -> CubeTensor<R> {
    let client = input.client.clone();

    static TUNER: LocalTuner<InterpolateAutotuneKey, CubeTuneId> = local_tuner!();

    let tunables = TUNER.init(|| {
        let mut set = TunableSet::new(create_key::<R>, input_gen::<R>);

        // The intents the cubek selector resolves against the device: how much of a cube one
        // problem occupies and whether the gathered input is staged. Every other choice is
        // solved from the hardware and the problem, so there is nothing else to sweep.
        for (name, strategy) in [
            (
                "maximize_throughput",
                InterpolateStrategy::MaximizeThroughput,
            ),
            ("minimize_latency", InterpolateStrategy::MinimizeLatency),
        ] {
            set = set.with(Tunable::new(name, move |(input, output_size, options)| {
                execute_interpolate::<R>(input, output_size, options, strategy)
            }));
        }

        set
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

    // The tensor is still NCHW here; the launch permutes it to the NHWC the key describes.
    let [_, channels, input_height, input_width] = input.meta.shape().dims();
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
