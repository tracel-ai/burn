use crate::{
    CubeTuneId,
    kernel::{
        autotune_bounds,
        interpolate::{execute_interpolate, map_options},
    },
    ops::permute_nchw_to_nhwc_shape,
    tensor::CubeTensor,
};
use burn_backend::cubecl::dtype_to_elem_type;
use burn_backend::ops::InterpolateOptions;
use cubecl::{
    std::throughput::roofline_bounds,
    tune::{LocalTuner, Tunable, TunableSet, local_tuner},
};
use cubek::interpolate::{
    InterpolateStrategy,
    definition::{InterpolateCost, InterpolateForwardProblem, InterpolateProblem},
    tune_key::InterpolateAutotuneKey,
};

type Inputs = (CubeTensor, [usize; 2], InterpolateOptions);

/// What the tuner measures: the bottleneck each strategy takes for granted.
///
/// The geometry behind them is cubek's to pick, from the device and the problem, so nothing here
/// enumerates plane counts or row runs. On a GPU the two differ in how much of a cube one problem
/// occupies and in whether the gathered input is staged, which is the choice that swings a run
/// either way and is therefore measured rather than modelled. A CPU resolves both to one
/// blueprint, and both stay registered there rather than being filtered: the tunable set is built
/// once per process and shared by every device in it, so it cannot be cut to the first device's
/// kind. The cost is one duplicate measurement per key.
///
/// [`MaximizeThroughput`](InterpolateStrategy::MaximizeThroughput) leads, because it stages
/// nothing and so is the one no device refuses: the short circuit gets its first chance on a
/// candidate that always runs.
///
/// The names are spelled out rather than derived from the variants, because a recorded tune result
/// is keyed by them and renaming a variant must not silently invalidate one.
const STRATEGIES: [(&str, InterpolateStrategy); 2] = [
    (
        "maximize_throughput",
        InterpolateStrategy::MaximizeThroughput,
    ),
    ("minimize_latency", InterpolateStrategy::MinimizeLatency),
];

/// Interpolate operation with autotuning. This benchmarks multiple strategies and selects the
/// best one at runtime.
pub fn interpolate_autotune(
    input: CubeTensor,
    output_size: [usize; 2],
    options: InterpolateOptions,
) -> CubeTensor {
    let client = input.client.clone();

    static TUNER: LocalTuner<InterpolateAutotuneKey, CubeTuneId> = local_tuner!();

    let tune_id = CubeTuneId::new(&client, &input.device);
    let tunables = TUNER.init(&tune_id, move || {
        let mut set = with_bounds(TunableSet::new(create_key, input_gen));

        for (name, strategy) in STRATEGIES {
            set = set.with(Tunable::new(name, move |(input, output_size, options)| {
                execute_interpolate(input, output_size, options, strategy)
            }));
        }

        set
    });

    TUNER.execute(&tune_id, &client, tunables, (input, output_size, options))
}

/// Registers the roofline bounds the short circuit needs.
///
/// Interpolation is memory bound for the cheap filters, but the arithmetic per output element
/// grows with the tap count, and Lanczos3 spends two sines on every weight. Costing both and
/// letting the roofline take whichever is slower is what keeps the limit reachable across modes,
/// rather than holding every mode to a bandwidth figure only nearest can approach.
fn with_bounds<Out: 'static>(
    set: TunableSet<InterpolateAutotuneKey, Inputs, Out>,
) -> TunableSet<InterpolateAutotuneKey, Inputs, Out> {
    autotune_bounds::with_bounds(
        set,
        |_key, (input, output_size, options): &Inputs, thresholds| {
            let problem = forward_problem(input, output_size, options);
            let cost = InterpolateCost::new(
                InterpolateProblem::Forward(problem),
                dtype_to_elem_type(input.dtype),
            );

            roofline_bounds(&input.client, cost.compute_key(), cost.work(), thresholds)
        },
    )
}

/// The problem the kernel actually runs, in the NHWC layout it reads.
///
/// The tensor is still NCHW here: `execute_interpolate` permutes it. Reading its extents
/// positionally at this point is what used to feed the width in as the channel count.
fn forward_problem(
    input: &CubeTensor,
    output_size: &[usize; 2],
    options: &InterpolateOptions,
) -> InterpolateForwardProblem {
    let shape = permute_nchw_to_nhwc_shape(input.meta.shape().clone());

    InterpolateForwardProblem::from_input_output_shapes(
        &shape,
        output_size,
        map_options(options.clone()),
    )
}

fn create_key((input, output_size, options): &Inputs) -> InterpolateAutotuneKey {
    let elem = dtype_to_elem_type(input.dtype);
    let problem = forward_problem(input, output_size, options);

    InterpolateAutotuneKey::generate(
        elem,
        elem,
        problem.options.mode,
        problem.options.align_corners,
        problem.input_height,
        problem.input_width,
        problem.channels,
        problem.output_height,
        problem.output_width,
    )
}

fn input_gen(_key: &InterpolateAutotuneKey, (input, output_size, options): &Inputs) -> Inputs {
    (input.clone(), *output_size, options.clone())
}
