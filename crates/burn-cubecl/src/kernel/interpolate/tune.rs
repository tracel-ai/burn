use crate::{
    CubeRuntime, CubeTuneId,
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
    client::ComputeClient,
    std::throughput::roofline_bounds,
    tune::{LocalTuner, Tunable, TunableSet, TuneGroup, local_tuner},
};
use cubek::interpolate::{
    InterpolateConfig, Residence,
    definition::{InterpolateCost, InterpolateForwardProblem, InterpolateProblem},
    tune_key::InterpolateAutotuneKey,
};

type Inputs<R> = (CubeTensor<R>, [usize; 2], InterpolateOptions);

const PRIORITY_CANDIDATE: i8 = 1;
const PRIORITY_FALLBACK: i8 = 0;
const PRIORITY_NEVER: i8 = -1;

/// The geometry box candidates are drawn from, which is cubek's `BenchTier::Extensive` GPU
/// catalogue: every geometry a recorded sweep has won with lies inside it. Its lighter tier caps
/// planes and rows at eight, which leaves out the deep row runs a wide GPU wants.
const MAX_PLANES_PER_CUBE: usize = 16;
const MAX_ROWS_PER_PLANE: usize = 64;

/// Cap on `planes * rows`, the output rows a cube holds live. Past this corner of the box is
/// register pressure the device refuses at launch, so there is nothing there to measure.
const MAX_ROWS_PER_CUBE: usize = 256;

/// The geometry the search fans out from, and the one a non-autotune build runs.
const DEFAULT_PLANES_PER_CUBE: usize = 4;
const DEFAULT_ROWS_PER_PLANE: usize = 2;

/// How far from the predicted geometry a candidate is still measured, in powers of two per axis.
const NEIGHBOURHOOD: u32 = 1;

/// Interpolate operation with autotuning. This benchmarks multiple configurations and selects the
/// best one at runtime.
pub fn interpolate_autotune<R: CubeRuntime>(
    input: CubeTensor<R>,
    output_size: [usize; 2],
    options: InterpolateOptions,
) -> CubeTensor<R> {
    let client = input.client.clone();

    static TUNER: LocalTuner<InterpolateAutotuneKey, CubeTuneId> = local_tuner!();

    let device = Device::of(&client);

    let tunables = TUNER.init(move || {
        let candidates =
            TuneGroup::<InterpolateAutotuneKey>::new("candidates", |_key| PRIORITY_CANDIDATE);
        // Reached only when the candidate group prunes or fails everything, which a small enough
        // output can do: every geometry then walks past the output it was meant to cover.
        let fallback =
            TuneGroup::<InterpolateAutotuneKey>::new("fallback", |_key| PRIORITY_FALLBACK);

        let mut set = with_bounds(TunableSet::new(create_key::<R>, input_gen::<R>));

        for config in candidate_configs() {
            let name = config_name(config);
            let mut tunable = Tunable::new(&name, move |(input, output_size, options)| {
                execute_interpolate::<R>(input, output_size, options, config)
            })
            .group(&candidates, move |key| {
                match worth_measuring(key, config, device) {
                    true => PRIORITY_CANDIDATE,
                    false => PRIORITY_NEVER,
                }
            });

            if config == FALLBACK_CONFIG {
                tunable = tunable.group(&fallback, |_key| PRIORITY_FALLBACK);
            }

            set = set.with(tunable);
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

/// A cube of one plane walking one output row: legal on every device, for every output.
const FALLBACK_CONFIG: InterpolateConfig = InterpolateConfig::new(Residence::InPlace, 1, 1, 1);

/// The device facts the candidate filter reads.
///
/// Sampled once, when the tunable set is built, which happens once per process: a second device
/// in the same process reuses the first one's set. The matmul tuner captures its device facts the
/// same way.
#[derive(Clone, Copy)]
struct Device {
    lanes: usize,
    max_units_per_cube: usize,
}

impl Device {
    fn of<R: CubeRuntime>(client: &ComputeClient<R>) -> Self {
        let hardware = &client.properties().hardware;

        Self {
            lanes: (hardware.plane_size_max as usize).max(1),
            max_units_per_cube: hardware.max_units_per_cube as usize,
        }
    }
}

/// Every configuration the tuner can pick from, in the order its first pass walks them.
///
/// Registration order is that walking order and it cannot vary per key, so it fans out from the
/// default geometry rather than from a corner of the box: the short circuit then gets its chance
/// to exit on an early candidate instead of after the whole batch.
fn candidate_configs() -> Vec<InterpolateConfig> {
    let mut geometries: Vec<(usize, usize)> = powers_of_two(MAX_PLANES_PER_CUBE)
        .flat_map(|planes| powers_of_two(MAX_ROWS_PER_PLANE).map(move |rows| (planes, rows)))
        .filter(|(planes, rows)| planes * rows <= MAX_ROWS_PER_CUBE)
        .collect();

    geometries.sort_by_key(|&(planes, rows)| {
        octaves(planes, DEFAULT_PLANES_PER_CUBE) + octaves(rows, DEFAULT_ROWS_PER_PLANE)
    });

    geometries
        .into_iter()
        .flat_map(|(planes, rows)| {
            [Residence::InPlace, Residence::Smem]
                .map(|residence| InterpolateConfig::new(residence, planes, rows, 1))
        })
        .collect()
}

/// Whether `config` is worth measuring for `key`.
///
/// A cube that walks past the output it covers, and a neighbourhood around the predicted
/// geometry. The neighbourhood is what keeps the measured set small: the box holds dozens of
/// geometries, and sweeping it from a corner spends the whole tuning budget before reaching the
/// ones that matter.
///
/// What a device refuses is not repeated here. cubek returns those before any launch work, and
/// autotune reads an `Err` as "drop this candidate", so the rule stays in the one place that
/// owns it.
fn worth_measuring(
    key: &InterpolateAutotuneKey,
    config: InterpolateConfig,
    device: Device,
) -> bool {
    // The key's output height is anchored up, so this drops a geometry only when it overruns a
    // problem at least as large as the real one: it keeps one that only just fits.
    if rows_per_cube(config) > key.output_height {
        return false;
    }

    let (planes, rows) = predicted_geometry(key, device);
    octaves(config.planes_per_cube, planes) <= NEIGHBOURHOOD
        && octaves(config.rows_per_plane, rows) <= NEIGHBOURHOOD
}

/// The geometry the search centres on, read off the key alone.
///
/// Planes fill the cube, since `planes * lanes` is its unit count and idle units are launch slots
/// paid for and not used. Rows follow the reuse a deeper run can amortize, which is how many
/// output rows draw on the same input rows, so the vertical resampling ratio.
///
/// This is a model, not a measurement, and it is the first thing to replace once cubek publishes
/// recorded sweeps.
fn predicted_geometry(key: &InterpolateAutotuneKey, device: Device) -> (usize, usize) {
    let planes = (device.max_units_per_cube / device.lanes).clamp(1, MAX_PLANES_PER_CUBE);
    let rows = (key.output_height / key.input_height.max(1)).clamp(1, MAX_ROWS_PER_PLANE);

    (planes.next_power_of_two(), rows.next_power_of_two())
}

/// Registers the roofline bounds the short circuit needs.
///
/// Interpolation is memory bound for the cheap filters, but the arithmetic per output element
/// grows with the tap count, and Lanczos3 spends two sines on every weight. Costing both and
/// letting the roofline take whichever is slower is what keeps the limit reachable across modes,
/// rather than holding every mode to a bandwidth figure only nearest can approach.
fn with_bounds<R: CubeRuntime, Out: 'static>(
    set: TunableSet<InterpolateAutotuneKey, Inputs<R>, Out>,
) -> TunableSet<InterpolateAutotuneKey, Inputs<R>, Out> {
    autotune_bounds::with_bounds(
        set,
        |_key, (input, output_size, options): &Inputs<R>, thresholds| {
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
fn forward_problem<R: CubeRuntime>(
    input: &CubeTensor<R>,
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

fn config_name(config: InterpolateConfig) -> String {
    let residence = match config.input_residence {
        Residence::Smem => "smem",
        _ => "in_place",
    };

    format!(
        "{residence}_p{}_r{}_c{}",
        config.planes_per_cube, config.rows_per_plane, config.cols_per_lane
    )
}

fn powers_of_two(max: usize) -> impl Iterator<Item = usize> {
    core::iter::successors(Some(1usize), move |n| (n * 2 <= max).then_some(n * 2))
}

/// Output rows one cube walks, which is every plane's run over every plane it holds.
fn rows_per_cube(config: InterpolateConfig) -> usize {
    config.planes_per_cube * config.rows_per_plane
}

/// How many powers of two apart two extents are.
fn octaves(a: usize, b: usize) -> u32 {
    a.ilog2().abs_diff(b.ilog2())
}

fn create_key<R: CubeRuntime>((input, output_size, options): &Inputs<R>) -> InterpolateAutotuneKey {
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

fn input_gen<R: CubeRuntime>(
    _key: &InterpolateAutotuneKey,
    (input, output_size, options): &Inputs<R>,
) -> Inputs<R> {
    (input.clone(), *output_size, options.clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::ir::{ElemType, FloatKind};
    use cubek::interpolate::definition::InterpolateMode;

    const F32: ElemType = ElemType::Float(FloatKind::F32);

    /// A device with 32-lane planes and 1024 units per cube, so a cube holds 32 planes and the
    /// box's plane cap is what bounds the prediction rather than the hardware.
    const DEVICE: Device = Device {
        lanes: 32,
        max_units_per_cube: 1024,
    };

    fn key(
        input_height: usize,
        output_height: usize,
        output_width: usize,
        channels: usize,
    ) -> InterpolateAutotuneKey {
        InterpolateAutotuneKey::generate(
            F32,
            F32,
            InterpolateMode::Bilinear,
            true,
            input_height,
            input_height,
            channels,
            output_height,
            output_width,
        )
    }

    fn survivors(key: &InterpolateAutotuneKey) -> Vec<InterpolateConfig> {
        candidate_configs()
            .into_iter()
            .filter(|config| worth_measuring(key, *config, DEVICE))
            .collect()
    }

    /// The tuner's first pass walks registration order, so the geometry a non-autotune build
    /// runs has to be the one it reaches first. Anything else and the short circuit gets its
    /// first chance on a candidate nobody picked as a default.
    #[test]
    fn the_default_geometry_is_registered_first() {
        let first = candidate_configs()[0];

        assert_eq!(first.input_residence, Residence::InPlace);
        assert_eq!(first.planes_per_cube, DEFAULT_PLANES_PER_CUBE);
        assert_eq!(first.rows_per_plane, DEFAULT_ROWS_PER_PLANE);
    }

    #[test]
    fn every_candidate_lies_inside_the_box() {
        for config in candidate_configs() {
            assert!(config.planes_per_cube <= MAX_PLANES_PER_CUBE);
            assert!(config.rows_per_plane <= MAX_ROWS_PER_PLANE);
            assert!(config.planes_per_cube * config.rows_per_plane <= MAX_ROWS_PER_CUBE);
            assert!(config.planes_per_cube.is_power_of_two());
            assert!(config.rows_per_plane.is_power_of_two());
        }
    }

    /// The plan falls through to the fallback group when the filter leaves nothing, so the
    /// configuration that group names has to be one the loop actually registers.
    #[test]
    fn the_fallback_configuration_is_registered() {
        assert!(candidate_configs().contains(&FALLBACK_CONFIG));
    }

    /// The filter has to leave something to measure on an ordinary problem, and it has to leave
    /// less than the box holds, which is the whole point of having one.
    #[test]
    fn an_ordinary_problem_keeps_some_candidates_but_not_the_box() {
        let survivors = survivors(&key(256, 512, 512, 64));

        assert!(!survivors.is_empty());
        assert!(survivors.len() < candidate_configs().len());
    }

    /// A cube walking more output rows than the problem has only masks the overhang it walked
    /// into, so nothing that deep is worth a measurement.
    #[test]
    fn a_geometry_deeper_than_the_output_is_dropped() {
        let key = key(4, 8, 8, 64);

        for config in survivors(&key) {
            assert!(rows_per_cube(config) <= key.output_height);
        }
    }

    /// An upsample reuses each input row across the output rows drawn from it, which is the
    /// reuse a deeper row run amortizes, so the centre follows the ratio.
    #[test]
    fn the_predicted_row_run_follows_the_resampling_ratio() {
        assert_eq!(predicted_geometry(&key(64, 256, 256, 64), DEVICE).1, 4);
        assert_eq!(predicted_geometry(&key(64, 64, 64, 64), DEVICE).1, 1);
        // A downsample draws each output row from rows no other output row wants.
        assert_eq!(predicted_geometry(&key(256, 64, 64, 64), DEVICE).1, 1);
    }
}
