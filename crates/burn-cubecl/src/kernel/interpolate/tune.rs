use crate::{
    CubeRuntime, CubeTuneId, kernel::interpolate::execute_interpolate, tensor::CubeTensor,
};
use burn_backend::cubecl::dtype_to_elem_type;
use burn_backend::ops::InterpolateOptions;
use cubecl::tune::{LocalTuner, Tunable, TunableSet, TuneGroup, local_tuner};
use cubek::interpolate::{
    definition::TileSize,
    launch::{InterpolateAutotuneKey, InterpolateStrategy},
    routines::{
        BlueprintStrategy, GlobalMemoryRoutine, GlobalMemoryStrategy, SharedMemoryRoutine,
        SharedMemoryStrategy,
    },
};

/// Interpolate operation with autotuning. This benchmarks multiple strategies and selects the best one at runtime.
pub fn interpolate_autotune<R: CubeRuntime>(
    input: CubeTensor<R>,
    output_size: [usize; 2],
    options: InterpolateOptions,
) -> CubeTensor<R> {
    let client = input.client.clone();

    static TUNER: LocalTuner<InterpolateAutotuneKey, CubeTuneId> = local_tuner!();

    let tunables = TUNER.init(|| {
        const PRIORITY: i8 = 0;

        let global_memory =
            TuneGroup::<InterpolateAutotuneKey>::new("global_memory", |_key| PRIORITY);
        let shared_memory =
            TuneGroup::<InterpolateAutotuneKey>::new("shared_memory", |_key| PRIORITY);

        let mut set = TunableSet::new(create_key::<R>, input_gen::<R>);

        let tile_sizes: [TileSize; 16] = [
            // Square shapes
            TileSize::new(8, 8),
            TileSize::new(16, 16),
            TileSize::new(32, 32),
            // Rectangular shapes
            TileSize::new(8, 16),
            TileSize::new(16, 8),
            TileSize::new(16, 32),
            TileSize::new(32, 16),
            // Flat horizontal shapes
            TileSize::new(1, 64),
            TileSize::new(1, 128),
            TileSize::new(1, 256),
            TileSize::new(2, 128),
            TileSize::new(1, 512),
            TileSize::new(1, 1024),
            // Flat vertical shapes
            TileSize::new(64, 1),
            TileSize::new(128, 1),
            TileSize::new(256, 1),
        ];

        for tile_size in tile_sizes {
            let name = format!(
                "global_memory_tile_size_{}_{}",
                tile_size.height(),
                tile_size.width()
            );
            set = set.with(
                Tunable::new(&name, move |(input, output_size, options)| {
                    execute_interpolate::<R>(
                        input,
                        output_size,
                        options,
                        InterpolateStrategy::GlobalMemoryStrategy(BlueprintStrategy::<
                            GlobalMemoryRoutine,
                        >::Inferred(
                            GlobalMemoryStrategy { tile_size },
                        )),
                    )
                })
                .group(&global_memory, |_key| PRIORITY),
            );

            let name = format!(
                "shared_memory_tile_size_{}_{}",
                tile_size.height(),
                tile_size.width()
            );
            set = set.with(
                Tunable::new(&name, move |(input, output_size, options)| {
                    execute_interpolate::<R>(
                        input,
                        output_size,
                        options,
                        InterpolateStrategy::SharedMemoryStrategy(BlueprintStrategy::<
                            SharedMemoryRoutine,
                        >::Inferred(
                            SharedMemoryStrategy { tile_size },
                        )),
                    )
                })
                .group(&shared_memory, |_key| PRIORITY),
            );
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
    (input, output_size, _options): &(CubeTensor<R>, [usize; 2], InterpolateOptions),
) -> InterpolateAutotuneKey {
    let elem_input = dtype_to_elem_type(input.dtype);
    let elem_output = dtype_to_elem_type(input.dtype);

    InterpolateAutotuneKey::generate(elem_input, elem_output, input.meta.shape(), output_size)
}

fn input_gen<R: CubeRuntime>(
    _key: &InterpolateAutotuneKey,
    (input, output_size, options): &(CubeTensor<R>, [usize; 2], InterpolateOptions),
) -> (CubeTensor<R>, [usize; 2], InterpolateOptions) {
    (input.clone(), *output_size, options.clone())
}
