use crate::engine::codegen::{
    io::ref_len,
    ir::{FuseArg, FuseBlockConfig, GlobalArgs, GlobalArgsLaunch, multi_block_variables_init},
    kernel::init_locals,
    view::FusedInput,
};
use cubecl::{
    prelude::*,
    std::tensor::{
        ErasedTensor, View,
        layout::{Coords1d, plain::PlainLayout},
    },
    tile::{
        Partitioning, Tile, TileSpec,
        kind::GlobalOperand,
        launch::{TileOperand, Unbound},
        layout::RuntimeGeometry,
    },
};

/// The trace's read block as a tile routine's input: each line the routine reads runs the block,
/// which may write the trace's materialized intermediates, and yields the block's value for the
/// routine's input, at the width the block computes its lines in.
pub struct TraceReadTileOperand;

/// The trace's arguments, the block that runs at each read and the slot it yields the input from,
/// and the geometry the routine settled for its input ([`Unbound`]), as runtime values so a new
/// shape compiles nothing.
#[derive(CubeType, CubeLaunch)]
pub struct TraceReadArg {
    pub inputs: GlobalArgs,
    pub outputs: GlobalArgs,
    pub shape: Sequence<u32>,
    pub strides: Sequence<u32>,
    #[cube(comptime)]
    pub rank: usize,
    #[cube(comptime)]
    pub spec: TileSpec,
    #[cube(comptime)]
    pub input: FuseArg,
    #[cube(comptime)]
    pub config: FuseBlockConfig,
}

impl TraceReadArgLaunch {
    /// The argument for an input the routine bound as `source`, read through `config`'s block
    /// from the trace's `input` slot.
    pub fn of(
        inputs: GlobalArgsLaunch,
        outputs: GlobalArgsLaunch,
        source: Unbound,
        input: FuseArg,
        config: FuseBlockConfig,
    ) -> Self {
        let Unbound { spec, geometry, .. } = source;
        TraceReadArgLaunch::new(
            inputs,
            outputs,
            geometry.shape().iter().map(|&e| e as u32).collect(),
            geometry.strides().iter().map(|&s| s as u32).collect(),
            geometry.rank(),
            spec,
            input,
            config,
        )
    }
}

#[cube]
impl TileOperand for TraceReadTileOperand {
    type Arg = TraceReadArg;

    fn tile<E: Numeric>(arg: &TraceReadArg, partitioning: &Partitioning) -> Tile {
        let config = comptime!(arg.config.clone());
        let mut outputs = arg.outputs.clone();
        multi_block_variables_init(&config, &mut outputs.variables);
        let mut locals = init_locals(&arg.inputs, &mut outputs, &config);
        let lines = ref_len(&arg.inputs, &outputs, &locals, &config);
        let block = FusedInput::new(
            &arg.inputs,
            &mut outputs,
            &mut locals,
            comptime!(arg.input.clone()),
            config.clone(),
        );
        let size!(W) = config.width;
        let view = View::new::<FusedInput, Coords1d>(block, PlainLayout::new(lines));
        let source = ErasedTensor::<E, ReadOnly>::of_view::<View<Vector<E, W>, Coords1d>, W>(view);

        let mut geometry = RuntimeGeometry::new();
        #[unroll]
        for axis in 0..comptime!(arg.rank) {
            geometry.push(*arg.shape.index(axis), *arg.strides.index(axis));
        }
        GlobalOperand::source(
            source,
            geometry,
            config.width,
            comptime!(partitioning.space().clone()),
            comptime!(arg.spec.clone()),
        )
        .tile::<E>(comptime!(partitioning.levels().to_vec()))
    }
}
