use crate::engine::codegen::{
    io::ref_len,
    ir::{FuseArg, FuseBlockConfig, GlobalArgs, GlobalArgsLaunch, multi_block_variables_init},
    kernel::init_locals,
    view::FusedOutput,
};
use cubecl::{
    prelude::*,
    std::tensor::{
        ErasedTensor, ViewMut, WriteOnly,
        layout::{Coords1d, plain::PlainLayout},
    },
    tile::{
        Geometry, Partitioning, Tile, TileSpec,
        kind::{GlobalOperand, Write},
        launch::{Destination, DestinationLaunch, Unbound},
        layout::RuntimeGeometry,
    },
};

/// The trace's element-wise block as a tile routine's output: each line the routine stores runs
/// the block, which writes the fused outputs, instead of landing at an address.
///
/// A sink is never read, so only a routine that finishes each cell before storing it once can take
/// one; cubek declines the others when it prepares the launch.
pub struct TraceSink;

/// The trace's arguments, the block that runs at the store and the slot it reads the product from,
/// and the product's geometry, which a sink has no address to answer for.
///
/// The geometry is the one the routine settled when it bound its output ([`Unbound`]), as runtime
/// values so a new shape compiles nothing.
#[derive(CubeType, CubeLaunch)]
pub struct TraceSinkArg {
    inputs: GlobalArgs,
    outputs: GlobalArgs,
    shape: Sequence<u32>,
    strides: Sequence<u32>,
    #[cube(comptime)]
    rank: usize,
    #[cube(comptime)]
    spec: TileSpec,
    #[cube(comptime)]
    product: FuseArg,
    #[cube(comptime)]
    config: FuseBlockConfig,
}

impl TraceSinkArgLaunch {
    /// The argument for a product the routine bound as `sink`, stored through `config`'s block
    /// into the trace's `product` slot.
    pub fn of(
        inputs: GlobalArgsLaunch,
        outputs: GlobalArgsLaunch,
        sink: Unbound,
        product: FuseArg,
        config: FuseBlockConfig,
    ) -> Self {
        let Unbound { spec, geometry, .. } = sink;
        let (shape, strides) = runtime_geometry(&geometry);
        TraceSinkArgLaunch::new(
            inputs,
            outputs,
            shape,
            strides,
            geometry.rank(),
            spec,
            product,
            config,
        )
    }
}

fn runtime_geometry(geometry: &Geometry) -> (SequenceArg<u32>, SequenceArg<u32>) {
    (
        geometry.shape().iter().map(|&e| e as u32).collect(),
        geometry.strides().iter().map(|&s| s as u32).collect(),
    )
}

#[cube]
impl Destination for TraceSink {
    type Arg<E: Numeric> = TraceSinkArg;

    fn tile<E: Numeric, V: Size>(arg: &TraceSinkArg, partitioning: &Partitioning) -> Tile {
        let config = comptime!(arg.config.clone());
        let mut outputs = arg.outputs.clone();
        multi_block_variables_init(&config, &mut outputs.variables);
        let mut locals = init_locals(&arg.inputs, &mut outputs, &config);
        let lines = ref_len(&arg.inputs, &outputs, &locals, &config);
        let block = FusedOutput::new(
            &arg.inputs,
            &mut outputs,
            &mut locals,
            comptime!(arg.product.clone()),
            config,
        );
        let view = ViewMut::new::<FusedOutput, Coords1d>(block, PlainLayout::new(lines));
        let sink =
            ErasedTensor::<E, WriteOnly>::of_view_mut::<ViewMut<Vector<E, V>, Coords1d>, V>(view);

        let mut geometry = RuntimeGeometry::new();
        #[unroll]
        for axis in 0..comptime!(arg.rank) {
            geometry.push(*arg.shape.index(axis), *arg.strides.index(axis));
        }
        // Read at expansion: a launch-time `Size` has no value as a Rust constant.
        let width = V::value();
        GlobalOperand::sink(
            sink,
            geometry,
            width,
            comptime!(partitioning.space().clone()),
            comptime!(arg.spec.clone()),
            Write::Replace,
        )
        .tile::<E>(comptime!(partitioning.levels().to_vec()))
    }
}

impl DestinationLaunch for TraceSink {
    type Operand = TraceSinkArgLaunch;

    fn arg<E: Numeric>(operand: TraceSinkArgLaunch) -> TraceSinkArgLaunch {
        operand
    }
}
