use cubecl::{
    matmul::components::{
        MatmulIdent,
        global::{
            GlobalConfig,
            args::MatmulArgs,
            memory::{BatchedGlobalLayout, GlobalMemoryConfig},
        },
    },
    prelude::*,
    std::{
        CubeOption, FastDivmod,
        tensor::{
            View,
            layout::{Coords1d, Coords3d},
        },
    },
};

use crate::shared::{
    ir::{Arg, FuseBlockConfig, GlobalArgs, LocalArgs},
    kernel::init_locals,
    view::{FusedOutput, GlobalInput},
};

#[derive(Clone)]
pub struct FusedMatmulArgs;

#[derive(CubeLaunch, CubeType)]
pub struct FusedMatmulInput {
    global: GlobalArgs,
    #[cube(comptime)]
    config: FuseBlockConfig,
    #[cube(comptime)]
    a: Arg,
    #[cube(comptime)]
    b: Arg,
    #[cube(comptime)]
    c: CubeOption<Arg>,
    #[cube(comptime)]
    out: Arg,
}

#[cube]
impl MatmulArgs for FusedMatmulArgs {
    type Output<EO: Numeric> = GlobalArgs;
    type Input<Lhs: Numeric, Rhs: Numeric, EO: Numeric> = FusedMatmulInput;
    type State<Lhs: Numeric, Rhs: Numeric, EO: Numeric> = FusedMatmulState;

    fn init_state<Lhs: Numeric, Rhs: Numeric, EO: Numeric, G: GlobalConfig>(
        inputs: &Self::Input<Lhs, Rhs, EO>,
        outputs: &mut Self::Output<EO>,
        #[comptime] config: G,
    ) -> Self::State<Lhs, Rhs, EO> {
        let mut locals = init_locals(&inputs.global, outputs, &inputs.config);
        let rank = comptime![inputs.config.rank];
        let mut batch_shape = Sequence::new();

        #[unroll]
        for i in 0..rank - 2 {
            batch_shape.push(FastDivmod::new_Fallback(locals.ref_shape[i]));
        }

        FusedMatmulState::new(
            inputs,
            outputs,
            &mut locals,
            batch_shape,
            &inputs.config,
            config.global_memory_config(MatmulIdent::Lhs),
            config.global_memory_config(MatmulIdent::Rhs),
            config.global_memory_config(MatmulIdent::Out),
        )
    }

    fn view_lhs<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &Self::State<Lhs, Rhs, EO>,
    ) -> View<Line<Lhs>, Coords3d> {
        global_view(
            &state.inputs,
            &state.locals,
            state.batch_shape.clone(),
            comptime![state.a.clone()],
            comptime![state.config.clone()],
            comptime![state.lhs_memory_config],
        )
    }

    fn view_rhs<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &Self::State<Lhs, Rhs, EO>,
    ) -> View<Line<Rhs>, Coords3d> {
        global_view(
            &state.inputs,
            &state.locals,
            state.batch_shape.clone(),
            comptime![state.b.clone()],
            comptime![state.config.clone()],
            comptime![state.rhs_memory_config],
        )
    }

    fn view_acc<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &Self::State<Lhs, Rhs, EO>,
    ) -> CubeOption<View<Line<EO>, Coords3d>> {
        match comptime![state.c.clone()] {
            CubeOption::Some(c) => {
                let view = global_view(
                    &state.inputs,
                    &state.locals,
                    state.batch_shape.clone(),
                    c,
                    comptime![state.config.clone()],
                    comptime![state.out_memory_config],
                );
                CubeOption::new_Some(view)
            }
            CubeOption::None => CubeOption::new_None(),
        }
    }

    fn view_out<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &mut Self::State<Lhs, Rhs, EO>,
    ) -> View<Line<EO>, Coords3d, ReadWrite> {
        let rank = comptime![state.config.rank];

        let mut batch_strides = Sequence::new();
        #[unroll]
        for i in 0..rank - 2 {
            batch_strides.push(state.locals.ref_strides[i]);
        }

        let shape_row = state.locals.ref_shape[rank - 2];
        let shape_col = state.locals.ref_shape[rank - 1];

        let stride_row = state.locals.ref_strides[rank - 2];
        let stride_col = state.locals.ref_strides[rank - 1];

        let layout = BatchedGlobalLayout::new(
            batch_strides,
            state.batch_shape.clone(),
            shape_row,
            shape_col,
            stride_row,
            stride_col,
            state.out_memory_config,
        );
        let mut buffer = FusedOutput::new(
            &state.inputs,
            &mut state.outputs,
            &mut state.locals,
            comptime![state.out.clone()],
            comptime![state.config.clone()],
        );
        View::new_mut::<FusedOutput, Coords1d>(&mut buffer, layout)
    }
}

#[cube]
fn global_view<E: CubePrimitive>(
    inputs: &GlobalArgs,
    locals: &LocalArgs,
    batch_shape: Sequence<FastDivmod>,
    #[comptime] arg: Arg,
    #[comptime] config: FuseBlockConfig,
    #[comptime] mem_config: GlobalMemoryConfig,
) -> View<Line<E>, Coords3d> {
    let rank = comptime![config.rank];
    let lhs = match comptime![arg.clone()] {
        Arg::Input(pos, ..) => inputs.tensors.index(pos),
        _ => panic!("Input must be concrete"),
    };

    let mut batch_strides = Sequence::new();
    #[unroll]
    for i in 0..rank - 2 {
        let shape = lhs.tensor.shape(i);
        let stride = select(shape == 1, 0, lhs.tensor.stride(i));
        batch_strides.push(stride);
    }

    let shape_row = lhs.tensor.shape(rank - 2);
    let shape_col = lhs.tensor.shape(rank - 1);

    let stride_row = lhs.tensor.stride(rank - 2);
    let stride_col = lhs.tensor.stride(rank - 1);

    let layout = BatchedGlobalLayout::new(
        batch_strides,
        batch_shape.clone(),
        shape_row,
        shape_col,
        stride_row,
        stride_col,
        mem_config,
    );
    let buffer = GlobalInput::new(inputs, locals, arg, comptime![config.clone()], None);
    View::new::<GlobalInput, Coords1d>(&buffer, layout)
}

#[derive(CubeType)]
pub struct FusedMatmulState {
    inputs: GlobalArgs,
    outputs: GlobalArgs,
    locals: LocalArgs,
    #[cube(comptime)]
    config: FuseBlockConfig,
    #[cube(comptime)]
    a: Arg,
    #[cube(comptime)]
    b: Arg,
    #[cube(comptime)]
    c: CubeOption<Arg>,
    #[cube(comptime)]
    out: Arg,
    #[cube(comptime)]
    lhs_memory_config: GlobalMemoryConfig,
    #[cube(comptime)]
    rhs_memory_config: GlobalMemoryConfig,
    #[cube(comptime)]
    out_memory_config: GlobalMemoryConfig,
    batch_shape: Sequence<FastDivmod>,
}

#[cube]
impl FusedMatmulState {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        inputs: &FusedMatmulInput,
        outputs: &mut GlobalArgs,
        locals: &mut LocalArgs,
        batch_shape: Sequence<FastDivmod>,
        #[comptime] config: &FuseBlockConfig,
        #[comptime] lhs_memory_config: GlobalMemoryConfig,
        #[comptime] rhs_memory_config: GlobalMemoryConfig,
        #[comptime] out_memory_config: GlobalMemoryConfig,
    ) -> FusedMatmulState {
        FusedMatmulState {
            inputs: inputs.global.clone(),
            outputs: outputs.clone(),
            config: comptime![config.clone()],
            locals: locals.clone(),
            a: comptime![inputs.a.clone()],
            b: comptime![inputs.b.clone()],
            c: comptime![inputs.c.clone()],
            out: comptime![inputs.out.clone()],
            batch_shape,
            lhs_memory_config,
            rhs_memory_config,
            out_memory_config,
        }
    }
}
