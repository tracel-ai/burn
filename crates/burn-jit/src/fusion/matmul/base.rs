use cubecl::prelude::*;

use crate::fusion::on_write::{
    io::read_input,
    ir::{Arg, ElemwiseConfig, GlobalArgs, GlobalArgsExpand, LayoutInfo},
    kernel::fuse_on_write,
};

#[cube]
pub trait GmmArgs<EG: Numeric> {
    type Output: LaunchArg + CubeType;
    type Input: LaunchArg + CubeType;

    type State: CubeType;

    fn init_state(input: &Self::Input, output: &mut Self::Output) -> Self::State;
    fn read_lhs(state: &Self::State, coordinate: u32) -> Line<EG>;
    fn read_rhs(state: &Self::State, coordinate: u32) -> Line<EG>;

    fn write_out(state: &mut Self::State, coordinate: u32, value: Line<EG>);
}

pub struct FusedMatmulState {
    inputs: *const GlobalArgs,
    outputs: *mut GlobalArgs,
    config: ElemwiseConfig,
    lhs: Arg,
    rhs: Arg,
    out: Arg,
}

#[cube]
impl FusedMatmulState {
    pub fn new(
        inputs: &FusedMatmulInput,
        outputs: &mut GlobalArgs,
        #[comptime] config: &ElemwiseConfig,
    ) -> FusedMatmulState {
        FusedMatmulState {
            inputs: &inputs.global,
            outputs,
            config: comptime![config.clone()],
            lhs: comptime![inputs.lhs],
            rhs: comptime![inputs.rhs],
            out: comptime![inputs.out],
        }
    }
}

#[derive(Clone)]
pub struct FusedMatmulStateExpand {
    inputs: GlobalArgsExpand,
    outputs: GlobalArgsExpand,
    config: ElemwiseConfig,
    lhs: Arg,
    rhs: Arg,
    out: Arg,
}

impl CubeType for FusedMatmulState {
    type ExpandType = FusedMatmulStateExpand;
}

impl Init for FusedMatmulStateExpand {
    fn init(self, _context: &mut CubeContext) -> Self {
        self
    }
}

#[derive(CubeLaunch)]
pub struct FusedMatmulInput {
    global: GlobalArgs,
    #[cube(comptime)]
    config: ElemwiseConfig,
    #[cube(comptime)]
    lhs: Arg,
    #[cube(comptime)]
    rhs: Arg,
    #[cube(comptime)]
    out: Arg,
}

pub struct FusedMatmulArgs;

#[cube]
impl<EG: Numeric> GmmArgs<EG> for FusedMatmulArgs {
    type Output = GlobalArgs;
    type Input = FusedMatmulInput;
    type State = FusedMatmulState;

    fn init_state(inputs: &Self::Input, outputs: &mut Self::Output) -> Self::State {
        FusedMatmulState::new(inputs, outputs, &inputs.config)
    }

    fn read_lhs(state: &Self::State, coordinate: u32) -> Line<EG> {
        let (pos, precision) = comptime! {
            match state.lhs {
                Arg::Input(pos, precision, _) => (pos.clone(), precision.clone()),
                _ => panic!("Lhs isn't an input"),
            }
        };

        read_input(
            unsafe { &(*state.inputs) },
            unsafe { &(*state.outputs) },
            pos,
            coordinate,
            LayoutInfo::IsRef,
            precision,
            &state.config,
        )
    }

    fn read_rhs(state: &Self::State, coordinate: u32) -> Line<EG> {
        let (pos, precision) = comptime! {
            match state.rhs {
                Arg::Input(pos, precision, _) => (pos.clone(), precision.clone()),
                _ => panic!("Lhs isn't an input"),
            }
        };

        read_input(
            unsafe { &(*state.inputs) },
            unsafe { &(*state.outputs) },
            pos,
            coordinate,
            LayoutInfo::IsRef,
            precision,
            &state.config,
        )
    }

    fn write_out(state: &mut Self::State, coordinate: u32, value: Line<EG>) {
        let mut values = Registry::<Arg, Line<EG>>::new();
        let mut args = comptime![Sequence::<Arg>::new()];

        values.insert(state.out, value);
        comptime![args.push(state.out.clone())];

        fuse_on_write(
            unsafe { &(*state.inputs) },
            unsafe { &mut (*state.outputs) },
            coordinate,
            values,
            args,
            &state.config,
        );
    }
}
