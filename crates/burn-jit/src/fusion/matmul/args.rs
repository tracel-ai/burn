use cubecl::{linalg::matmul::components::global::args::MatmulArgs, prelude::*};

use crate::fusion::on_write::{
    io::{global_rank, global_shape, global_stride, read_input},
    ir::{Arg, ElemwiseConfig, GlobalArgs, GlobalArgsExpand, LayoutInfo},
    kernel::fuse_on_write,
};

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

#[derive(Clone)]
pub struct FusedMatmulArgs;

#[cube]
impl<EG: Numeric> MatmulArgs<EG> for FusedMatmulArgs {
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

    fn rank_lhs(state: &Self::State) -> u32 {
        let (pos, precision) = comptime! {
            match state.lhs {
                Arg::Input(pos, precision, _) => (pos.clone(), precision.clone()),
                _ => panic!("Lhs isn't an input"),
            }
        };

        global_rank(unsafe { &(*state.inputs) }, pos, precision)
    }

    fn rank_rhs(state: &Self::State) -> u32 {
        let (pos, precision) = comptime! {
            match state.rhs {
                Arg::Input(pos, precision, _) => (pos.clone(), precision.clone()),
                _ => panic!("Rhs isn't an input"),
            }
        };

        global_rank(unsafe { &(*state.inputs) }, pos, precision)
    }

    fn rank_out(state: &Self::State) -> u32 {
        let (pos, precision, is_input) = comptime! {
            match state.config.ref_layout {
                Arg::Input(pos, precision, _) => (pos.clone(), precision.clone(), true),
                Arg::Output(pos, precision, _) => (pos.clone(), precision.clone(), false),
                _ => panic!("Out isn't an input or output"),
            }
        };

        if is_input {
            global_rank(unsafe { &(*state.inputs) }, pos, precision)
        } else {
            global_rank(unsafe { &(*state.outputs) }, pos, precision)
        }
    }

    fn shape_lhs(state: &Self::State, dim: u32) -> u32 {
        let (pos, precision) = comptime! {
            match state.lhs {
                Arg::Input(pos, precision, _) => (pos.clone(), precision.clone()),
                _ => panic!("Lhs isn't an input"),
            }
        };

        global_shape(unsafe { &(*state.inputs) }, dim, pos, precision)
    }

    fn shape_rhs(state: &Self::State, dim: u32) -> u32 {
        let (pos, precision) = comptime! {
            match state.rhs {
                Arg::Input(pos, precision, _) => (pos.clone(), precision.clone()),
                _ => panic!("Rhs isn't an input"),
            }
        };

        global_shape(unsafe { &(*state.inputs) }, dim, pos, precision)
    }

    fn shape_out(state: &Self::State, dim: u32) -> u32 {
        let (pos, precision, is_input) = comptime! {
            match state.config.ref_layout {
                Arg::Input(pos, precision, _) => (pos.clone(), precision.clone(), true),
                Arg::Output(pos, precision, _) => (pos.clone(), precision.clone(), false),
                _ => panic!("Out isn't an input or output"),
            }
        };

        if is_input {
            global_shape(unsafe { &(*state.inputs) }, dim, pos, precision)
        } else {
            global_shape(unsafe { &(*state.outputs) }, dim, pos, precision)
        }
    }

    fn stride_lhs(state: &Self::State, dim: u32) -> u32 {
        let (pos, precision) = comptime! {
            match state.lhs {
                Arg::Input(pos, precision, _) => (pos.clone(), precision.clone()),
                _ => panic!("Lhs isn't an input"),
            }
        };

        global_stride(unsafe { &(*state.inputs) }, dim, pos, precision)
    }

    fn stride_rhs(state: &Self::State, dim: u32) -> u32 {
        let (pos, precision) = comptime! {
            match state.rhs {
                Arg::Input(pos, precision, _) => (pos.clone(), precision.clone()),
                _ => panic!("Rhs isn't an input"),
            }
        };

        global_stride(unsafe { &(*state.inputs) }, dim, pos, precision)
    }

    fn stride_out(state: &Self::State, dim: u32) -> u32 {
        let (pos, precision, is_input) = comptime! {
            match state.config.ref_layout {
                Arg::Input(pos, precision, _) => (pos.clone(), precision.clone(), true),
                Arg::Output(pos, precision, _) => (pos.clone(), precision.clone(), false),
                _ => panic!("Out isn't an input or output"),
            }
        };

        if is_input {
            global_stride(unsafe { &(*state.inputs) }, dim, pos, precision)
        } else {
            global_stride(unsafe { &(*state.outputs) }, dim, pos, precision)
        }
    }
}
