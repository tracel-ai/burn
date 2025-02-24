use cubecl::{linalg::matmul::components::global::args::MatmulArgs, prelude::*};

use crate::on_write::{
    io::{global_rank, global_shape, global_stride, read_input},
    ir::{Arg, ElemwiseConfig, GlobalArgs, GlobalArgsExpand, LayoutInfo},
    kernel::fuse_on_write,
};

#[derive(Clone)]
pub struct FusedMatmulArgs;

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

#[cube]
impl MatmulArgs for FusedMatmulArgs {
    type Output<EG: Numeric> = GlobalArgs;
    type Input<EG: Numeric> = FusedMatmulInput;
    type State<EG: Numeric> = FusedMatmulState;

    fn init_state<EG: Numeric>(
        inputs: &Self::Input<EG>,
        outputs: &mut Self::Output<EG>,
    ) -> Self::State<EG> {
        FusedMatmulState::new(inputs, outputs, &inputs.config)
    }

    fn read_lhs<EG: Numeric>(state: &Self::State<EG>, coordinate: u32) -> Line<EG> {
        let pos = comptime! {
            match state.lhs {
                Arg::Input(pos, ..) => pos,
                _ => panic!("Lhs isn't an input"),
            }
        };

        read_input(
            unsafe { &(*state.inputs) },
            unsafe { &(*state.outputs) },
            pos,
            coordinate,
            LayoutInfo::IsRef,
            &state.config,
            None,
        )
    }

    fn read_rhs<EG: Numeric>(state: &Self::State<EG>, coordinate: u32) -> Line<EG> {
        let pos = comptime! {
            match state.rhs {
                Arg::Input(pos, ..) => pos,
                _ => panic!("Lhs isn't an input"),
            }
        };

        read_input(
            unsafe { &(*state.inputs) },
            unsafe { &(*state.outputs) },
            pos,
            coordinate,
            LayoutInfo::IsRef,
            &state.config,
            None,
        )
    }

    fn read_window_lhs<EG: Numeric>(
        _state: &Self::State<EG>,
        _start: u32,
        _end: u32,
    ) -> Slice<Line<EG>> {
        comptime!(todo!());
        // TODO This is a dummy return value to satisfy the type checker
        //      before working on an implementation.
        //      Remove the allow annotation after implementing this function.
        #[allow(unreachable_code)]
        SharedMemory::new_lined(0, 0_u32).to_slice()
    }

    #[allow(unreachable_code)]
    fn read_window_rhs<EG: Numeric>(
        _state: &Self::State<EG>,
        _start: u32,
        _end: u32,
    ) -> Slice<Line<EG>> {
        comptime!(todo!());
        // TODO This is a dummy return value to satisfy the type checker
        //      before working on an implementation.
        //      Remove the allow annotation after implementing this function.
        #[allow(unreachable_code)]
        SharedMemory::new_lined(0, 0_u32).to_slice()
    }

    fn write_out<EG: Numeric>(state: &mut Self::State<EG>, coordinate: u32, value: Line<EG>) {
        let mut values = Registry::<Arg, Line<EG>>::new();
        let mut args = comptime![Sequence::<Arg>::new()];

        values.insert(comptime![state.out.clone()], value);
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

    fn rank_lhs<EG: Numeric>(state: &Self::State<EG>) -> u32 {
        let pos = comptime! {
            match state.lhs {
                Arg::Input(pos, ..) => pos,
                _ => panic!("Lhs isn't an input"),
            }
        };

        global_rank(unsafe { &(*state.inputs) }, pos)
    }

    fn rank_rhs<EG: Numeric>(state: &Self::State<EG>) -> u32 {
        let pos = comptime! {
            match state.rhs {
                Arg::Input(pos, ..) => pos,
                _ => panic!("Rhs isn't an input"),
            }
        };

        global_rank(unsafe { &(*state.inputs) }, pos)
    }

    fn rank_out<EG: Numeric>(state: &Self::State<EG>) -> u32 {
        let (pos, is_input) = comptime! {
            match state.config.ref_layout {
                Arg::Input(pos, ..) => (pos, true),
                Arg::Output(pos, ..) => (pos, false),
                _ => panic!("Out isn't an input or output"),
            }
        };

        if is_input {
            global_rank(unsafe { &(*state.inputs) }, pos)
        } else {
            global_rank(unsafe { &(*state.outputs) }, pos)
        }
    }

    fn shape_lhs<EG: Numeric>(state: &Self::State<EG>, dim: u32) -> u32 {
        let pos = comptime! {
            match state.lhs {
                Arg::Input(pos, ..) => pos,
                _ => panic!("Lhs isn't an input"),
            }
        };

        global_shape(unsafe { &(*state.inputs) }, dim, pos)
    }

    fn shape_rhs<EG: Numeric>(state: &Self::State<EG>, dim: u32) -> u32 {
        let pos = comptime! {
            match state.rhs {
                Arg::Input(pos, ..) => pos,
                _ => panic!("Rhs isn't an input"),
            }
        };

        global_shape(unsafe { &(*state.inputs) }, dim, pos)
    }

    fn shape_out<EG: Numeric>(state: &Self::State<EG>, dim: u32) -> u32 {
        let (pos, is_input) = comptime! {
            match state.config.ref_layout {
                Arg::Input(pos, ..) => (pos, true),
                Arg::Output(pos, ..) => (pos, false),
                _ => panic!("Out isn't an input or output"),
            }
        };

        if is_input {
            global_shape(unsafe { &(*state.inputs) }, dim, pos)
        } else {
            global_shape(unsafe { &(*state.outputs) }, dim, pos)
        }
    }

    fn stride_lhs<EG: Numeric>(state: &Self::State<EG>, dim: u32) -> u32 {
        let pos = comptime! {
            match state.lhs {
                Arg::Input(pos, ..) => pos,
                _ => panic!("Lhs isn't an input"),
            }
        };

        global_stride(unsafe { &(*state.inputs) }, dim, pos)
    }

    fn stride_rhs<EG: Numeric>(state: &Self::State<EG>, dim: u32) -> u32 {
        let pos = comptime! {
            match state.rhs {
                Arg::Input(pos, ..) => pos,
                _ => panic!("Rhs isn't an input"),
            }
        };

        global_stride(unsafe { &(*state.inputs) }, dim, pos)
    }

    fn stride_out<EG: Numeric>(state: &Self::State<EG>, dim: u32) -> u32 {
        let (pos, is_input) = comptime! {
            match state.config.ref_layout {
                Arg::Input(pos, ..) => (pos, true),
                Arg::Output(pos, ..) => (pos, false),
                _ => panic!("Out isn't an input or output"),
            }
        };

        if is_input {
            global_stride(unsafe { &(*state.inputs) }, dim, pos)
        } else {
            global_stride(unsafe { &(*state.outputs) }, dim, pos)
        }
    }
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
            lhs: comptime![inputs.lhs.clone()],
            rhs: comptime![inputs.rhs.clone()],
            out: comptime![inputs.out.clone()],
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
    fn init(self, _context: &mut Scope) -> Self {
        self
    }
}

impl CubeDebug for FusedMatmulStateExpand {}
