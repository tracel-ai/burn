use cubecl::{linalg::matmul::components::global::args::MatmulArgs, prelude::*};

use crate::shared::{
    io::{
        global_buffer_len, global_len, global_rank, global_shape, global_stride, num_elements,
        read_input, ref_buffer_len, ref_len, ref_rank, ref_shape, ref_stride,
    },
    ir::{
        Arg, ElemwiseConfig, GlobalArgs, GlobalArgsExpand, LayoutInfo, LocalArgs, LocalArgsExpand,
    },
    kernel::{fuse_on_write, init_locals},
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
        let mut locals = init_locals(&inputs.global, outputs, &inputs.config);
        FusedMatmulState::new(inputs, outputs, &mut locals, &inputs.config)
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
            unsafe { &(*state.locals) },
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
            unsafe { &(*state.locals) },
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
            unsafe { &mut (*state.locals) },
            coordinate,
            values,
            args,
            &state.config,
        );
    }

    fn len_lhs<EG: Numeric>(state: &Self::State<EG>) -> u32 {
        let pos = comptime! {
            match state.lhs {
                Arg::Input(pos, ..) => pos,
                _ => panic!("Lhs isn't an input"),
            }
        };

        global_len(unsafe { &(*state.inputs) }, pos)
    }

    fn len_rhs<EG: Numeric>(state: &Self::State<EG>) -> u32 {
        let pos = comptime! {
            match state.rhs {
                Arg::Input(pos, ..) => pos,
                _ => panic!("Rhs isn't an input"),
            }
        };

        global_len(unsafe { &(*state.inputs) }, pos)
    }

    fn len_out<EG: Numeric>(state: &Self::State<EG>) -> u32 {
        ref_len(
            unsafe { &(*state.inputs) },
            unsafe { &(*state.outputs) },
            unsafe { &(*state.locals) },
            &state.config,
        )
    }

    fn buffer_len_lhs<EG: Numeric>(state: &Self::State<EG>) -> u32 {
        match comptime![state.lhs.clone()] {
            Arg::Input(pos, ..) => global_buffer_len(unsafe { &(*state.inputs) }, pos),
            Arg::InputReshaped { .. } => num_elements(unsafe { &(*state.locals) }, &state.config),
            _ => panic!("Lhs isn't an input"),
        }
    }

    fn buffer_len_rhs<EG: Numeric>(state: &Self::State<EG>) -> u32 {
        match comptime![state.rhs.clone()] {
            Arg::Input(pos, ..) => global_len(unsafe { &(*state.inputs) }, pos),
            Arg::InputReshaped { .. } => num_elements(unsafe { &(*state.locals) }, &state.config),
            _ => panic!("Lhs isn't an input"),
        }
    }

    fn buffer_len_out<EG: Numeric>(state: &Self::State<EG>) -> u32 {
        ref_buffer_len(
            unsafe { &(*state.inputs) },
            unsafe { &(*state.outputs) },
            unsafe { &(*state.locals) },
            &state.config,
        )
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
        ref_rank(
            unsafe { &(*state.inputs) },
            unsafe { &(*state.outputs) },
            &state.config,
        )
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
        ref_shape(unsafe { &(*state.locals) }, dim)
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
        ref_stride(unsafe { &(*state.locals) }, dim)
    }
}

pub struct FusedMatmulState {
    inputs: *const GlobalArgs,
    outputs: *mut GlobalArgs,
    locals: *mut LocalArgs,
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
        locals: &mut LocalArgs,
        #[comptime] config: &ElemwiseConfig,
    ) -> FusedMatmulState {
        FusedMatmulState {
            inputs: &inputs.global,
            outputs,
            config: comptime![config.clone()],
            locals,
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
    locals: LocalArgsExpand,
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
