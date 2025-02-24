use cubecl::reduce::args::ReduceArgs;
use cubecl::{prelude::*, reduce::args::ReducePrecision};

use crate::shared::io::{
    global_buffer_len, global_len, global_rank, global_shape, global_stride, write,
};
use crate::shared::ir::{Arg, ElemwiseConfig, GlobalArgs, GlobalArgsExpand, LocalArgs};
use crate::shared::kernel::fuse_on_read;

#[derive(Clone)]
pub struct FusedReduceArgs;

#[derive(CubeLaunch)]
pub struct FusedReduceInput {
    global: GlobalArgs,
    #[cube(comptime)]
    config: ElemwiseConfig,
    #[cube(comptime)]
    arg: Arg,
}

#[derive(CubeLaunch)]
pub struct FusedReduceOutput {
    global: GlobalArgs,
    #[cube(comptime)]
    config: ElemwiseConfig,
    #[cube(comptime)]
    arg: Arg,
}

pub struct FusedReduceState {
    inputs: *const GlobalArgs,
    outputs: *mut GlobalArgs,
    config_on_write: ElemwiseConfig,
    config_on_read: ElemwiseConfig,
    input: Arg,
    out: Arg,
}

#[derive(Clone)]
pub struct FusedReduceStateExpand {
    inputs: GlobalArgsExpand,
    outputs: GlobalArgsExpand,
    config_on_write: ElemwiseConfig,
    config_on_read: ElemwiseConfig,
    input: Arg,
    out: Arg,
}

#[cube]
impl ReduceArgs for FusedReduceArgs {
    type Input<E: Numeric> = FusedReduceInput;
    type Output<E: Numeric> = FusedReduceOutput;
    type State<P: ReducePrecision> = FusedReduceState;

    fn init_state<P: ReducePrecision>(
        input: &Self::Input<P::In>,
        output: &mut Self::Output<P::Out>,
    ) -> Self::State<P> {
        FusedReduceState::new(input, output)
    }

    fn read_input<P: ReducePrecision>(state: &Self::State<P>, index: u32) -> Line<P::In> {
        *fuse_on_read::<P::In>(
            unsafe { &(*state.inputs) },
            unsafe { &mut (*state.outputs) },
            index,
            comptime! {
                let mut sequence = Sequence::new();
                sequence.push(state.input.clone());
                sequence
            },
            &state.config_on_read,
        )
        .index(0)
    }

    fn read_output<P: ReducePrecision>(state: &Self::State<P>, index: u32) -> Line<P::Out> {
        Line::empty(1)
    }

    fn write_output<P: ReducePrecision>(
        state: &mut Self::State<P>,
        index: u32,
        value: Line<P::Out>,
    ) {
        let out = comptime![state.out.clone()];
        let mut local = LocalArgs::new();

        write::<P::Out>(
            unsafe { &(*state.inputs) },
            unsafe { &mut (*state.outputs) },
            &mut local,
            index,
            value,
            out,
            &state.config_on_read,
        );
    }

    fn len_input<P: ReducePrecision>(state: &Self::State<P>) -> u32 {
        let (pos, is_input) = comptime! {
            match state.config_on_read.ref_layout {
                Arg::Input(pos, ..) => (pos, true),
                Arg::Output(pos, ..) => (pos, false),
                _ => panic!("It isn't an input or output"),
            }
        };

        if is_input {
            global_len(unsafe { &(*state.inputs) }, pos)
        } else {
            global_len(unsafe { &(*state.outputs) }, pos)
        }
    }

    fn len_output<P: ReducePrecision>(state: &Self::State<P>) -> u32 {
        let (pos, is_input) = comptime! {
            match state.config_on_write.ref_layout {
                Arg::Input(pos, ..) => (pos, true),
                Arg::Output(pos, ..) => (pos, false),
                _ => panic!("Out isn't an input or output"),
            }
        };

        if is_input {
            global_len(unsafe { &(*state.inputs) }, pos)
        } else {
            global_len(unsafe { &(*state.outputs) }, pos)
        }
    }

    fn buffer_len_input<P: ReducePrecision>(state: &Self::State<P>) -> u32 {
        let (pos, is_input) = comptime! {
            match state.config_on_read.ref_layout {
                Arg::Input(pos, ..) => (pos, true),
                Arg::Output(pos, ..) => (pos, false),
                _ => panic!("It isn't an input or output"),
            }
        };

        if is_input {
            global_buffer_len(unsafe { &(*state.inputs) }, pos)
        } else {
            global_buffer_len(unsafe { &(*state.outputs) }, pos)
        }
    }

    fn buffer_len_output<P: ReducePrecision>(state: &Self::State<P>) -> u32 {
        let (pos, is_input) = comptime! {
            match state.config_on_write.ref_layout {
                Arg::Input(pos, ..) => (pos, true),
                Arg::Output(pos, ..) => (pos, false),
                _ => panic!("It isn't an input or output"),
            }
        };

        if is_input {
            global_buffer_len(unsafe { &(*state.inputs) }, pos)
        } else {
            global_buffer_len(unsafe { &(*state.outputs) }, pos)
        }
    }

    fn rank_input<P: ReducePrecision>(state: &Self::State<P>) -> u32 {
        let (pos, is_input) = comptime! {
            match state.config_on_read.ref_layout {
                Arg::Input(pos, ..) => (pos, true),
                Arg::Output(pos, ..) => (pos, false),
                _ => panic!("It isn't an input or output"),
            }
        };

        if is_input {
            global_rank(unsafe { &(*state.inputs) }, pos)
        } else {
            global_rank(unsafe { &(*state.outputs) }, pos)
        }
    }

    fn rank_output<P: ReducePrecision>(state: &Self::State<P>) -> u32 {
        let (pos, is_input) = comptime! {
            match state.config_on_write.ref_layout {
                Arg::Input(pos, ..) => (pos, true),
                Arg::Output(pos, ..) => (pos, false),
                _ => panic!("It isn't an input or output"),
            }
        };

        if is_input {
            global_rank(unsafe { &(*state.inputs) }, pos)
        } else {
            global_rank(unsafe { &(*state.outputs) }, pos)
        }
    }

    fn shape_input<P: ReducePrecision>(state: &Self::State<P>, dim: u32) -> u32 {
        let (pos, is_input) = comptime! {
            match state.config_on_read.ref_layout {
                Arg::Input(pos, ..) => (pos, true),
                Arg::Output(pos, ..) => (pos, false),
                _ => panic!("It isn't an input or output"),
            }
        };

        if is_input {
            global_shape(unsafe { &(*state.inputs) }, dim, pos)
        } else {
            global_shape(unsafe { &(*state.outputs) }, dim, pos)
        }
    }

    fn shape_output<P: ReducePrecision>(state: &Self::State<P>, dim: u32) -> u32 {
        let (pos, is_input) = comptime! {
            match state.config_on_write.ref_layout {
                Arg::Input(pos, ..) => (pos, true),
                Arg::Output(pos, ..) => (pos, false),
                _ => panic!("It isn't an input or output"),
            }
        };

        if is_input {
            global_shape(unsafe { &(*state.inputs) }, dim, pos)
        } else {
            global_shape(unsafe { &(*state.outputs) }, dim, pos)
        }
    }

    fn stride_input<P: ReducePrecision>(state: &Self::State<P>, dim: u32) -> u32 {
        let (pos, is_input) = comptime! {
            match state.config_on_read.ref_layout {
                Arg::Input(pos, ..) => (pos, true),
                Arg::Output(pos, ..) => (pos, false),
                _ => panic!("It isn't an input or output"),
            }
        };

        if is_input {
            global_stride(unsafe { &(*state.inputs) }, dim, pos)
        } else {
            global_stride(unsafe { &(*state.outputs) }, dim, pos)
        }
    }

    fn stride_output<P: ReducePrecision>(state: &Self::State<P>, dim: u32) -> u32 {
        let (pos, is_input) = comptime! {
            match state.config_on_write.ref_layout {
                Arg::Input(pos, ..) => (pos, true),
                Arg::Output(pos, ..) => (pos, false),
                _ => panic!("It isn't an input or output"),
            }
        };

        if is_input {
            global_stride(unsafe { &(*state.inputs) }, dim, pos)
        } else {
            global_stride(unsafe { &(*state.outputs) }, dim, pos)
        }
    }
}

#[cube]
impl FusedReduceState {
    pub fn new(inputs: &FusedReduceInput, outputs: &mut FusedReduceOutput) -> FusedReduceState {
        FusedReduceState {
            inputs: &inputs.global,
            outputs: &mut outputs.global,
            config_on_read: comptime![inputs.config.clone()],
            config_on_write: comptime![outputs.config.clone()],
            input: comptime![inputs.arg.clone()],
            out: comptime![outputs.arg.clone()],
        }
    }
}

impl CubeType for FusedReduceState {
    type ExpandType = FusedReduceStateExpand;
}

impl Init for FusedReduceStateExpand {
    fn init(self, _context: &mut Scope) -> Self {
        self
    }
}

impl CubeDebug for FusedReduceStateExpand {}
