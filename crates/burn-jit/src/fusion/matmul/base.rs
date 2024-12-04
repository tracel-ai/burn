use cubecl::prelude::*;

use crate::fusion::on_write::{
    io::{global_shape, global_stride, read_input},
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

    fn shape_lhs(state: &Self::State, dim: u32) -> u32;
    fn shape_rhs(state: &Self::State, dim: u32) -> u32;
    fn shape_out(state: &Self::State, dim: u32) -> u32;

    fn stride_lhs(state: &Self::State, dim: u32) -> u32;
    fn stride_rhs(state: &Self::State, dim: u32) -> u32;
    fn stride_out(state: &Self::State, dim: u32) -> u32;

    fn write_out(state: &mut Self::State, coordinate: u32, value: Line<EG>);
}

#[derive(Clone)]
pub enum Ident {
    Lhs,
    Rhs,
}

pub struct TensorInput<EG: Numeric, GA: GmmArgs<EG>> {
    state: *const GA::State,
    ident: Ident,
}

pub struct TensorInputExpand<EG: Numeric, GA: GmmArgs<EG>> {
    state: <GA::State as CubeType>::ExpandType,
    ident: Ident,
}

impl<EG: Numeric, GA: GmmArgs<EG>> Clone for TensorInputExpand<EG, GA> {
    fn clone(&self) -> Self {
        Self {
            state: self.state.clone(),
            ident: self.ident.clone(),
        }
    }
}

impl<EG: Numeric, GA: GmmArgs<EG>> Init for TensorInputExpand<EG, GA> {
    fn init(mut self, context: &mut CubeContext) -> Self {
        self.state = self.state.init(context);
        self
    }
}

impl<EG: Numeric, GA: GmmArgs<EG>> CubeType for TensorOutput<EG, GA> {
    type ExpandType = TensorOutputExpand<EG, GA>;
}

pub struct TensorOutput<EG: Numeric, GA: GmmArgs<EG>> {
    state: *mut GA::State,
}

pub struct TensorOutputExpand<EG: Numeric, GA: GmmArgs<EG>> {
    state: <GA::State as CubeType>::ExpandType,
}

impl<EG: Numeric, GA: GmmArgs<EG>> Clone for TensorOutputExpand<EG, GA> {
    fn clone(&self) -> Self {
        Self {
            state: self.state.clone(),
        }
    }
}

impl<EG: Numeric, GA: GmmArgs<EG>> Init for TensorOutputExpand<EG, GA> {
    fn init(mut self, context: &mut CubeContext) -> Self {
        self.state = self.state.init(context);
        self
    }
}

impl<EG: Numeric, GA: GmmArgs<EG>> CubeType for TensorInput<EG, GA> {
    type ExpandType = TensorInputExpand<EG, GA>;
}

#[cube]
impl<EG: Numeric, GA: GmmArgs<EG>> TensorInput<EG, GA> {
    pub fn new(state: &GA::State, #[comptime] ident: Ident) -> TensorInput<EG, GA> {
        TensorInput::<EG, GA> { state, ident }
    }

    pub fn read(&self, coordinate: u32) -> Line<EG> {
        unsafe {
            match comptime![&self.ident] {
                Ident::Lhs => GA::read_lhs(&(*self.state), coordinate),
                Ident::Rhs => GA::read_rhs(&(*self.state), coordinate),
            }
        }
    }
    pub fn shape(&self, dim: u32) -> u32 {
        unsafe {
            match comptime![&self.ident] {
                Ident::Lhs => GA::shape_lhs(&(*self.state), dim),
                Ident::Rhs => GA::shape_rhs(&(*self.state), dim),
            }
        }
    }
    pub fn stride(&self, dim: u32) -> u32 {
        unsafe {
            match comptime![&self.ident] {
                Ident::Lhs => GA::stride_lhs(&(*self.state), dim),
                Ident::Rhs => GA::stride_rhs(&(*self.state), dim),
            }
        }
    }
}

#[cube]
impl<EG: Numeric, GA: GmmArgs<EG>> TensorOutput<EG, GA> {
    pub fn new(state: &mut GA::State) -> TensorOutput<EG, GA> {
        TensorOutput::<EG, GA> { state }
    }

    pub fn write(&self, coordinate: u32, value: Line<EG>) {
        unsafe { GA::write_out(&mut (*self.state), coordinate, value) }
    }
    pub fn shape(&self, dim: u32) -> u32 {
        unsafe { GA::shape_out(&mut (*self.state), dim) }
    }
    pub fn stride(&self, dim: u32) -> u32 {
        unsafe { GA::stride_out(&mut (*self.state), dim) }
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

#[derive(CubeLaunch)]
pub struct TensorInputs<EG: Numeric> {
    pub lhs: Tensor<Line<EG>>,
    pub rhs: Tensor<Line<EG>>,
}

#[cube]
impl<EG: Numeric> GmmArgs<EG> for Tensor<Line<EG>> {
    type Output = Tensor<Line<EG>>;
    type Input = TensorInputs<EG>;
    type State = (
        *const Tensor<Line<EG>>,
        *const Tensor<Line<EG>>,
        *mut Tensor<Line<EG>>,
    );

    fn init_state(input: &Self::Input, output: &mut Self::Output) -> Self::State {
        (&input.lhs, &input.rhs, output)
    }

    fn read_lhs(state: &Self::State, coordinate: u32) -> Line<EG> {
        unsafe { (*state.0)[coordinate] }
    }

    fn read_rhs(state: &Self::State, coordinate: u32) -> Line<EG> {
        unsafe { (*state.1)[coordinate] }
    }

    fn shape_lhs(state: &Self::State, dim: u32) -> u32 {
        unsafe { (*state.0).shape(dim) }
    }

    fn shape_rhs(state: &Self::State, dim: u32) -> u32 {
        unsafe { (*state.1).shape(dim) }
    }

    fn shape_out(state: &Self::State, dim: u32) -> u32 {
        unsafe { (*state.2).shape(dim) }
    }

    fn stride_lhs(state: &Self::State, dim: u32) -> u32 {
        unsafe { (*state.0).stride(dim) }
    }

    fn stride_rhs(state: &Self::State, dim: u32) -> u32 {
        unsafe { (*state.1).stride(dim) }
    }

    fn stride_out(state: &Self::State, dim: u32) -> u32 {
        unsafe { (*state.2).stride(dim) }
    }

    fn write_out(state: &mut Self::State, coordinate: u32, value: Line<EG>) {
        unsafe { (*state.2)[coordinate] = value }
    }
}
