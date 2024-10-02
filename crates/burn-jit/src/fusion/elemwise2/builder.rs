use std::{borrow::Borrow, collections::BTreeMap};

use super::{
    ir::{
        Arg, BinaryElemwiseOp, ElemwiseOp, FusionArgsLaunch, FusionConfig, OpPrecision,
        UnaryElemwiseOp,
    },
    kernel::elemwise_fuse,
};
use crate::{
    fusion::{JitFusionHandle, JitOptimization},
    JitRuntime,
};
use burn_fusion::{
    stream::Context, OptimizationBuilder, OptimizationProperties, OptimizationStatus,
};
use burn_tensor::{
    repr::{
        BaseOperationDescription, BinaryOperationDescription, FloatOperationDescription,
        NumericOperationDescription, OperationDescription, ScalarOperationDescription,
        TensorDescription, TensorId, TensorStatus, UnaryOperationDescription,
    },
    DType, Element,
};
use cubecl::{
    calculate_cube_count_elemwise,
    prelude::{Sequence, SequenceArg},
    CubeDim,
};
use cubecl::{client::ComputeClient, ir::Elem};

/// Fused element wise operations that are normally memory bound.
pub(crate) struct ElementWise2Builder<R: JitRuntime> {
    builder: Tracel2Builder,
    current_output_shape: Vec<usize>,
    status: OptimizationStatus,
    num_added: usize,
    device: R::Device,
}

impl<R: JitRuntime> OptimizationBuilder<JitOptimization<R>> for ElementWise2Builder<R> {
    fn register(&mut self, ops: &OperationDescription) {
        if let OptimizationStatus::Closed = self.status {
            return;
        }

        match ops {
            OperationDescription::BaseFloat(ops) => {
                if !self.register_base(ops) {
                    self.status = OptimizationStatus::Closed;
                    return;
                }
            }
            OperationDescription::BaseInt(ops) => {
                if !self.register_base(ops) {
                    self.status = OptimizationStatus::Closed;
                    return;
                }
            }
            OperationDescription::Float(_dtype, ops) => {
                if !self.register_float(ops) {
                    self.status = OptimizationStatus::Closed;
                    return;
                }
            }
            OperationDescription::NumericFloat(_dtype, ops) => {
                if !self.register_numeric::<f32>(ops) {
                    self.status = OptimizationStatus::Closed;
                    return;
                }
            }
            OperationDescription::NumericInt(_dtype, ops) => {
                if !self.register_numeric::<i32>(ops) {
                    self.status = OptimizationStatus::Closed;
                    return;
                }
            }
            _ => {
                self.status = OptimizationStatus::Closed;
                return;
            }
        };

        self.status = OptimizationStatus::Open;
        self.num_added += 1;
    }

    fn build(&self) -> JitOptimization<R> {
        todo!();
    }

    fn len(&self) -> usize {
        self.num_added
    }

    fn reset(&mut self) {
        self.num_added = 0;
        self.status = OptimizationStatus::Open;
        self.current_output_shape.clear();
    }

    fn status(&self) -> OptimizationStatus {
        self.status
    }

    fn properties(&self) -> OptimizationProperties {
        let ready = self.num_added > 0;

        OptimizationProperties {
            ready,
            score: self.num_added as u64,
        }
    }
}

impl<R: JitRuntime> ElementWise2Builder<R> {
    pub fn new(device: R::Device) -> Self {
        Self {
            builder: Tracel2Builder::new(),
            num_added: 0,
            current_output_shape: Vec::new(),
            status: OptimizationStatus::Open,
            device,
        }
    }

    fn register_base(&mut self, ops: &BaseOperationDescription) -> bool {
        match ops {
            BaseOperationDescription::Equal(desc) => self
                .register_binary_ops(desc, |lhs, rhs, out| {
                    ElemwiseOp::Equal(BinaryElemwiseOp { lhs, rhs, out })
                }),
            BaseOperationDescription::Cast(desc) => self.register_unary_ops(desc, |input, out| {
                ElemwiseOp::Assign(UnaryElemwiseOp { input, out })
            }),
            _ => false,
        }
    }

    fn register_float(&mut self, ops: &FloatOperationDescription) -> bool {
        match ops {
            FloatOperationDescription::Exp(desc) => self.register_unary_ops(desc, |input, out| {
                ElemwiseOp::Exp(UnaryElemwiseOp { input, out })
            }),
            FloatOperationDescription::Log(desc) => self.register_unary_ops(desc, |input, out| {
                ElemwiseOp::Log(UnaryElemwiseOp { input, out })
            }),
            FloatOperationDescription::Log1p(desc) => self
                .register_unary_ops(desc, |input, out| {
                    ElemwiseOp::Log1p(UnaryElemwiseOp { input, out })
                }),
            FloatOperationDescription::Cos(desc) => self.register_unary_ops(desc, |input, out| {
                ElemwiseOp::Cos(UnaryElemwiseOp { input, out })
            }),
            FloatOperationDescription::Sin(desc) => self.register_unary_ops(desc, |input, out| {
                ElemwiseOp::Sin(UnaryElemwiseOp { input, out })
            }),
            FloatOperationDescription::PowfScalar(desc) => self
                .register_scalar_ops(desc, |lhs, rhs, out| {
                    ElemwiseOp::Powf(BinaryElemwiseOp { lhs, rhs, out })
                }),
            FloatOperationDescription::Tanh(desc) => self.register_unary_ops(desc, |input, out| {
                ElemwiseOp::Tanh(UnaryElemwiseOp { input, out })
            }),
            FloatOperationDescription::Erf(desc) => self.register_unary_ops(desc, |input, out| {
                ElemwiseOp::Erf(UnaryElemwiseOp { input, out })
            }),
            FloatOperationDescription::Recip(desc) => self
                .register_unary_ops(desc, |input, out| {
                    ElemwiseOp::Recip(UnaryElemwiseOp { input, out })
                }),
            _ => false,
        }
    }

    fn register_numeric<E: Element>(&mut self, ops: &NumericOperationDescription<E>) -> bool {
        match ops {
            NumericOperationDescription::Add(desc) => self
                .register_binary_ops(desc, |lhs, rhs, out| {
                    ElemwiseOp::Add(BinaryElemwiseOp { lhs, rhs, out })
                }),
            NumericOperationDescription::AddScalar(desc) => self
                .register_scalar_ops(desc, |lhs, rhs, out| {
                    ElemwiseOp::Add(BinaryElemwiseOp { lhs, rhs, out })
                }),
            NumericOperationDescription::Sub(desc) => self
                .register_binary_ops(desc, |lhs, rhs, out| {
                    ElemwiseOp::Sub(BinaryElemwiseOp { lhs, rhs, out })
                }),
            NumericOperationDescription::SubScalar(desc) => self
                .register_scalar_ops(desc, |lhs, rhs, out| {
                    ElemwiseOp::Sub(BinaryElemwiseOp { lhs, rhs, out })
                }),
            NumericOperationDescription::Mul(desc) => self
                .register_binary_ops(desc, |lhs, rhs, out| {
                    ElemwiseOp::Mul(BinaryElemwiseOp { lhs, rhs, out })
                }),
            NumericOperationDescription::MulScalar(desc) => self
                .register_scalar_ops(desc, |lhs, rhs, out| {
                    ElemwiseOp::Mul(BinaryElemwiseOp { lhs, rhs, out })
                }),
            NumericOperationDescription::Div(desc) => self
                .register_binary_ops(desc, |lhs, rhs, out| {
                    ElemwiseOp::Div(BinaryElemwiseOp { lhs, rhs, out })
                }),
            NumericOperationDescription::DivScalar(desc) => self
                .register_scalar_ops(desc, |lhs, rhs, out| {
                    ElemwiseOp::Div(BinaryElemwiseOp { lhs, rhs, out })
                }),
            NumericOperationDescription::Abs(desc) => self
                .register_unary_ops(desc, |input, out| {
                    ElemwiseOp::Abs(UnaryElemwiseOp { input, out })
                }),
            NumericOperationDescription::Lower(desc) => self
                .register_binary_ops(desc, |lhs, rhs, out| {
                    ElemwiseOp::Lower(BinaryElemwiseOp { lhs, rhs, out })
                }),
            NumericOperationDescription::LowerElem(desc) => self
                .register_scalar_ops(desc, |lhs, rhs, out| {
                    ElemwiseOp::Lower(BinaryElemwiseOp { lhs, rhs, out })
                }),
            NumericOperationDescription::Greater(desc) => self
                .register_binary_ops(desc, |lhs, rhs, out| {
                    ElemwiseOp::Greater(BinaryElemwiseOp { lhs, rhs, out })
                }),
            NumericOperationDescription::GreaterElem(desc) => self
                .register_scalar_ops(desc, |lhs, rhs, out| {
                    ElemwiseOp::Greater(BinaryElemwiseOp { lhs, rhs, out })
                }),
            NumericOperationDescription::LowerEqual(desc) => self
                .register_binary_ops(desc, |lhs, rhs, out| {
                    ElemwiseOp::LowerEqual(BinaryElemwiseOp { lhs, rhs, out })
                }),
            NumericOperationDescription::LowerEqualElem(desc) => self
                .register_scalar_ops(desc, |lhs, rhs, out| {
                    ElemwiseOp::LowerEqual(BinaryElemwiseOp { lhs, rhs, out })
                }),
            NumericOperationDescription::GreaterEqual(desc) => self
                .register_binary_ops(desc, |lhs, rhs, out| {
                    ElemwiseOp::GreaterEqual(BinaryElemwiseOp { lhs, rhs, out })
                }),
            NumericOperationDescription::GreaterEqualElem(desc) => self
                .register_scalar_ops(desc, |lhs, rhs, out| {
                    ElemwiseOp::GreaterEqual(BinaryElemwiseOp { lhs, rhs, out })
                }),
            NumericOperationDescription::EqualElem(desc) => self
                .register_scalar_ops(desc, |lhs, rhs, out| {
                    ElemwiseOp::Equal(BinaryElemwiseOp { lhs, rhs, out })
                }),
            NumericOperationDescription::MaskWhere(desc) => {
                if !self.output_is_compatible(&desc.out) {
                    return false;
                }

                let cond = self.builder.input(&desc.mask);
                let lhs = self.builder.input(&desc.value);
                let rhs = self.builder.input(&desc.tensor);
                let out = self.builder.output(&desc.out);

                self.builder
                    .register_operation(ElemwiseOp::ConditionalAssign {
                        cond,
                        lhs,
                        rhs,
                        out,
                    });

                true
            }
            NumericOperationDescription::MaskFill(desc) => {
                if !self.output_is_compatible(&desc.out) {
                    return false;
                }

                let cond = self.builder.input(&desc.mask);
                let lhs = self.builder.scalar(&desc.value, desc.out.dtype.into());
                let rhs = self.builder.input(&desc.tensor);
                let out = self.builder.output(&desc.out);

                self.builder
                    .register_operation(ElemwiseOp::ConditionalAssign {
                        cond,
                        lhs,
                        rhs,
                        out,
                    });

                true
            }
            NumericOperationDescription::Ones(desc) => {
                if !self.output_is_compatible(desc) {
                    return false;
                }

                let elem: Elem = desc.dtype.into();
                let precision = elem.into();
                let input = Arg::Literal(1, precision);
                let out = self.builder.output(desc);

                self.builder
                    .register_operation(ElemwiseOp::Assign(UnaryElemwiseOp { input, out }));

                true
            }
            NumericOperationDescription::Zeros(desc) => {
                if !self.output_is_compatible(desc) {
                    return false;
                }

                let elem: Elem = desc.dtype.into();
                let precision = elem.into();
                let input = Arg::Literal(0, precision);
                let out = self.builder.output(desc);

                self.builder
                    .register_operation(ElemwiseOp::Assign(UnaryElemwiseOp { input, out }));

                true
            }
            NumericOperationDescription::Full((desc, elem)) => {
                if !self.output_is_compatible(desc) {
                    return false;
                }

                let input = self.builder.scalar(elem, desc.dtype.into());
                let out = self.builder.output(desc);

                self.builder
                    .register_operation(ElemwiseOp::Assign(UnaryElemwiseOp { input, out }));

                true
            }
            _ => false,
        }
    }

    fn register_binary_ops<Func>(&mut self, desc: &BinaryOperationDescription, func: Func) -> bool
    where
        Func: Fn(Arg, Arg, Arg) -> ElemwiseOp,
    {
        if !self.output_is_compatible(&desc.out) {
            return false;
        }

        let lhs = self.builder.input(&desc.lhs);
        let rhs = self.builder.input(&desc.rhs);
        let out = self.builder.output(&desc.out);

        self.builder.register_operation(func(lhs, rhs, out));

        true
    }

    fn register_unary_ops<Func>(&mut self, desc: &UnaryOperationDescription, func: Func) -> bool
    where
        Func: Fn(Arg, Arg) -> ElemwiseOp,
    {
        if !self.output_is_compatible(&desc.out) {
            return false;
        }

        let input = self.builder.input(&desc.input);
        let out = self.builder.output(&desc.out);

        self.builder.register_operation(func(input, out));

        true
    }

    fn register_scalar_ops<Func, E: Element>(
        &mut self,
        desc: &ScalarOperationDescription<E>,
        func: Func,
    ) -> bool
    where
        Func: Fn(Arg, Arg, Arg) -> ElemwiseOp,
    {
        if !self.output_is_compatible(&desc.out) {
            return false;
        }

        let elem = desc.lhs.dtype.into();
        let lhs = self.builder.input(&desc.lhs);
        let rhs = self.builder.scalar(&desc.rhs, elem);
        let out = self.builder.output(&desc.out);

        self.builder.register_operation(func(lhs, rhs, out));

        true
    }

    fn output_is_compatible(&mut self, out: &TensorDescription) -> bool {
        if self.current_output_shape.is_empty() {
            self.current_output_shape.clone_from(&out.shape);
        } else if self.current_output_shape != out.shape {
            return false;
        }

        true
    }
}

pub struct Tracel2Builder {
    pub locals: Tensor2Index,
    pub outputs: Index2Tensor,
    pub inputs: Index2Tensor,
    pub ops: Sequence<ElemwiseOp>,
}

pub type LocalIndex = u32;

#[derive(Default)]
pub struct Tensor2Index {
    pub t_f32: BTreeMap<TensorId, u32>,
    pub t_f16: BTreeMap<TensorId, u32>,
    pub t_bf16: BTreeMap<TensorId, u32>,
    pub t_i32: BTreeMap<TensorId, u32>,
    pub t_u32: BTreeMap<TensorId, u32>,
}

#[derive(Default)]
pub struct Index2Tensor {
    pub t_f32: BTreeMap<u32, TensorId>,
    pub t_f16: BTreeMap<u32, TensorId>,
    pub t_bf16: BTreeMap<u32, TensorId>,
    pub t_i32: BTreeMap<u32, TensorId>,
    pub t_u32: BTreeMap<u32, TensorId>,
}

impl Tracel2Builder {
    pub fn new() -> Self {
        Self {
            locals: Tensor2Index::default(),
            outputs: Index2Tensor::default(),
            inputs: Index2Tensor::default(),
            ops: Sequence::new(),
        }
    }

    pub fn register_operation(&mut self, op: ElemwiseOp) {
        self.ops.push(op);
    }

    pub fn input(&mut self, tensor: &TensorDescription) -> Arg {
        let precision = tensor.dtype.into();

        match precision {
            OpPrecision::F32 => match self.locals.t_f32.get(&tensor.id) {
                Some(val) => Arg::Local(*val, precision),
                None => {
                    let new_input = self.inputs.t_f32.len() as u32;
                    let new_local = self.locals.t_f32.len() as u32;

                    let input = Arg::Input(new_input, precision);
                    let out = Arg::Local(new_local, precision);

                    self.ops
                        .push(ElemwiseOp::Assign(UnaryElemwiseOp { input, out }));

                    self.inputs.t_f32.insert(new_input, tensor.id);
                    self.locals.t_f32.insert(tensor.id, new_local);

                    out
                }
            },
            OpPrecision::F16 => todo!(),
            OpPrecision::BF16 => todo!(),
            OpPrecision::I32 => todo!(),
            OpPrecision::I8 => todo!(),
            OpPrecision::U32 => todo!(),
            OpPrecision::U8 => todo!(),
            OpPrecision::Bool => todo!(),
        }
    }

    pub fn output(&mut self, tensor: &TensorDescription) -> Arg {
        let precision = tensor.dtype.into();

        match precision {
            OpPrecision::F32 => match self.locals.t_f32.get(&tensor.id) {
                Some(val) => Arg::Local(*val, precision),
                None => {
                    let new_local = self.locals.t_f32.len() as u32;
                    let out = Arg::Local(new_local, precision);

                    self.locals.t_f32.insert(tensor.id, new_local);
                    self.outputs.t_f32.insert(new_local, tensor.id);

                    out
                }
            },
            OpPrecision::F16 => todo!(),
            OpPrecision::BF16 => todo!(),
            OpPrecision::I32 => todo!(),
            OpPrecision::I8 => todo!(),
            OpPrecision::U32 => todo!(),
            OpPrecision::U8 => todo!(),
            OpPrecision::Bool => todo!(),
        }
    }

    pub fn scalar<E: Element>(&mut self, tensor: &E, dtype: DType) -> Arg {
        Arg::Scalar(0, dtype.into())
    }

    pub fn run<'a, R: JitRuntime, L: Launch>(
        &self,
        client: &ComputeClient<R::Server, R::Channel>,
        launch: L,
        vectorization: u8,
        context: &mut Context<'a, JitFusionHandle<R>>,
    ) {
        let mut handles = Vec::new();
        let mut inputs = FusionArgsLaunch::new(
            SequenceArg::new(),
            SequenceArg::new(),
            SequenceArg::new(),
            SequenceArg::new(),
            SequenceArg::new(),
            SequenceArg::new(),
            SequenceArg::new(),
            SequenceArg::new(),
        );
        let mut outputs = FusionArgsLaunch::new(
            SequenceArg::new(),
            SequenceArg::new(),
            SequenceArg::new(),
            SequenceArg::new(),
            SequenceArg::new(),
            SequenceArg::new(),
            SequenceArg::new(),
            SequenceArg::new(),
        );

        for (_index, tensor_id) in self.inputs.t_f32.iter() {
            let desc = context.tensors.get(tensor_id).unwrap();
            let handle = context
                .handles
                .get_handle(&tensor_id, &TensorStatus::ReadOnly);

            handles.push(InputHandles::F32(handle, desc.shape.clone()));
        }

        let config = FusionConfig {
            rank: 6,
            ref_layout: super::ir::RefLayout {
                arg: Arg::Output(0, OpPrecision::F32),
            },
            ops: self.ops.clone(),
        };

        // Register everything
        for item in handles.iter() {
            match item {
                InputHandles::F32(handle, shape) => {
                    let arg = handle.as_tensor_arg(shape, vectorization);

                    inputs.t_f32.push(arg);
                }
            }
        }

        launch.run(client, inputs, outputs, config)
    }
}

pub trait Launch {
    fn run<'a, R: JitRuntime>(
        self,
        client: &ComputeClient<R::Server, R::Channel>,
        inputs: FusionArgsLaunch<'a, R>,
        outputs: FusionArgsLaunch<'a, R>,
        config: FusionConfig,
    );
}

struct ElemwiseKernel {
    shape_max: Vec<usize>,
}

impl Launch for ElemwiseKernel {
    fn run<'a, R: JitRuntime>(
        self,
        client: &ComputeClient<R::Server, R::Channel>,
        inputs: FusionArgsLaunch<'a, R>,
        outputs: FusionArgsLaunch<'a, R>,
        config: FusionConfig,
    ) {
        let total_elem = self.shape_max.iter().product();
        let cube_dim = CubeDim::default();
        let cube_count = calculate_cube_count_elemwise(total_elem, cube_dim);

        unsafe {
            elemwise_fuse::launch_unchecked(client, cube_count, cube_dim, inputs, outputs, config)
        }
    }
}

pub enum InputHandles<R: JitRuntime> {
    F32(JitFusionHandle<R>, Vec<usize>),
}
