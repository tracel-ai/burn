use super::ir::{Arg, BinaryElemwiseOp, ElemwiseOp, UnaryElemwiseOp};
use crate::{fusion::JitOptimization, JitRuntime};
use burn_fusion::{OptimizationBuilder, OptimizationProperties, OptimizationStatus};
use burn_tensor::{
    repr::{
        BaseOperationDescription, BinaryOperationDescription, FloatOperationDescription,
        NumericOperationDescription, OperationDescription, ScalarOperationDescription,
        TensorDescription, UnaryOperationDescription,
    },
    DType, Element,
};
use cubecl::ir::Elem;
use cubecl::prelude::Sequence;

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
    pub ops: Sequence<ElemwiseOp>,
    pub t_f32: usize,
    pub t_f16: usize,
    pub t_i32: usize,
    pub t_u32: usize,
    pub s_f32: usize,
    pub s_f16: usize,
    pub s_i32: usize,
    pub s_u32: usize,
}

impl Tracel2Builder {
    pub fn new() -> Self {
        Self {
            t_f32: todo!(),
            t_f16: todo!(),
            t_i32: todo!(),
            t_u32: todo!(),
            s_f32: todo!(),
            s_f16: todo!(),
            s_i32: todo!(),
            s_u32: todo!(),
            ops: Sequence::new(),
        }
    }

    pub fn register_operation(&mut self, op: ElemwiseOp) {}

    pub fn input(&mut self, tensor: &TensorDescription) -> Arg {
        Arg::Input(0, tensor.dtype.into())
    }
    pub fn output(&mut self, tensor: &TensorDescription) -> Arg {
        Arg::Input(0, tensor.dtype.into())
    }
    pub fn scalar<E: Element>(&mut self, tensor: &E, dtype: DType) -> Arg {
        Arg::Input(0, dtype.into())
    }
}
