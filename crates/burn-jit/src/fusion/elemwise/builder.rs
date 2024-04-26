use core::marker::PhantomData;

use super::{optimization::ElementWise, CompilationPhase};
use crate::{
    codegen::dialect::gpu::{
        BinaryOperator, ConditionalAssign, Elem, Operator, Procedure, UnaryOperator, Variable,
    },
    element::JitElement,
    fusion::{tracing::TraceBuilder, JitOptimization},
    FloatElement, IntElement, JitBackend, Runtime,
};
use burn_fusion::{OptimizationBuilder, OptimizationProperties, OptimizationStatus};
use burn_tensor::{
    ops::{FloatElem, IntElem},
    repr::{
        BaseOperationDescription, BinaryOperationDescription, FloatOperationDescription,
        NumericOperationDescription, OperationDescription, ScalarOperationDescription,
        TensorDescription, UnaryOperationDescription,
    },
    Device, Element,
};

/// Fused element wise operations that are normally memory bound.
pub(crate) struct ElementWiseBuilder<R: Runtime, F: FloatElement, I: IntElement> {
    builder: TraceBuilder,
    current_output_shape: Vec<usize>,
    status: OptimizationStatus,
    num_added: usize,
    device: R::Device,
    _float_elem: PhantomData<F>,
    _int_elem: PhantomData<I>,
}

impl<R, F, I> OptimizationBuilder<JitOptimization<R>> for ElementWiseBuilder<R, F, I>
where
    R: Runtime,
    F: FloatElement,
    I: IntElement,
{
    fn register(&mut self, ops: &OperationDescription) {
        if let OptimizationStatus::Closed = self.status {
            return;
        }

        match ops {
            OperationDescription::BaseFloat(ops) => {
                if !self.register_base::<FloatElem<JitBackend<R, F, I>>>(ops) {
                    self.status = OptimizationStatus::Closed;
                    return;
                }
            }
            OperationDescription::BaseInt(ops) => {
                if !self.register_base::<IntElem<JitBackend<R, F, I>>>(ops) {
                    self.status = OptimizationStatus::Closed;
                    return;
                }
            }
            OperationDescription::Float(ops) => {
                if !self.register_float::<FloatElem<JitBackend<R, F, I>>>(ops) {
                    self.status = OptimizationStatus::Closed;
                    return;
                }
            }
            OperationDescription::NumericFloat(ops) => {
                if !self.register_numeric::<FloatElem<JitBackend<R, F, I>>, _>(ops) {
                    self.status = OptimizationStatus::Closed;
                    return;
                }
            }
            OperationDescription::NumericInt(ops) => {
                if !self.register_numeric::<IntElem<JitBackend<R, F, I>>, _>(ops) {
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
        let op = ElementWise::new(
            self.builder.clone().build(),
            self.num_added,
            self.device.clone(),
            CompilationPhase,
        );

        JitOptimization::ElementWise(op.compile())
    }

    fn len(&self) -> usize {
        self.num_added
    }

    fn reset(&mut self) {
        self.builder = TraceBuilder::new();
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

impl<R: Runtime, F: FloatElement, I: IntElement> ElementWiseBuilder<R, F, I> {
    pub fn new(device: Device<JitBackend<R, F, I>>) -> Self {
        Self {
            builder: TraceBuilder::new(),
            num_added: 0,
            current_output_shape: Vec::new(),
            status: OptimizationStatus::Open,
            device,
            _float_elem: PhantomData,
            _int_elem: PhantomData,
        }
    }

    fn register_base<E: JitElement>(&mut self, ops: &BaseOperationDescription) -> bool {
        match ops {
            BaseOperationDescription::Equal(desc) => self.register_binary_ops(
                desc,
                (E::gpu_elem(), E::gpu_elem(), Elem::Bool),
                |lhs, rhs, out| Operator::Equal(BinaryOperator { lhs, rhs, out }),
            ),
            _ => false,
        }
    }

    fn register_float<E: JitElement>(&mut self, ops: &FloatOperationDescription) -> bool {
        match ops {
            FloatOperationDescription::Exp(desc) => {
                self.register_unary_ops(desc, (E::gpu_elem(), E::gpu_elem()), |input, out| {
                    Operator::Exp(UnaryOperator { input, out })
                })
            }
            FloatOperationDescription::Log(desc) => {
                self.register_unary_ops(desc, (E::gpu_elem(), E::gpu_elem()), |input, out| {
                    Operator::Log(UnaryOperator { input, out })
                })
            }
            FloatOperationDescription::Log1p(desc) => {
                self.register_unary_ops(desc, (E::gpu_elem(), E::gpu_elem()), |input, out| {
                    Operator::Log1p(UnaryOperator { input, out })
                })
            }
            FloatOperationDescription::Cos(desc) => {
                self.register_unary_ops(desc, (E::gpu_elem(), E::gpu_elem()), |input, out| {
                    Operator::Cos(UnaryOperator { input, out })
                })
            }
            FloatOperationDescription::Sin(desc) => {
                self.register_unary_ops(desc, (E::gpu_elem(), E::gpu_elem()), |input, out| {
                    Operator::Sin(UnaryOperator { input, out })
                })
            }
            FloatOperationDescription::PowfScalar(desc) => self.register_scalar_ops(
                desc,
                (E::gpu_elem(), E::gpu_elem(), E::gpu_elem()),
                |lhs, rhs, out| Operator::Powf(BinaryOperator { lhs, rhs, out }),
            ),
            FloatOperationDescription::Tanh(desc) => {
                self.register_unary_ops(desc, (E::gpu_elem(), E::gpu_elem()), |input, out| {
                    Operator::Tanh(UnaryOperator { input, out })
                })
            }
            FloatOperationDescription::Erf(desc) => {
                self.register_unary_ops(desc, (E::gpu_elem(), E::gpu_elem()), |input, out| {
                    Operator::Erf(UnaryOperator { input, out })
                })
            }
            FloatOperationDescription::Recip(desc) => {
                self.register_unary_ops(desc, (E::gpu_elem(), E::gpu_elem()), |input, out| {
                    Operator::Recip(UnaryOperator { input, out })
                })
            }
            _ => false,
        }
    }

    fn register_numeric<E: JitElement, EDesc: JitElement>(
        &mut self,
        ops: &NumericOperationDescription<EDesc>,
    ) -> bool {
        match ops {
            NumericOperationDescription::Add(desc) => self.register_binary_ops(
                desc,
                (E::gpu_elem(), E::gpu_elem(), E::gpu_elem()),
                |lhs, rhs, out| Operator::Add(BinaryOperator { lhs, rhs, out }),
            ),
            NumericOperationDescription::AddScalar(desc) => self.register_scalar_ops(
                desc,
                (E::gpu_elem(), E::gpu_elem(), E::gpu_elem()),
                |lhs, rhs, out| Operator::Add(BinaryOperator { lhs, rhs, out }),
            ),
            NumericOperationDescription::Sub(desc) => self.register_binary_ops(
                desc,
                (E::gpu_elem(), E::gpu_elem(), E::gpu_elem()),
                |lhs, rhs, out| Operator::Sub(BinaryOperator { lhs, rhs, out }),
            ),
            NumericOperationDescription::SubScalar(desc) => self.register_scalar_ops(
                desc,
                (E::gpu_elem(), E::gpu_elem(), E::gpu_elem()),
                |lhs, rhs, out| Operator::Sub(BinaryOperator { lhs, rhs, out }),
            ),
            NumericOperationDescription::Mul(desc) => self.register_binary_ops(
                desc,
                (E::gpu_elem(), E::gpu_elem(), E::gpu_elem()),
                |lhs, rhs, out| Operator::Mul(BinaryOperator { lhs, rhs, out }),
            ),
            NumericOperationDescription::MulScalar(desc) => self.register_scalar_ops(
                desc,
                (E::gpu_elem(), E::gpu_elem(), E::gpu_elem()),
                |lhs, rhs, out| Operator::Mul(BinaryOperator { lhs, rhs, out }),
            ),
            NumericOperationDescription::Div(desc) => self.register_binary_ops(
                desc,
                (E::gpu_elem(), E::gpu_elem(), E::gpu_elem()),
                |lhs, rhs, out| Operator::Div(BinaryOperator { lhs, rhs, out }),
            ),
            NumericOperationDescription::DivScalar(desc) => self.register_scalar_ops(
                desc,
                (E::gpu_elem(), E::gpu_elem(), E::gpu_elem()),
                |lhs, rhs, out| Operator::Div(BinaryOperator { lhs, rhs, out }),
            ),
            NumericOperationDescription::Abs(desc) => {
                self.register_unary_ops(desc, (E::gpu_elem(), E::gpu_elem()), |input, out| {
                    Operator::Abs(UnaryOperator { input, out })
                })
            }
            NumericOperationDescription::Lower(desc) => self.register_binary_ops(
                desc,
                (E::gpu_elem(), E::gpu_elem(), Elem::Bool),
                |lhs, rhs, out| Operator::Lower(BinaryOperator { lhs, rhs, out }),
            ),
            NumericOperationDescription::LowerElem(desc) => self.register_scalar_ops(
                desc,
                (E::gpu_elem(), E::gpu_elem(), Elem::Bool),
                |lhs, rhs, out| Operator::Lower(BinaryOperator { lhs, rhs, out }),
            ),
            NumericOperationDescription::Greater(desc) => self.register_binary_ops(
                desc,
                (E::gpu_elem(), E::gpu_elem(), Elem::Bool),
                |lhs, rhs, out| Operator::Greater(BinaryOperator { lhs, rhs, out }),
            ),
            NumericOperationDescription::GreaterElem(desc) => self.register_scalar_ops(
                desc,
                (E::gpu_elem(), E::gpu_elem(), Elem::Bool),
                |lhs, rhs, out| Operator::Greater(BinaryOperator { lhs, rhs, out }),
            ),
            NumericOperationDescription::LowerEqual(desc) => self.register_binary_ops(
                desc,
                (E::gpu_elem(), E::gpu_elem(), Elem::Bool),
                |lhs, rhs, out| Operator::LowerEqual(BinaryOperator { lhs, rhs, out }),
            ),
            NumericOperationDescription::LowerEqualElem(desc) => self.register_scalar_ops(
                desc,
                (E::gpu_elem(), E::gpu_elem(), Elem::Bool),
                |lhs, rhs, out| Operator::LowerEqual(BinaryOperator { lhs, rhs, out }),
            ),
            NumericOperationDescription::GreaterEqual(desc) => self.register_binary_ops(
                desc,
                (E::gpu_elem(), E::gpu_elem(), Elem::Bool),
                |lhs, rhs, out| Operator::GreaterEqual(BinaryOperator { lhs, rhs, out }),
            ),
            NumericOperationDescription::GreaterEqualElem(desc) => self.register_scalar_ops(
                desc,
                (E::gpu_elem(), E::gpu_elem(), Elem::Bool),
                |lhs, rhs, out| Operator::GreaterEqual(BinaryOperator { lhs, rhs, out }),
            ),
            NumericOperationDescription::EqualElem(desc) => self.register_scalar_ops(
                desc,
                (E::gpu_elem(), E::gpu_elem(), Elem::Bool),
                |lhs, rhs, out| Operator::Equal(BinaryOperator { lhs, rhs, out }),
            ),
            NumericOperationDescription::MaskWhere(desc) => {
                if !self.output_is_compatible(&desc.out) {
                    return false;
                }

                let cond = self.builder.input(&desc.mask, Elem::Bool);
                let lhs = self.builder.input(&desc.value, E::gpu_elem());
                let rhs = self.builder.input(&desc.tensor, E::gpu_elem());
                let out = self.builder.output(&desc.out, E::gpu_elem());

                self.builder
                    .register_operation(Procedure::ConditionalAssign(ConditionalAssign {
                        cond,
                        lhs,
                        rhs,
                        out,
                    }));

                true
            }
            NumericOperationDescription::MaskFill(desc) => {
                if !self.output_is_compatible(&desc.out) {
                    return false;
                }

                let cond = self.builder.input(&desc.mask, Elem::Bool);
                let lhs = self.builder.scalar(&desc.value, E::gpu_elem());
                let rhs = self.builder.input(&desc.tensor, E::gpu_elem());
                let out = self.builder.output(&desc.out, E::gpu_elem());

                self.builder
                    .register_operation(Procedure::ConditionalAssign(ConditionalAssign {
                        cond,
                        lhs,
                        rhs,
                        out,
                    }));

                true
            }
            NumericOperationDescription::Ones(desc) => {
                if !self.output_is_compatible(desc) {
                    return false;
                }

                let input = Variable::ConstantScalar(1.0, E::gpu_elem());
                let out = self.builder.output(desc, E::gpu_elem());

                self.builder
                    .register_operation(Operator::Assign(UnaryOperator { input, out }));

                true
            }
            NumericOperationDescription::Zeros(desc) => {
                if !self.output_is_compatible(desc) {
                    return false;
                }

                let input = Variable::ConstantScalar(0.0, E::gpu_elem());
                let out = self.builder.output(desc, E::gpu_elem());

                self.builder
                    .register_operation(Operator::Assign(UnaryOperator { input, out }));

                true
            }
            NumericOperationDescription::Full((desc, elem)) => {
                if !self.output_is_compatible(desc) {
                    return false;
                }

                let input = self.builder.scalar(elem, E::gpu_elem());
                let out = self.builder.output(desc, E::gpu_elem());

                self.builder
                    .register_operation(Operator::Assign(UnaryOperator { input, out }));

                true
            }
            _ => false,
        }
    }

    fn register_binary_ops<Func>(
        &mut self,
        desc: &BinaryOperationDescription,
        (elem_lhs, elem_rhs, elem_out): (Elem, Elem, Elem),
        func: Func,
    ) -> bool
    where
        Func: Fn(Variable, Variable, Variable) -> Operator,
    {
        if !self.output_is_compatible(&desc.out) {
            return false;
        }

        let lhs = self.builder.input(&desc.lhs, elem_lhs);
        let rhs = self.builder.input(&desc.rhs, elem_rhs);
        let out = self.builder.output(&desc.out, elem_out);

        self.builder.register_operation(func(lhs, rhs, out));

        true
    }

    fn register_unary_ops<Func>(
        &mut self,
        desc: &UnaryOperationDescription,
        (elem_input, elem_out): (Elem, Elem),
        func: Func,
    ) -> bool
    where
        Func: Fn(Variable, Variable) -> Operator,
    {
        if !self.output_is_compatible(&desc.out) {
            return false;
        }

        let input = self.builder.input(&desc.input, elem_input);
        let out = self.builder.output(&desc.out, elem_out);

        self.builder.register_operation(func(input, out));

        true
    }

    fn register_scalar_ops<Func, E: Element>(
        &mut self,
        desc: &ScalarOperationDescription<E>,
        (elem_lhs, elem_rhs, elem_out): (Elem, Elem, Elem),
        func: Func,
    ) -> bool
    where
        Func: Fn(Variable, Variable, Variable) -> Operator,
    {
        if !self.output_is_compatible(&desc.out) {
            return false;
        }

        let lhs = self.builder.input(&desc.lhs, elem_lhs);
        let rhs = self.builder.scalar(&desc.rhs, elem_rhs);
        let out = self.builder.output(&desc.out, elem_out);

        self.builder.register_operation(func(lhs, rhs, out));

        true
    }

    fn output_is_compatible(&mut self, out: &TensorDescription) -> bool {
        if self.current_output_shape.is_empty() {
            self.current_output_shape = out.shape.clone();
        } else if self.current_output_shape != out.shape {
            return false;
        }

        true
    }
}
