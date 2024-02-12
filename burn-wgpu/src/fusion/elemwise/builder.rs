use super::{optimization::ElementWise, CompilationPhase};
use crate::{
    codegen::dialect::gpu::{
        BinaryOperation, ConditionalAssignOperation, Elem, Item, Operation, UnaryOperation,
        Variable,
    },
    element::JitElement,
    fusion::{tracing::TraceBuilder, WgpuOptimization},
    JitBackend, Runtime,
};
use burn_fusion::{
    stream::{
        BaseOperationDescription, BinaryOperationDescription, FloatOperationDescription,
        NumericOperationDescription, OperationDescription, ScalarOperationDescription,
        UnaryOperationDescription,
    },
    OptimizationBuilder, OptimizationProperties, OptimizationStatus, TensorDescription,
};
use burn_tensor::{
    ops::{FloatElem, IntElem},
    Device, Element,
};

/// Fused element wise operations that are normally memory bound.
pub(crate) struct ElementWiseBuilder<R: Runtime> {
    tracer: TraceBuilder,
    current_output_shape: Vec<usize>,
    status: OptimizationStatus,
    num_added: usize,
    device: R::Device,
}

impl<R: Runtime> OptimizationBuilder<WgpuOptimization<R>> for ElementWiseBuilder<R> {
    fn register(&mut self, ops: &OperationDescription) {
        if let OptimizationStatus::Closed = self.status {
            return;
        }

        match ops {
            OperationDescription::BaseFloat(ops) => {
                if !self.register_base::<FloatElem<JitBackend<R>>>(ops) {
                    self.status = OptimizationStatus::Closed;
                    return;
                }
            }
            OperationDescription::BaseInt(ops) => {
                if !self.register_base::<IntElem<JitBackend<R>>>(ops) {
                    self.status = OptimizationStatus::Closed;
                    return;
                }
            }
            OperationDescription::Float(ops) => {
                if !self.register_float::<FloatElem<JitBackend<R>>>(ops) {
                    self.status = OptimizationStatus::Closed;
                    return;
                }
            }
            OperationDescription::NumericFloat(ops) => {
                if !self.register_numeric::<FloatElem<JitBackend<R>>, _>(ops) {
                    self.status = OptimizationStatus::Closed;
                    return;
                }
            }
            OperationDescription::NumericInt(ops) => {
                if !self.register_numeric::<IntElem<JitBackend<R>>, _>(ops) {
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

    fn build(&self) -> WgpuOptimization<R> {
        let op = ElementWise::new(
            self.tracer.clone().build(),
            self.num_added,
            self.device.clone(),
            CompilationPhase,
        );

        WgpuOptimization::ElementWise(op.compile())
    }

    fn len(&self) -> usize {
        self.num_added
    }

    fn reset(&mut self) {
        self.tracer = TraceBuilder::new("elem_wise");
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

impl<R: Runtime> ElementWiseBuilder<R> {
    pub fn new(device: Device<JitBackend<R>>) -> Self {
        Self {
            tracer: TraceBuilder::new("elem_wise"),
            num_added: 0,
            current_output_shape: Vec::new(),
            status: OptimizationStatus::Open,
            device,
        }
    }

    fn register_base<E: JitElement>(&mut self, ops: &BaseOperationDescription) -> bool {
        match ops {
            BaseOperationDescription::Equal(desc) => self.register_binary_ops(
                desc,
                (E::gpu_elem(), E::gpu_elem(), Elem::Bool),
                |lhs, rhs, out| Operation::Equal(BinaryOperation { lhs, rhs, out }),
            ),
            _ => false,
        }
    }

    fn register_float<E: JitElement>(&mut self, ops: &FloatOperationDescription) -> bool {
        match ops {
            FloatOperationDescription::Exp(desc) => {
                self.register_unary_ops(desc, (E::gpu_elem(), E::gpu_elem()), |input, out| {
                    Operation::Exp(UnaryOperation { input, out })
                })
            }
            FloatOperationDescription::Log(desc) => {
                self.register_unary_ops(desc, (E::gpu_elem(), E::gpu_elem()), |input, out| {
                    Operation::Log(UnaryOperation { input, out })
                })
            }
            FloatOperationDescription::Log1p(desc) => {
                self.register_unary_ops(desc, (E::gpu_elem(), E::gpu_elem()), |input, out| {
                    Operation::Log1p(UnaryOperation { input, out })
                })
            }
            FloatOperationDescription::Cos(desc) => {
                self.register_unary_ops(desc, (E::gpu_elem(), E::gpu_elem()), |input, out| {
                    Operation::Cos(UnaryOperation { input, out })
                })
            }
            FloatOperationDescription::Sin(desc) => {
                self.register_unary_ops(desc, (E::gpu_elem(), E::gpu_elem()), |input, out| {
                    Operation::Sin(UnaryOperation { input, out })
                })
            }
            FloatOperationDescription::PowfScalar(desc) => self.register_scalar_ops(
                desc,
                (E::gpu_elem(), E::gpu_elem(), E::gpu_elem()),
                |lhs, rhs, out| Operation::Powf(BinaryOperation { lhs, rhs, out }),
            ),
            FloatOperationDescription::Tanh(desc) => {
                self.register_unary_ops(desc, (E::gpu_elem(), E::gpu_elem()), |input, out| {
                    Operation::Tanh(UnaryOperation { input, out })
                })
            }
            FloatOperationDescription::Erf(desc) => {
                self.register_unary_ops(desc, (E::gpu_elem(), E::gpu_elem()), |input, out| {
                    Operation::Erf(UnaryOperation { input, out })
                })
            }
            FloatOperationDescription::Recip(desc) => {
                self.register_unary_ops(desc, (E::gpu_elem(), E::gpu_elem()), |input, out| {
                    Operation::Recip(UnaryOperation { input, out })
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
                |lhs, rhs, out| Operation::Add(BinaryOperation { lhs, rhs, out }),
            ),
            NumericOperationDescription::AddScalar(desc) => self.register_scalar_ops(
                desc,
                (E::gpu_elem(), E::gpu_elem(), E::gpu_elem()),
                |lhs, rhs, out| Operation::Add(BinaryOperation { lhs, rhs, out }),
            ),
            NumericOperationDescription::Sub(desc) => self.register_binary_ops(
                desc,
                (E::gpu_elem(), E::gpu_elem(), E::gpu_elem()),
                |lhs, rhs, out| Operation::Sub(BinaryOperation { lhs, rhs, out }),
            ),
            NumericOperationDescription::SubScalar(desc) => self.register_scalar_ops(
                desc,
                (E::gpu_elem(), E::gpu_elem(), E::gpu_elem()),
                |lhs, rhs, out| Operation::Sub(BinaryOperation { lhs, rhs, out }),
            ),
            NumericOperationDescription::Mul(desc) => self.register_binary_ops(
                desc,
                (E::gpu_elem(), E::gpu_elem(), E::gpu_elem()),
                |lhs, rhs, out| Operation::Mul(BinaryOperation { lhs, rhs, out }),
            ),
            NumericOperationDescription::MulScalar(desc) => self.register_scalar_ops(
                desc,
                (E::gpu_elem(), E::gpu_elem(), E::gpu_elem()),
                |lhs, rhs, out| Operation::Mul(BinaryOperation { lhs, rhs, out }),
            ),
            NumericOperationDescription::Div(desc) => self.register_binary_ops(
                desc,
                (E::gpu_elem(), E::gpu_elem(), E::gpu_elem()),
                |lhs, rhs, out| Operation::Div(BinaryOperation { lhs, rhs, out }),
            ),
            NumericOperationDescription::DivScalar(desc) => self.register_scalar_ops(
                desc,
                (E::gpu_elem(), E::gpu_elem(), E::gpu_elem()),
                |lhs, rhs, out| Operation::Div(BinaryOperation { lhs, rhs, out }),
            ),
            NumericOperationDescription::Abs(desc) => {
                self.register_unary_ops(desc, (E::gpu_elem(), E::gpu_elem()), |input, out| {
                    Operation::Abs(UnaryOperation { input, out })
                })
            }
            NumericOperationDescription::Lower(desc) => self.register_binary_ops(
                desc,
                (E::gpu_elem(), E::gpu_elem(), Elem::Bool),
                |lhs, rhs, out| Operation::Lower(BinaryOperation { lhs, rhs, out }),
            ),
            NumericOperationDescription::LowerElem(desc) => self.register_scalar_ops(
                desc,
                (E::gpu_elem(), E::gpu_elem(), Elem::Bool),
                |lhs, rhs, out| Operation::Lower(BinaryOperation { lhs, rhs, out }),
            ),
            NumericOperationDescription::Greater(desc) => self.register_binary_ops(
                desc,
                (E::gpu_elem(), E::gpu_elem(), Elem::Bool),
                |lhs, rhs, out| Operation::Greater(BinaryOperation { lhs, rhs, out }),
            ),
            NumericOperationDescription::GreaterElem(desc) => self.register_scalar_ops(
                desc,
                (E::gpu_elem(), E::gpu_elem(), Elem::Bool),
                |lhs, rhs, out| Operation::Greater(BinaryOperation { lhs, rhs, out }),
            ),
            NumericOperationDescription::LowerEqual(desc) => self.register_binary_ops(
                desc,
                (E::gpu_elem(), E::gpu_elem(), Elem::Bool),
                |lhs, rhs, out| Operation::LowerEqual(BinaryOperation { lhs, rhs, out }),
            ),
            NumericOperationDescription::LowerEqualElem(desc) => self.register_scalar_ops(
                desc,
                (E::gpu_elem(), E::gpu_elem(), Elem::Bool),
                |lhs, rhs, out| Operation::LowerEqual(BinaryOperation { lhs, rhs, out }),
            ),
            NumericOperationDescription::GreaterEqual(desc) => self.register_binary_ops(
                desc,
                (E::gpu_elem(), E::gpu_elem(), Elem::Bool),
                |lhs, rhs, out| Operation::GreaterEqual(BinaryOperation { lhs, rhs, out }),
            ),
            NumericOperationDescription::GreaterEqualElem(desc) => self.register_scalar_ops(
                desc,
                (E::gpu_elem(), E::gpu_elem(), Elem::Bool),
                |lhs, rhs, out| Operation::GreaterEqual(BinaryOperation { lhs, rhs, out }),
            ),
            NumericOperationDescription::EqualElem(desc) => self.register_scalar_ops(
                desc,
                (E::gpu_elem(), E::gpu_elem(), Elem::Bool),
                |lhs, rhs, out| Operation::Equal(BinaryOperation { lhs, rhs, out }),
            ),
            NumericOperationDescription::MaskWhere(desc) => {
                if !self.output_is_compatible(&desc.out) {
                    return false;
                }

                let cond = self.tracer.input_to_var(&desc.mask, Elem::Bool);
                let lhs = self.tracer.input_to_var(&desc.value, E::gpu_elem());
                let rhs = self.tracer.input_to_var(&desc.tensor, E::gpu_elem());
                let out = self.tracer.output_to_var(&desc.out, E::gpu_elem());

                self.tracer.register_operation(Operation::ConditionalAssign(
                    ConditionalAssignOperation {
                        cond,
                        lhs,
                        rhs,
                        out,
                    },
                ));

                true
            }
            NumericOperationDescription::MaskFill(desc) => {
                if !self.output_is_compatible(&desc.out) {
                    return false;
                }

                let cond = self.tracer.input_to_var(&desc.mask, Elem::Bool);
                let lhs = self.tracer.scalar_to_var(&desc.value, E::gpu_elem());
                let rhs = self.tracer.input_to_var(&desc.tensor, E::gpu_elem());
                let out = self.tracer.output_to_var(&desc.out, E::gpu_elem());

                self.tracer.register_operation(Operation::ConditionalAssign(
                    ConditionalAssignOperation {
                        cond,
                        lhs,
                        rhs,
                        out,
                    },
                ));

                true
            }
            NumericOperationDescription::Ones(desc) => {
                if !self.output_is_compatible(desc) {
                    return false;
                }

                let input = Variable::Constant(1.0, Item::Scalar(E::gpu_elem()));
                let out = self.tracer.output_to_var(desc, E::gpu_elem());

                self.tracer
                    .register_operation(Operation::AssignLocal(UnaryOperation { input, out }));

                true
            }
            NumericOperationDescription::Zeros(desc) => {
                if !self.output_is_compatible(desc) {
                    return false;
                }

                let input = Variable::Constant(0.0, Item::Scalar(E::gpu_elem()));
                let out = self.tracer.output_to_var(desc, E::gpu_elem());

                self.tracer
                    .register_operation(Operation::AssignLocal(UnaryOperation { input, out }));

                true
            }
            NumericOperationDescription::Full((desc, elem)) => {
                if !self.output_is_compatible(desc) {
                    return false;
                }

                let input = self.tracer.scalar_to_var(elem, E::gpu_elem());
                let out = self.tracer.output_to_var(desc, E::gpu_elem());

                self.tracer
                    .register_operation(Operation::AssignLocal(UnaryOperation { input, out }));

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
        Func: Fn(Variable, Variable, Variable) -> Operation,
    {
        if !self.output_is_compatible(&desc.out) {
            return false;
        }

        let lhs = self.tracer.input_to_var(&desc.lhs, elem_lhs);
        let rhs = self.tracer.input_to_var(&desc.rhs, elem_rhs);
        let out = self.tracer.output_to_var(&desc.out, elem_out);

        self.tracer.register_operation(func(lhs, rhs, out));

        true
    }

    fn register_unary_ops<Func>(
        &mut self,
        desc: &UnaryOperationDescription,
        (elem_input, elem_out): (Elem, Elem),
        func: Func,
    ) -> bool
    where
        Func: Fn(Variable, Variable) -> Operation,
    {
        if !self.output_is_compatible(&desc.out) {
            return false;
        }

        let input = self.tracer.input_to_var(&desc.input, elem_input);
        let out = self.tracer.output_to_var(&desc.out, elem_out);

        self.tracer.register_operation(func(input, out));

        true
    }

    fn register_scalar_ops<Func, E: Element>(
        &mut self,
        desc: &ScalarOperationDescription<E>,
        (elem_lhs, elem_rhs, elem_out): (Elem, Elem, Elem),
        func: Func,
    ) -> bool
    where
        Func: Fn(Variable, Variable, Variable) -> Operation,
    {
        if !self.output_is_compatible(&desc.out) {
            return false;
        }

        let lhs = self.tracer.input_to_var(&desc.lhs, elem_lhs);
        let rhs = self.tracer.scalar_to_var(&desc.rhs, elem_rhs);
        let out = self.tracer.output_to_var(&desc.out, elem_out);

        self.tracer.register_operation(func(lhs, rhs, out));

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
