use super::{
    ir::{Arg, BinaryElemwiseArgs, ElemwiseOp, ElemwisePrecision, UnaryElemwiseArgs},
    settings::FuseSettings,
    trace::{FuseOnWriteTrace, FuseOnWriteTraceBuilder},
};
use burn_fusion::{OptimizationBuilder, OptimizationProperties, OptimizationStatus};
use burn_tensor::{
    repr::{
        BaseOperationDescription, BinaryOperationDescription, FloatOperationDescription,
        NumericOperationDescription, OperationDescription, ScalarOperationDescription,
        TensorDescription, UnaryOperationDescription,
    },
    Element,
};
use cubecl::ir::Elem;

/// Fused element wise operations that are normally memory bound.
pub(crate) struct FuseOnWriteBuilder {
    builder: TryFuseBuilder,
    settings: FuseSettings,
    current_output_shape: Vec<usize>,
    status: OptimizationStatus,
    pub(crate) num_ops: usize,
    pub(crate) num_reshapes: usize,
    max_bindings: u32,
}

struct TryFuseBuilder {
    builder: FuseOnWriteTraceBuilder,
    max_bindings: u32,
    added_ops: bool,
}

impl TryFuseBuilder {
    fn new(max_bindings: u32, bool_precision: ElemwisePrecision, settings: FuseSettings) -> Self {
        Self {
            builder: FuseOnWriteTraceBuilder::new(bool_precision, settings),
            max_bindings,
            added_ops: false,
        }
    }

    fn register(&mut self, add_ops: impl FnOnce(&mut FuseOnWriteTraceBuilder) -> bool) -> bool {
        // Always allow the first operation to be added.
        if !self.added_ops {
            self.added_ops = true;

            if !add_ops(&mut self.builder) {
                return false;
            }
            return true;
        }

        let mut cloned = self.builder.clone();
        if !add_ops(&mut cloned) {
            return false;
        }

        if cloned.estimate_bindings() > self.max_bindings {
            return false;
        }

        self.builder = cloned;
        true
    }

    fn build(&self, shape: Vec<usize>) -> FuseOnWriteTrace {
        self.builder.build(shape)
    }
}

impl OptimizationBuilder<FuseOnWriteTrace> for FuseOnWriteBuilder {
    fn register(&mut self, op: &OperationDescription) {
        if let OptimizationStatus::Closed = self.status {
            return;
        }

        match op {
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
            OperationDescription::BaseBool(ops) => {
                if !self.register_base(ops) {
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
        self.num_ops += 1;
    }

    fn build(&self) -> FuseOnWriteTrace {
        self.builder.build(self.current_output_shape.clone())
    }

    fn len(&self) -> usize {
        self.num_ops
    }

    fn reset(&mut self) {
        self.num_ops = 0;
        self.status = OptimizationStatus::Open;
        self.builder = TryFuseBuilder::new(
            self.max_bindings,
            self.builder.builder.bool_precision,
            self.settings,
        );
        self.current_output_shape.clear();
    }

    fn status(&self) -> OptimizationStatus {
        self.status
    }

    fn properties(&self) -> OptimizationProperties {
        let ready = self.num_ops > 0;

        OptimizationProperties {
            ready,
            score: self.num_ops as u64,
        }
    }
}

impl FuseOnWriteBuilder {
    pub fn new(
        max_bindings: u32,
        bool_precision: ElemwisePrecision,
        settings: FuseSettings,
    ) -> Self {
        Self {
            builder: TryFuseBuilder::new(max_bindings, bool_precision, settings),
            settings,
            num_ops: 0,
            num_reshapes: 0,
            max_bindings,
            current_output_shape: Vec::new(),
            status: OptimizationStatus::Open,
        }
    }

    pub fn close(&mut self) {
        self.status = OptimizationStatus::Closed;
    }

    pub fn input_unhandled(&mut self, tensor: &TensorDescription) -> Arg {
        self.builder.builder.input_unhandled(tensor)
    }

    pub fn output_unhandled(&mut self, tensor: &TensorDescription) -> Arg {
        if self.current_output_shape.is_empty() {
            self.current_output_shape = tensor.shape.clone();
        } else if self.current_output_shape.iter().sum::<usize>() < tensor.shape.iter().sum() {
            // The larguest shape win.
            self.current_output_shape = tensor.shape.clone();
        }

        self.builder.builder.output_unhandled(tensor)
    }

    fn register_base(&mut self, ops: &BaseOperationDescription) -> bool {
        match ops {
            BaseOperationDescription::Equal(desc) => self
                .register_binary_ops(desc, |lhs, rhs, out| {
                    ElemwiseOp::Equal(BinaryElemwiseArgs { lhs, rhs, out })
                }),
            BaseOperationDescription::Cast(desc) => self.register_unary_ops(desc, |input, out| {
                ElemwiseOp::Assign(UnaryElemwiseArgs { input, out })
            }),
            BaseOperationDescription::Reshape(desc) => {
                if desc.input.shape == desc.out.shape {
                    return self.register_unary_ops(desc, |input, out| {
                        ElemwiseOp::Assign(UnaryElemwiseArgs { input, out })
                    });
                }

                if desc.input.shape.len() > desc.out.shape.len() {
                    // Not yet supported.
                    return false;
                }

                if !self.output_is_compatible(&desc.out) {
                    return false;
                }

                if self.builder.register(|build| {
                    let input = match build.input_reshaped(&desc.input, &desc.out) {
                        Some(val) => val,
                        None => return false,
                    };
                    let out = build.output(&desc.out);

                    build.register_operation(ElemwiseOp::Assign(UnaryElemwiseArgs { input, out }));

                    true
                }) {
                    self.num_reshapes += 1;
                    true
                } else {
                    false
                }
            }
            _ => false,
        }
    }

    fn register_float(&mut self, ops: &FloatOperationDescription) -> bool {
        match ops {
            FloatOperationDescription::Exp(desc) => self.register_unary_ops(desc, |input, out| {
                ElemwiseOp::Exp(UnaryElemwiseArgs { input, out })
            }),
            FloatOperationDescription::Log(desc) => self.register_unary_ops(desc, |input, out| {
                ElemwiseOp::Log(UnaryElemwiseArgs { input, out })
            }),
            FloatOperationDescription::Log1p(desc) => self
                .register_unary_ops(desc, |input, out| {
                    ElemwiseOp::Log1p(UnaryElemwiseArgs { input, out })
                }),
            FloatOperationDescription::Cos(desc) => self.register_unary_ops(desc, |input, out| {
                ElemwiseOp::Cos(UnaryElemwiseArgs { input, out })
            }),
            FloatOperationDescription::Sin(desc) => self.register_unary_ops(desc, |input, out| {
                ElemwiseOp::Sin(UnaryElemwiseArgs { input, out })
            }),
            FloatOperationDescription::PowfScalar(desc) => self
                .register_scalar_ops(desc, |lhs, rhs, out| {
                    ElemwiseOp::Powf(BinaryElemwiseArgs { lhs, rhs, out })
                }),
            FloatOperationDescription::Tanh(desc) => self.register_unary_ops(desc, |input, out| {
                ElemwiseOp::Tanh(UnaryElemwiseArgs { input, out })
            }),
            FloatOperationDescription::Erf(desc) => self.register_unary_ops(desc, |input, out| {
                ElemwiseOp::Erf(UnaryElemwiseArgs { input, out })
            }),
            FloatOperationDescription::Recip(desc) => self
                .register_unary_ops(desc, |input, out| {
                    ElemwiseOp::Recip(UnaryElemwiseArgs { input, out })
                }),
            _ => false,
        }
    }

    fn register_numeric<E: Element>(&mut self, op: &NumericOperationDescription<E>) -> bool {
        match op {
            NumericOperationDescription::Add(desc) => self
                .register_binary_ops(desc, |lhs, rhs, out| {
                    ElemwiseOp::Add(BinaryElemwiseArgs { lhs, rhs, out })
                }),
            NumericOperationDescription::AddScalar(desc) => self
                .register_scalar_ops(desc, |lhs, rhs, out| {
                    ElemwiseOp::Add(BinaryElemwiseArgs { lhs, rhs, out })
                }),
            NumericOperationDescription::Sub(desc) => self
                .register_binary_ops(desc, |lhs, rhs, out| {
                    ElemwiseOp::Sub(BinaryElemwiseArgs { lhs, rhs, out })
                }),
            NumericOperationDescription::SubScalar(desc) => self
                .register_scalar_ops(desc, |lhs, rhs, out| {
                    ElemwiseOp::Sub(BinaryElemwiseArgs { lhs, rhs, out })
                }),
            NumericOperationDescription::Mul(desc) => self
                .register_binary_ops(desc, |lhs, rhs, out| {
                    ElemwiseOp::Mul(BinaryElemwiseArgs { lhs, rhs, out })
                }),
            NumericOperationDescription::MulScalar(desc) => self
                .register_scalar_ops(desc, |lhs, rhs, out| {
                    ElemwiseOp::Mul(BinaryElemwiseArgs { lhs, rhs, out })
                }),
            NumericOperationDescription::Div(desc) => self
                .register_binary_ops(desc, |lhs, rhs, out| {
                    ElemwiseOp::Div(BinaryElemwiseArgs { lhs, rhs, out })
                }),
            NumericOperationDescription::DivScalar(desc) => self
                .register_scalar_ops(desc, |lhs, rhs, out| {
                    ElemwiseOp::Div(BinaryElemwiseArgs { lhs, rhs, out })
                }),
            NumericOperationDescription::Abs(desc) => self
                .register_unary_ops(desc, |input, out| {
                    ElemwiseOp::Abs(UnaryElemwiseArgs { input, out })
                }),
            NumericOperationDescription::Lower(desc) => self
                .register_binary_ops(desc, |lhs, rhs, out| {
                    ElemwiseOp::Lower(BinaryElemwiseArgs { lhs, rhs, out })
                }),
            NumericOperationDescription::LowerElem(desc) => self
                .register_scalar_ops(desc, |lhs, rhs, out| {
                    ElemwiseOp::Lower(BinaryElemwiseArgs { lhs, rhs, out })
                }),
            NumericOperationDescription::Greater(desc) => self
                .register_binary_ops(desc, |lhs, rhs, out| {
                    ElemwiseOp::Greater(BinaryElemwiseArgs { lhs, rhs, out })
                }),
            NumericOperationDescription::GreaterElem(desc) => self
                .register_scalar_ops(desc, |lhs, rhs, out| {
                    ElemwiseOp::Greater(BinaryElemwiseArgs { lhs, rhs, out })
                }),
            NumericOperationDescription::LowerEqual(desc) => self
                .register_binary_ops(desc, |lhs, rhs, out| {
                    ElemwiseOp::LowerEqual(BinaryElemwiseArgs { lhs, rhs, out })
                }),
            NumericOperationDescription::LowerEqualElem(desc) => self
                .register_scalar_ops(desc, |lhs, rhs, out| {
                    ElemwiseOp::LowerEqual(BinaryElemwiseArgs { lhs, rhs, out })
                }),
            NumericOperationDescription::GreaterEqual(desc) => self
                .register_binary_ops(desc, |lhs, rhs, out| {
                    ElemwiseOp::GreaterEqual(BinaryElemwiseArgs { lhs, rhs, out })
                }),
            NumericOperationDescription::GreaterEqualElem(desc) => self
                .register_scalar_ops(desc, |lhs, rhs, out| {
                    ElemwiseOp::GreaterEqual(BinaryElemwiseArgs { lhs, rhs, out })
                }),
            NumericOperationDescription::EqualElem(desc) => self
                .register_scalar_ops(desc, |lhs, rhs, out| {
                    ElemwiseOp::Equal(BinaryElemwiseArgs { lhs, rhs, out })
                }),
            NumericOperationDescription::MaskWhere(desc) => {
                if !self.output_is_compatible(&desc.out) {
                    return false;
                }

                self.builder.register(|build| {
                    let cond = build.input(&desc.mask);
                    let lhs = build.input(&desc.value);
                    let rhs = build.input(&desc.tensor);
                    let out = build.output(&desc.out);

                    build.register_operation(ElemwiseOp::ConditionalAssign {
                        cond,
                        lhs,
                        rhs,
                        out,
                    });

                    true
                })
            }
            NumericOperationDescription::MaskFill(desc) => {
                if !self.output_is_compatible(&desc.out) {
                    return false;
                }

                self.builder.register(|build| {
                    let cond = build.input(&desc.mask);
                    let lhs = build.scalar(&desc.value, desc.out.dtype);
                    let rhs = build.input(&desc.tensor);
                    let out = build.output(&desc.out);

                    build.register_operation(ElemwiseOp::ConditionalAssign {
                        cond,
                        lhs,
                        rhs,
                        out,
                    });

                    true
                })
            }
            NumericOperationDescription::Ones(desc) => {
                if !self.output_is_compatible(desc) {
                    return false;
                }

                let elem: Elem = desc.dtype.into();
                let precision = elem.into();
                let input = Arg::Literal(1, precision);

                self.builder.register(|build| {
                    let out = build.output(desc);

                    build.register_operation(ElemwiseOp::Assign(UnaryElemwiseArgs { input, out }));

                    true
                })
            }
            NumericOperationDescription::Zeros(desc) => {
                if !self.output_is_compatible(desc) {
                    return false;
                }

                let elem: Elem = desc.dtype.into();
                let precision = elem.into();
                let input = Arg::Literal(0, precision);

                self.builder.register(|build| {
                    let out = build.output(desc);

                    build.register_operation(ElemwiseOp::Assign(UnaryElemwiseArgs { input, out }));

                    true
                })
            }
            NumericOperationDescription::Full((desc, elem)) => {
                if !self.output_is_compatible(desc) {
                    return false;
                }

                self.builder.register(|build| {
                    let input = build.scalar(elem, desc.dtype);
                    let out = build.output(desc);

                    build.register_operation(ElemwiseOp::Assign(UnaryElemwiseArgs { input, out }));

                    true
                })
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

        self.builder.register(|build| {
            let lhs = build.input(&desc.lhs);
            let rhs = build.input(&desc.rhs);
            let out = build.output(&desc.out);

            build.register_operation(func(lhs, rhs, out));

            true
        })
    }

    fn register_unary_ops<Func>(&mut self, desc: &UnaryOperationDescription, func: Func) -> bool
    where
        Func: Fn(Arg, Arg) -> ElemwiseOp,
    {
        if !self.output_is_compatible(&desc.out) {
            return false;
        }

        self.builder.register(|build| {
            let input = build.input(&desc.input);
            let out = build.output(&desc.out);
            build.register_operation(func(input, out));
            true
        })
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

        self.builder.register(|build| {
            let elem = desc.lhs.dtype;
            let lhs = build.input(&desc.lhs);
            let rhs = build.scalar(&desc.rhs, elem);
            let out = build.output(&desc.out);

            build.register_operation(func(lhs, rhs, out));

            true
        })
    }

    fn output_is_compatible(&mut self, out: &TensorDescription) -> bool {
        if self.current_output_shape.is_empty() {
            self.current_output_shape.clone_from(&out.shape);
            return true;
        }

        let rank = self.current_output_shape.len();

        // Rank should be equal.
        if rank != out.shape.len() {
            return false;
        }

        let mut updated = self.current_output_shape.clone();

        #[allow(clippy::needless_range_loop)]
        for i in 0..rank {
            let curr = self.current_output_shape[i];
            let new = out.shape[i];

            if curr == new {
                continue;
            }

            // Broadcast not enabled.
            if !self.settings.broadcast {
                return false;
            }

            // Broadcasted on new dim.
            if new == 0 {
                continue;
            }

            // Broadcasted on curr dim - update reference output shape.
            if curr == 0 && self.settings.output_shape_updates {
                updated[i] = new;
                continue;
            }

            return false;
        }

        if updated != out.shape {
            return false;
        }
        self.current_output_shape.clone_from_slice(&out.shape);

        true
    }
}
