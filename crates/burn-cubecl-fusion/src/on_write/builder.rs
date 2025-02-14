use super::{
    ir::{Arg, BinaryElemwiseArgs, ElemwiseOp, ElemwisePrecision, UnaryElemwiseArgs},
    settings::FuseSettings,
    trace::{FuseOnWriteTrace, FuseOnWriteTraceBuilder},
};
use burn_fusion::{OptimizationBuilder, OptimizationProperties, OptimizationStatus};
use burn_ir::{
    BaseOperationIr, BinaryOpIr, FloatOperationIr, NumericOperationIr, OperationIr, ScalarOpIr,
    TensorIr, UnaryOpIr,
};
use burn_tensor::Element;
use cubecl::ir::Elem;

/// Fused element wise operations that are normally memory bound.
pub(crate) struct FuseOnWriteBuilder {
    builder: TryFuseBuilder,
    settings: FuseSettings,
    current_output_shape: Vec<usize>,
    status: OptimizationStatus,
    pub(crate) num_ops: usize,
    pub(crate) num_views: usize,
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

    fn register(
        &mut self,
        add_ops: impl FnOnce(&mut FuseOnWriteTraceBuilder) -> Option<()>,
    ) -> bool {
        // Always allow the first operation to be added.
        if !self.added_ops {
            self.added_ops = true;

            if add_ops(&mut self.builder).is_none() {
                return false;
            }
            return true;
        }

        let mut cloned = self.builder.clone();
        if add_ops(&mut cloned).is_none() {
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
    fn register(&mut self, op: &OperationIr) {
        if let OptimizationStatus::Closed = self.status {
            return;
        }

        match op {
            OperationIr::BaseFloat(ops) => {
                if !self.register_base(ops) {
                    self.status = OptimizationStatus::Closed;
                    return;
                }
            }
            OperationIr::BaseInt(ops) => {
                if !self.register_base(ops) {
                    self.status = OptimizationStatus::Closed;
                    return;
                }
            }
            OperationIr::Float(_dtype, ops) => {
                if !self.register_float(ops) {
                    self.status = OptimizationStatus::Closed;
                    return;
                }
            }
            OperationIr::NumericFloat(_dtype, ops) => {
                if !self.register_numeric::<f32>(ops) {
                    self.status = OptimizationStatus::Closed;
                    return;
                }
            }
            OperationIr::NumericInt(_dtype, ops) => {
                if !self.register_numeric::<i32>(ops) {
                    self.status = OptimizationStatus::Closed;
                    return;
                }
            }
            OperationIr::BaseBool(ops) => {
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
            num_views: 0,
            max_bindings,
            current_output_shape: Vec::new(),
            status: OptimizationStatus::Open,
        }
    }

    pub fn close(&mut self) {
        self.status = OptimizationStatus::Closed;
    }

    pub fn input_unhandled(&mut self, tensor: &TensorIr) -> Arg {
        self.builder.builder.input_unhandled(tensor)
    }

    pub fn output_unhandled(&mut self, tensor: &TensorIr) -> Arg {
        if self.current_output_shape.is_empty() {
            self.current_output_shape = tensor.shape.clone();
        } else if self.current_output_shape.iter().sum::<usize>() < tensor.shape.iter().sum() {
            // The larguest shape win.
            self.current_output_shape = tensor.shape.clone();
        }

        self.builder.builder.output_unhandled(tensor)
    }

    fn register_base(&mut self, ops: &BaseOperationIr) -> bool {
        match ops {
            BaseOperationIr::Equal(desc) => self.register_binary_ops(desc, |lhs, rhs, out| {
                ElemwiseOp::Equal(BinaryElemwiseArgs { lhs, rhs, out })
            }),
            BaseOperationIr::Cast(desc) => self.register_unary_ops(desc, |input, out| {
                ElemwiseOp::Assign(UnaryElemwiseArgs { input, out })
            }),
            BaseOperationIr::SwapDims(desc) => {
                if !self.output_is_compatible(&desc.out) {
                    return false;
                }

                if self.builder.register(|build| {
                    build.input_swap_dims(
                        &desc.input,
                        &desc.out,
                        (desc.dim1 as u32, desc.dim2 as u32),
                    )?;

                    Some(())
                }) {
                    self.num_views += 1;
                    true
                } else {
                    false
                }
            }
            BaseOperationIr::Reshape(desc) => {
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
                    build.input_reshaped(&desc.input, &desc.out)?;
                    Some(())
                }) {
                    self.num_views += 1;
                    true
                } else {
                    false
                }
            }
            _ => false,
        }
    }

    fn register_float(&mut self, ops: &FloatOperationIr) -> bool {
        match ops {
            FloatOperationIr::Exp(desc) => self.register_unary_ops(desc, |input, out| {
                ElemwiseOp::Exp(UnaryElemwiseArgs { input, out })
            }),
            FloatOperationIr::Log(desc) => self.register_unary_ops(desc, |input, out| {
                ElemwiseOp::Log(UnaryElemwiseArgs { input, out })
            }),
            FloatOperationIr::Log1p(desc) => self.register_unary_ops(desc, |input, out| {
                ElemwiseOp::Log1p(UnaryElemwiseArgs { input, out })
            }),
            FloatOperationIr::Cos(desc) => self.register_unary_ops(desc, |input, out| {
                ElemwiseOp::Cos(UnaryElemwiseArgs { input, out })
            }),
            FloatOperationIr::Sin(desc) => self.register_unary_ops(desc, |input, out| {
                ElemwiseOp::Sin(UnaryElemwiseArgs { input, out })
            }),
            FloatOperationIr::PowfScalar(desc) => self
                .register_scalar_ops(desc, |lhs, rhs, out| {
                    ElemwiseOp::Powf(BinaryElemwiseArgs { lhs, rhs, out })
                }),
            FloatOperationIr::Tanh(desc) => self.register_unary_ops(desc, |input, out| {
                ElemwiseOp::Tanh(UnaryElemwiseArgs { input, out })
            }),
            FloatOperationIr::Erf(desc) => self.register_unary_ops(desc, |input, out| {
                ElemwiseOp::Erf(UnaryElemwiseArgs { input, out })
            }),
            FloatOperationIr::Recip(desc) => self.register_unary_ops(desc, |input, out| {
                ElemwiseOp::Recip(UnaryElemwiseArgs { input, out })
            }),
            _ => false,
        }
    }

    fn register_numeric<E: Element>(&mut self, op: &NumericOperationIr<E>) -> bool {
        match op {
            NumericOperationIr::Add(desc) => self.register_binary_ops(desc, |lhs, rhs, out| {
                ElemwiseOp::Add(BinaryElemwiseArgs { lhs, rhs, out })
            }),
            NumericOperationIr::AddScalar(desc) => self
                .register_scalar_ops(desc, |lhs, rhs, out| {
                    ElemwiseOp::Add(BinaryElemwiseArgs { lhs, rhs, out })
                }),
            NumericOperationIr::Sub(desc) => self.register_binary_ops(desc, |lhs, rhs, out| {
                ElemwiseOp::Sub(BinaryElemwiseArgs { lhs, rhs, out })
            }),
            NumericOperationIr::SubScalar(desc) => self
                .register_scalar_ops(desc, |lhs, rhs, out| {
                    ElemwiseOp::Sub(BinaryElemwiseArgs { lhs, rhs, out })
                }),
            NumericOperationIr::Mul(desc) => self.register_binary_ops(desc, |lhs, rhs, out| {
                ElemwiseOp::Mul(BinaryElemwiseArgs { lhs, rhs, out })
            }),
            NumericOperationIr::MulScalar(desc) => self
                .register_scalar_ops(desc, |lhs, rhs, out| {
                    ElemwiseOp::Mul(BinaryElemwiseArgs { lhs, rhs, out })
                }),
            NumericOperationIr::Div(desc) => self.register_binary_ops(desc, |lhs, rhs, out| {
                ElemwiseOp::Div(BinaryElemwiseArgs { lhs, rhs, out })
            }),
            NumericOperationIr::DivScalar(desc) => self
                .register_scalar_ops(desc, |lhs, rhs, out| {
                    ElemwiseOp::Div(BinaryElemwiseArgs { lhs, rhs, out })
                }),
            NumericOperationIr::Abs(desc) => self.register_unary_ops(desc, |input, out| {
                ElemwiseOp::Abs(UnaryElemwiseArgs { input, out })
            }),
            NumericOperationIr::Lower(desc) => self.register_binary_ops(desc, |lhs, rhs, out| {
                ElemwiseOp::Lower(BinaryElemwiseArgs { lhs, rhs, out })
            }),
            NumericOperationIr::LowerElem(desc) => self
                .register_scalar_ops(desc, |lhs, rhs, out| {
                    ElemwiseOp::Lower(BinaryElemwiseArgs { lhs, rhs, out })
                }),
            NumericOperationIr::Greater(desc) => self.register_binary_ops(desc, |lhs, rhs, out| {
                ElemwiseOp::Greater(BinaryElemwiseArgs { lhs, rhs, out })
            }),
            NumericOperationIr::GreaterElem(desc) => self
                .register_scalar_ops(desc, |lhs, rhs, out| {
                    ElemwiseOp::Greater(BinaryElemwiseArgs { lhs, rhs, out })
                }),
            NumericOperationIr::LowerEqual(desc) => self
                .register_binary_ops(desc, |lhs, rhs, out| {
                    ElemwiseOp::LowerEqual(BinaryElemwiseArgs { lhs, rhs, out })
                }),
            NumericOperationIr::LowerEqualElem(desc) => self
                .register_scalar_ops(desc, |lhs, rhs, out| {
                    ElemwiseOp::LowerEqual(BinaryElemwiseArgs { lhs, rhs, out })
                }),
            NumericOperationIr::GreaterEqual(desc) => self
                .register_binary_ops(desc, |lhs, rhs, out| {
                    ElemwiseOp::GreaterEqual(BinaryElemwiseArgs { lhs, rhs, out })
                }),
            NumericOperationIr::GreaterEqualElem(desc) => self
                .register_scalar_ops(desc, |lhs, rhs, out| {
                    ElemwiseOp::GreaterEqual(BinaryElemwiseArgs { lhs, rhs, out })
                }),
            NumericOperationIr::EqualElem(desc) => self
                .register_scalar_ops(desc, |lhs, rhs, out| {
                    ElemwiseOp::Equal(BinaryElemwiseArgs { lhs, rhs, out })
                }),
            NumericOperationIr::MaskWhere(desc) => {
                if !self.output_is_compatible(&desc.out) {
                    return false;
                }

                self.builder.register(|build| {
                    let cond = build.input(&desc.mask)?;
                    let lhs = build.input(&desc.value)?;
                    let rhs = build.input(&desc.tensor)?;
                    let out = build.output(&desc.out)?;

                    build.register_operation(ElemwiseOp::ConditionalAssign {
                        cond,
                        lhs,
                        rhs,
                        out,
                    });

                    Some(())
                })
            }
            NumericOperationIr::MaskFill(desc) => {
                if !self.output_is_compatible(&desc.out) {
                    return false;
                }

                self.builder.register(|build| {
                    let cond = build.input(&desc.mask)?;
                    let lhs = build.scalar(&desc.value, desc.out.dtype);
                    let rhs = build.input(&desc.tensor)?;
                    let out = build.output(&desc.out)?;

                    build.register_operation(ElemwiseOp::ConditionalAssign {
                        cond,
                        lhs,
                        rhs,
                        out,
                    });

                    Some(())
                })
            }
            NumericOperationIr::Ones(desc) => {
                if !self.output_is_compatible(desc) {
                    return false;
                }

                let elem: Elem = desc.dtype.into();
                let precision = elem.into();
                let input = Arg::Literal(1, precision);

                self.builder.register(|build| {
                    let out = build.output(desc)?;

                    build.register_operation(ElemwiseOp::Assign(UnaryElemwiseArgs { input, out }));

                    Some(())
                })
            }
            NumericOperationIr::Zeros(desc) => {
                if !self.output_is_compatible(desc) {
                    return false;
                }

                let elem: Elem = desc.dtype.into();
                let precision = elem.into();
                let input = Arg::Literal(0, precision);

                self.builder.register(|build| {
                    let out = build.output(desc)?;

                    build.register_operation(ElemwiseOp::Assign(UnaryElemwiseArgs { input, out }));

                    Some(())
                })
            }
            NumericOperationIr::Full((desc, elem)) => {
                if !self.output_is_compatible(desc) {
                    return false;
                }

                self.builder.register(|build| {
                    let input = build.scalar(elem, desc.dtype);
                    let out = build.output(desc)?;

                    build.register_operation(ElemwiseOp::Assign(UnaryElemwiseArgs { input, out }));

                    Some(())
                })
            }
            NumericOperationIr::Gather(desc) => {
                if !self.output_is_compatible(&desc.out) {
                    return false;
                }

                self.builder.register(|build| {
                    let input = build.input_indexed(&desc.tensor)?;
                    let indices = build.input(&desc.indices)?;
                    let output = build.output(&desc.out)?;

                    build.register_operation(ElemwiseOp::Gather {
                        input,
                        indices,
                        output,
                        dim: desc.dim as u32,
                    });

                    Some(())
                })
            }
            NumericOperationIr::Select(desc) => {
                if !self.output_is_compatible(&desc.out) {
                    return false;
                }

                self.builder.register(|build| {
                    let input = build.input_indexed(&desc.tensor)?;
                    let indices = build.input_indexed(&desc.indices)?;
                    let output = build.output(&desc.out)?;

                    build.register_operation(ElemwiseOp::Select {
                        input,
                        indices,
                        output,
                        dim: desc.dim as u32,
                    });

                    Some(())
                })
            }
            _ => false,
        }
    }

    fn register_binary_ops<Func>(&mut self, desc: &BinaryOpIr, func: Func) -> bool
    where
        Func: Fn(Arg, Arg, Arg) -> ElemwiseOp,
    {
        if !self.output_is_compatible(&desc.out) {
            return false;
        }

        self.builder.register(|build| {
            let lhs = build.input(&desc.lhs)?;
            let rhs = build.input(&desc.rhs)?;
            let out = build.output(&desc.out)?;

            build.register_operation(func(lhs, rhs, out));

            Some(())
        })
    }

    fn register_unary_ops<Func>(&mut self, desc: &UnaryOpIr, func: Func) -> bool
    where
        Func: Fn(Arg, Arg) -> ElemwiseOp,
    {
        if !self.output_is_compatible(&desc.out) {
            return false;
        }

        self.builder.register(|build| {
            let input = build.input(&desc.input)?;
            let out = build.output(&desc.out)?;
            build.register_operation(func(input, out));
            Some(())
        })
    }

    fn register_scalar_ops<Func, E: Element>(&mut self, desc: &ScalarOpIr<E>, func: Func) -> bool
    where
        Func: Fn(Arg, Arg, Arg) -> ElemwiseOp,
    {
        if !self.output_is_compatible(&desc.out) {
            return false;
        }

        self.builder.register(|build| {
            let elem = desc.lhs.dtype;
            let lhs = build.input(&desc.lhs)?;
            let rhs = build.scalar(&desc.rhs, elem);
            let out = build.output(&desc.out)?;

            build.register_operation(func(lhs, rhs, out));

            Some(())
        })
    }

    fn output_is_compatible(&mut self, out: &TensorIr) -> bool {
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
