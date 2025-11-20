use super::{
    ir::{BinaryFuseArgs, FuseArg, FuseOp, FuseType, UnaryFuseArgs},
    settings::FuseSettings,
    trace::{FuseTrace, FuseTraceBuilder, block::QuantInput},
};
use crate::shared::ir::QuantSchemeFuse;
use burn_fusion::{OptimizationBuilder, OptimizationProperties, OptimizationStatus};
use burn_ir::{
    BaseOperationIr, BinaryOpIr, FloatOperationIr, NumericOperationIr, OperationIr, ScalarOpIr,
    TensorIr, UnaryOpIr,
};
use burn_tensor::DType;
use cubecl::ir::ElemType;

/// The base optimization builder that can be used to fuse all elemwise operations.
///
/// This builder doesn't create a ready-to-execute kernel, but rather generates a
/// [trace](FuseTrace) that be used with a [runner](super::trace::TraceRunner).
///
/// Since this builder supports fusing multiple blocks, you can fuse any compute-bound operations
/// with the combination of fuse-on-read and fuse-on-write strategy.
#[derive(Debug, Clone)]
pub(crate) struct FuseOptimizationBuilder {
    builder: TryFuseBuilder,
    pub(crate) settings: FuseSettings,
    pub(crate) current_output_shape: Vec<usize>,
    status: OptimizationStatus,
    pub(crate) num_ops: usize,
    pub(crate) num_views: usize,
    max_bindings: u32,
}

impl FuseOptimizationBuilder {
    /// Checks if the [operation](OperationIr) can be fused with the current builder.
    pub(crate) fn can_register(&self, op: &OperationIr) -> bool {
        let len_previous = self.len();
        let mut builder_cloned = self.clone();

        builder_cloned.register(op);
        let len_after = builder_cloned.len();

        len_after > len_previous
    }
}

impl OptimizationBuilder<FuseTrace> for FuseOptimizationBuilder {
    fn register(&mut self, op: &OperationIr) {
        if let OptimizationStatus::Closed = self.status {
            return;
        }

        match op {
            OperationIr::Drop(tensor) => {
                if self.num_ops == 0 {
                    self.status = OptimizationStatus::Closed;
                    return;
                }

                self.builder.builder.register_dropped(tensor.id);
            }
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
                if !self.register_numeric(ops) {
                    self.status = OptimizationStatus::Closed;
                    return;
                }
            }
            OperationIr::NumericInt(_dtype, ops) => {
                if !self.register_numeric(ops) {
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

    fn build(&self) -> FuseTrace {
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

    fn clone_dyn(&self) -> Box<dyn OptimizationBuilder<FuseTrace>> {
        Box::new(self.clone())
    }
}

impl FuseOptimizationBuilder {
    /// Creates a new builder.
    pub fn new(max_bindings: u32, bool_precision: FuseType, settings: FuseSettings) -> Self {
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

    /// Closes the builder.
    pub fn close(&mut self) {
        self.status = OptimizationStatus::Closed;
    }

    /// Declares an input tensor argument where the kernel is responsible to load.
    ///
    /// # Returns
    ///
    /// - The argument that maps to the tensor to be used during kernel expansion.
    pub fn input_unhandled(&mut self, tensor: &TensorIr) -> FuseArg {
        self.builder.builder.input_unhandled(tensor)
    }

    /// Declares an input quantized tensor argument where the kernel is responsible to load.
    ///
    /// # Returns
    ///
    /// None if it's not possible to register a quantized tensor. Otherwise:
    ///
    /// - The argument that maps to the tensor values to be used during kernel expansion.
    /// - The argument that maps to the tensor params to be used during kernel expansion.
    pub fn input_quantized_unhandled(&mut self, tensor: &TensorIr) -> Option<(FuseArg, FuseArg)> {
        self.builder.builder.input_quantized_unhandled(tensor)
    }

    /// Declares an output tensor argument where the kernel is responsible to write values.
    ///
    /// # Notes
    ///
    /// Normally you don't have to declare outputs explicitly before they are going to be
    /// registered based on the operations [registered](Self::register).
    ///
    /// # Returns
    ///
    /// - The argument that maps to the tensor to be used during kernel expansion.
    pub fn output_unhandled(&mut self, tensor: &TensorIr) -> FuseArg {
        if self.current_output_shape.is_empty() {
            self.current_output_shape = tensor.shape.dims.clone();
        } else if self.current_output_shape.iter().sum::<usize>() < tensor.shape.iter().sum() {
            // The larguest shape win.
            self.current_output_shape = tensor.shape.dims.clone();
        }

        self.builder.builder.output_unhandled(tensor)
    }

    /// Closes the previous block and declares a new one.
    ///
    /// # Arguments
    ///
    /// - arguments: Tensors that are logical outputs of the current block and inputs of the
    /// following blocks.
    /// - settings: [FuseSettings] to be used by the next block.
    ///
    /// # Returns
    ///
    /// None if it's impossible to create a next block with the given arguments. Otherwise, the
    /// corresponding [arguments](Arg) to the given tensors are returned.
    pub fn next_block<const N: usize>(
        &mut self,
        arguments: [&TensorIr; N],
        settings: FuseSettings,
    ) -> Option<[FuseArg; N]> {
        let mut is_success = true;
        let args = arguments.map(|arg| {
            // We need to register the argument as input in the current block so that we retrieve
            // its value locally.
            let input = self.builder.builder.input(arg);

            if input.is_none() {
                is_success = false;
            } else {
                // This flag the new input local value as local output to be used in a following
                // block.
                self.builder.builder.block_local_output(arg);
            }

            input
        });

        let args = if !is_success {
            return None;
        } else {
            args.map(|arg| arg.unwrap())
        };

        let current_output_shape = core::mem::take(&mut self.current_output_shape);

        self.builder
            .builder
            .next_block(current_output_shape, settings);

        self.settings = settings;
        self.status = OptimizationStatus::Open;

        Some(args)
    }

    fn register_base(&mut self, ops: &BaseOperationIr) -> bool {
        match ops {
            BaseOperationIr::Equal(desc) => self.register_binary_ops(desc, |lhs, rhs, out| {
                FuseOp::Equal(BinaryFuseArgs { lhs, rhs, out })
            }),
            BaseOperationIr::Cast(desc) => {
                self.register_unary_op(&desc.input, &desc.out, |input, out| {
                    FuseOp::Assign(UnaryFuseArgs { input, out })
                })
            }
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
                    return self.register_unary_op(&desc.input, &desc.out, |input, out| {
                        FuseOp::Assign(UnaryFuseArgs { input, out })
                    });
                }

                if desc.input.shape.rank() > desc.out.shape.rank() {
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
            BaseOperationIr::Ones(desc) => {
                if !self.output_is_compatible(&desc.out) {
                    return false;
                }

                let elem: ElemType = desc.out.dtype.into();
                let precision = elem.into();
                let input = FuseArg::Literal(1, precision);

                self.builder.register(|build| {
                    let out = build.output(&desc.out)?;

                    build.register_operation(FuseOp::Assign(UnaryFuseArgs { input, out }));

                    Some(())
                })
            }
            BaseOperationIr::Zeros(desc) => {
                if !self.output_is_compatible(&desc.out) {
                    return false;
                }

                let elem: ElemType = desc.out.dtype.into();
                let precision = elem.into();
                let input = FuseArg::Literal(0, precision);

                self.builder.register(|build| {
                    let out = build.output(&desc.out)?;

                    build.register_operation(FuseOp::Assign(UnaryFuseArgs { input, out }));

                    Some(())
                })
            }
            _ => false,
        }
    }

    fn register_float(&mut self, ops: &FloatOperationIr) -> bool {
        match ops {
            FloatOperationIr::Exp(desc) => self
                .register_unary_ops(desc, |input, out| FuseOp::Exp(UnaryFuseArgs { input, out })),
            FloatOperationIr::Log(desc) => self
                .register_unary_ops(desc, |input, out| FuseOp::Log(UnaryFuseArgs { input, out })),
            FloatOperationIr::Log1p(desc) => self.register_unary_ops(desc, |input, out| {
                FuseOp::Log1p(UnaryFuseArgs { input, out })
            }),
            FloatOperationIr::Cos(desc) => self
                .register_unary_ops(desc, |input, out| FuseOp::Cos(UnaryFuseArgs { input, out })),
            FloatOperationIr::Sin(desc) => self
                .register_unary_ops(desc, |input, out| FuseOp::Sin(UnaryFuseArgs { input, out })),
            FloatOperationIr::PowfScalar(desc) => self
                .register_scalar_ops(desc, |lhs, rhs, out| {
                    FuseOp::Powf(BinaryFuseArgs { lhs, rhs, out })
                }),
            FloatOperationIr::Tanh(desc) => self.register_unary_ops(desc, |input, out| {
                FuseOp::Tanh(UnaryFuseArgs { input, out })
            }),
            FloatOperationIr::Erf(desc) => self
                .register_unary_ops(desc, |input, out| FuseOp::Erf(UnaryFuseArgs { input, out })),
            FloatOperationIr::Sqrt(desc) => self.register_unary_ops(desc, |input, out| {
                FuseOp::Sqrt(UnaryFuseArgs { input, out })
            }),
            FloatOperationIr::Recip(desc) => self.register_unary_ops(desc, |input, out| {
                FuseOp::Recip(UnaryFuseArgs { input, out })
            }),
            FloatOperationIr::Dequantize(desc) => {
                if !self.output_is_compatible(&desc.out) {
                    return false;
                }

                self.builder.register(|build| {
                    let qinput = build.input_quantized(&desc.input)?;
                    let out = build.output(&desc.out)?;

                    match qinput {
                        QuantInput::AlreadyDequantized { local } => {
                            build.register_operation(FuseOp::Assign(UnaryFuseArgs {
                                input: local,
                                out,
                            }));
                        }
                        QuantInput::Quantized { values, params } => {
                            build.register_operation(FuseOp::Dequantize {
                                values,
                                params,
                                output: out,
                                scheme: match desc.input.dtype {
                                    DType::QFloat(scheme) => QuantSchemeFuse { scheme },
                                    _ => unreachable!("Should be a quant tensor."),
                                },
                            });
                        }
                    }

                    Some(())
                })
            }
            _ => false,
        }
    }

    fn register_numeric(&mut self, op: &NumericOperationIr) -> bool {
        match op {
            NumericOperationIr::Add(desc) => self.register_binary_ops(desc, |lhs, rhs, out| {
                FuseOp::Add(BinaryFuseArgs { lhs, rhs, out })
            }),
            NumericOperationIr::AddScalar(desc) => self
                .register_scalar_ops(desc, |lhs, rhs, out| {
                    FuseOp::Add(BinaryFuseArgs { lhs, rhs, out })
                }),
            NumericOperationIr::Sub(desc) => self.register_binary_ops(desc, |lhs, rhs, out| {
                FuseOp::Sub(BinaryFuseArgs { lhs, rhs, out })
            }),
            NumericOperationIr::SubScalar(desc) => self
                .register_scalar_ops(desc, |lhs, rhs, out| {
                    FuseOp::Sub(BinaryFuseArgs { lhs, rhs, out })
                }),
            NumericOperationIr::Mul(desc) => self.register_binary_ops(desc, |lhs, rhs, out| {
                FuseOp::Mul(BinaryFuseArgs { lhs, rhs, out })
            }),
            NumericOperationIr::MulScalar(desc) => self
                .register_scalar_ops(desc, |lhs, rhs, out| {
                    FuseOp::Mul(BinaryFuseArgs { lhs, rhs, out })
                }),
            NumericOperationIr::Div(desc) => self.register_binary_ops(desc, |lhs, rhs, out| {
                FuseOp::Div(BinaryFuseArgs { lhs, rhs, out })
            }),
            NumericOperationIr::DivScalar(desc) => self
                .register_scalar_ops(desc, |lhs, rhs, out| {
                    FuseOp::Div(BinaryFuseArgs { lhs, rhs, out })
                }),
            NumericOperationIr::Abs(desc) => self
                .register_unary_ops(desc, |input, out| FuseOp::Abs(UnaryFuseArgs { input, out })),
            NumericOperationIr::Lower(desc) => self.register_binary_ops(desc, |lhs, rhs, out| {
                FuseOp::Lower(BinaryFuseArgs { lhs, rhs, out })
            }),
            NumericOperationIr::LowerElem(desc) => self
                .register_scalar_ops(desc, |lhs, rhs, out| {
                    FuseOp::Lower(BinaryFuseArgs { lhs, rhs, out })
                }),
            NumericOperationIr::Greater(desc) => self.register_binary_ops(desc, |lhs, rhs, out| {
                FuseOp::Greater(BinaryFuseArgs { lhs, rhs, out })
            }),
            NumericOperationIr::GreaterElem(desc) => self
                .register_scalar_ops(desc, |lhs, rhs, out| {
                    FuseOp::Greater(BinaryFuseArgs { lhs, rhs, out })
                }),
            NumericOperationIr::LowerEqual(desc) => self
                .register_binary_ops(desc, |lhs, rhs, out| {
                    FuseOp::LowerEqual(BinaryFuseArgs { lhs, rhs, out })
                }),
            NumericOperationIr::LowerEqualElem(desc) => self
                .register_scalar_ops(desc, |lhs, rhs, out| {
                    FuseOp::LowerEqual(BinaryFuseArgs { lhs, rhs, out })
                }),
            NumericOperationIr::GreaterEqual(desc) => self
                .register_binary_ops(desc, |lhs, rhs, out| {
                    FuseOp::GreaterEqual(BinaryFuseArgs { lhs, rhs, out })
                }),
            NumericOperationIr::GreaterEqualElem(desc) => self
                .register_scalar_ops(desc, |lhs, rhs, out| {
                    FuseOp::GreaterEqual(BinaryFuseArgs { lhs, rhs, out })
                }),
            NumericOperationIr::EqualElem(desc) => self
                .register_scalar_ops(desc, |lhs, rhs, out| {
                    FuseOp::Equal(BinaryFuseArgs { lhs, rhs, out })
                }),
            NumericOperationIr::MaskWhere(desc) => {
                if !self.output_is_compatible(&desc.out) {
                    return false;
                }

                self.builder.register(|build| {
                    let cond = build.input(&desc.mask)?;
                    let rhs = build.input(&desc.tensor)?;
                    let lhs = build.input(&desc.value)?;
                    let out = build.output(&desc.out)?;

                    build.register_operation(FuseOp::ConditionalAssign {
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

                    build.register_operation(FuseOp::ConditionalAssign {
                        cond,
                        lhs,
                        rhs,
                        out,
                    });

                    Some(())
                })
            }
            NumericOperationIr::Full(desc) => {
                if !self.output_is_compatible(&desc.out) {
                    return false;
                }

                self.builder.register(|build| {
                    let input = build.scalar(&desc.value, desc.out.dtype);
                    let out = build.output(&desc.out)?;

                    build.register_operation(FuseOp::Assign(UnaryFuseArgs { input, out }));

                    Some(())
                })
            }
            NumericOperationIr::Gather(desc) => {
                if !self.output_is_compatible(&desc.out) {
                    return false;
                }

                self.builder.register(|build| {
                    let input = build.input_indexed(&desc.tensor)?;
                    let indices = build.input_indexed(&desc.indices)?;
                    let output = build.output(&desc.out)?;

                    build.register_operation(FuseOp::Gather {
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

                    build.register_operation(FuseOp::Select {
                        input,
                        indices,
                        output,
                        dim: desc.dim as u32,
                    });

                    Some(())
                })
            }
            NumericOperationIr::Rem(desc) => self.register_binary_ops(desc, |lhs, rhs, out| {
                FuseOp::Rem(BinaryFuseArgs { lhs, rhs, out })
            }),
            NumericOperationIr::RemScalar(desc) => self
                .register_scalar_ops(desc, |lhs, rhs, out| {
                    FuseOp::Rem(BinaryFuseArgs { lhs, rhs, out })
                }),
            NumericOperationIr::Powf(desc) => self.register_binary_ops(desc, |lhs, rhs, out| {
                FuseOp::Powf(BinaryFuseArgs { lhs, rhs, out })
            }),
            NumericOperationIr::Clamp(desc) => {
                if !self.output_is_compatible(&desc.out) {
                    return false;
                }

                self.builder.register(|build| {
                    let input = build.input(&desc.tensor)?;
                    let min = build.scalar(&desc.min, desc.out.dtype);
                    let max = build.scalar(&desc.max, desc.out.dtype);
                    let out = build.output(&desc.out)?;

                    build.register_operation(FuseOp::Clamp {
                        input,
                        min,
                        max,
                        out,
                    });

                    Some(())
                })
            }
            _ => false,
        }
    }

    fn register_binary_ops<Func>(&mut self, desc: &BinaryOpIr, func: Func) -> bool
    where
        Func: Fn(FuseArg, FuseArg, FuseArg) -> FuseOp,
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
        Func: Fn(FuseArg, FuseArg) -> FuseOp,
    {
        self.register_unary_op(&desc.input, &desc.out, func)
    }

    fn register_unary_op<Func>(&mut self, input: &TensorIr, out: &TensorIr, func: Func) -> bool
    where
        Func: Fn(FuseArg, FuseArg) -> FuseOp,
    {
        if !self.output_is_compatible(out) {
            return false;
        }

        self.builder.register(|build| {
            let input = build.input(input)?;
            let out = build.output(out)?;
            build.register_operation(func(input, out));
            Some(())
        })
    }

    fn register_scalar_ops<Func>(&mut self, desc: &ScalarOpIr, func: Func) -> bool
    where
        Func: Fn(FuseArg, FuseArg, FuseArg) -> FuseOp,
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
            self.current_output_shape.clone_from(&out.shape.dims);
            return true;
        }

        let rank = self.current_output_shape.len();

        // Rank should be equal.
        if rank != out.shape.num_dims() {
            return false;
        }

        let mut updated = self.current_output_shape.clone();
        let mut should_update = false;

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
                should_update = true;
                updated[i] = new;
                continue;
            }

            return false;
        }

        if should_update {
            // For now forced to have exact shape.
            if updated != out.shape.dims {
                return false;
            }

            self.current_output_shape.clone_from_slice(&out.shape.dims);
        }

        true
    }
}

#[derive(Debug, Clone)]
/// Builder wrapper to limit the number of bindings in generated kernels.
struct TryFuseBuilder {
    builder: FuseTraceBuilder,
    max_bindings: u32,
    max_ops: u32,
    added_ops: bool,
}

impl TryFuseBuilder {
    fn new(max_bindings: u32, bool_precision: FuseType, settings: FuseSettings) -> Self {
        Self {
            builder: FuseTraceBuilder::new(bool_precision, settings),
            max_bindings,
            // A good default, avoid errors with for loops over only memory
            // bound operations.
            max_ops: 64,
            added_ops: false,
        }
    }

    fn register(&mut self, add_ops: impl FnOnce(&mut FuseTraceBuilder) -> Option<()>) -> bool {
        if self.builder.num_ops_fused() > self.max_ops {
            return false;
        }

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

    fn build(&self, shape: Vec<usize>) -> FuseTrace {
        self.builder.build(shape)
    }
}
