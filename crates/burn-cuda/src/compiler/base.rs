use burn_cube::{ir as gpu, Compiler};

use super::{Instruction, WarpInstruction};

#[allow(clippy::too_many_arguments)]
#[derive(new, Clone, Debug, Default)]
pub struct CudaCompiler {
    shape: bool,
    stride: bool,
    num_inputs: usize,
    num_outputs: usize,
    shared_memories: Vec<super::SharedMemory>,
    local_arrays: Vec<super::LocalArray>,
    id: bool,
    rank: bool,
    invocation_index: bool,
    global_invocation_id: (bool, bool, bool),
    wrap_size_checked: bool,
}

impl Compiler for CudaCompiler {
    type Representation = super::ComputeShader;

    fn compile(shader: burn_cube::ir::KernelDefinition) -> Self::Representation {
        let compiler = Self::default();
        compiler.compile_shader(shader)
    }

    fn elem_size(elem: gpu::Elem) -> usize {
        Self::compile_elem(elem).size()
    }

    fn max_shared_memory_size() -> usize {
        // TODO: Find out this value.
        usize::MAX
    }
}

impl CudaCompiler {
    fn compile_shader(mut self, mut value: gpu::KernelDefinition) -> super::ComputeShader {
        self.num_inputs = value.inputs.len();
        self.num_outputs = value.outputs.len();

        let instructions = self.compile_scope(&mut value.body);
        let body = super::Body {
            instructions,
            stride: true,
            shape: true,
            shared_memories: self.shared_memories,
            local_arrays: self.local_arrays,
            rank: self.rank,
            id: self.id,
            invocation_index: self.invocation_index,
            global_invocation_id: self.global_invocation_id,
            wrap_size_checked: self.wrap_size_checked,
        };

        super::ComputeShader {
            inputs: value
                .inputs
                .into_iter()
                .map(Self::compile_binding)
                .collect(),
            outputs: value
                .outputs
                .into_iter()
                .map(Self::compile_binding)
                .collect(),
            named: value
                .named
                .into_iter()
                .map(|(name, binding)| (name, Self::compile_binding(binding)))
                .collect(),
            cube_dim: value.cube_dim,
            body,
        }
    }

    fn compile_scope(&mut self, value: &mut gpu::Scope) -> Vec<Instruction> {
        let mut instructions = Vec::new();
        let processing = value.process();

        for var in processing.variables {
            instructions.push(Instruction::DeclareVariable {
                var: self.compile_variable(var),
            });
        }

        processing
            .operations
            .into_iter()
            .for_each(|op| self.compile_operation(&mut instructions, op, value));

        instructions
    }

    fn compile_operation(
        &mut self,
        instructions: &mut Vec<Instruction>,
        operation: gpu::Operation,
        scope: &mut gpu::Scope,
    ) {
        match operation {
            gpu::Operation::Operator(op) => instructions.push(self.compile_instruction(op)),
            gpu::Operation::Procedure(proc) => self.compile_procedure(instructions, proc, scope),
            gpu::Operation::Metadata(op) => instructions.push(self.compile_metadata(op)),
            gpu::Operation::Branch(val) => self.compile_branch(instructions, val),
            gpu::Operation::Synchronization(val) => match val {
                gpu::Synchronization::SyncUnits => instructions.push(Instruction::SyncThreads),
            },
            gpu::Operation::Subcube(op) => {
                self.wrap_size_checked = true;
                match op {
                    gpu::Subcube::Sum(op) => {
                        instructions.push(Instruction::Wrap(WarpInstruction::ReduceSum {
                            input: self.compile_variable(op.input),
                            out: self.compile_variable(op.out),
                        }))
                    }
                    gpu::Subcube::Prod(op) => {
                        instructions.push(Instruction::Wrap(WarpInstruction::ReduceProd {
                            input: self.compile_variable(op.input),
                            out: self.compile_variable(op.out),
                        }))
                    }
                    gpu::Subcube::Max(op) => {
                        instructions.push(Instruction::Wrap(WarpInstruction::ReduceMax {
                            input: self.compile_variable(op.input),
                            out: self.compile_variable(op.out),
                        }))
                    }

                    gpu::Subcube::Min(op) => {
                        instructions.push(Instruction::Wrap(WarpInstruction::ReduceMin {
                            input: self.compile_variable(op.input),
                            out: self.compile_variable(op.out),
                        }))
                    }

                    _ => todo!(),
                }
            }
        }
    }

    fn compile_metadata(&mut self, metadata: gpu::Metadata) -> Instruction {
        match metadata {
            gpu::Metadata::Stride { dim, var, out } => {
                self.stride = true;
                let position = match var {
                    gpu::Variable::GlobalInputArray(idx, _) => idx as usize,
                    gpu::Variable::GlobalOutputArray(idx, _) => self.num_inputs + idx as usize,
                    _ => panic!("Only Input and Output have a stride, got: {:?}", var),
                };
                Instruction::Stride {
                    dim: self.compile_variable(dim),
                    position,
                    out: self.compile_variable(out),
                }
            }
            gpu::Metadata::Shape { dim, var, out } => {
                self.shape = true;
                let position = match var {
                    gpu::Variable::GlobalInputArray(idx, _) => idx as usize,
                    gpu::Variable::GlobalOutputArray(idx, _) => self.num_inputs + idx as usize,
                    _ => panic!("Only Input and Output have a shape, got {:?}", var),
                };
                Instruction::Shape {
                    dim: self.compile_variable(dim),
                    position,
                    out: self.compile_variable(out),
                }
            }
            gpu::Metadata::ArrayLength { var, out } => super::Instruction::ArrayLength {
                input: self.compile_variable(var),
                out: self.compile_variable(out),
                num_inputs: self.num_inputs,
                num_outputs: self.num_outputs,
            },
        }
    }

    fn compile_branch(&mut self, instructions: &mut Vec<Instruction>, branch: gpu::Branch) {
        match branch {
            gpu::Branch::If(mut op) => instructions.push(Instruction::If {
                cond: self.compile_variable(op.cond),
                instructions: self.compile_scope(&mut op.scope),
            }),
            gpu::Branch::IfElse(mut op) => instructions.push(Instruction::IfElse {
                cond: self.compile_variable(op.cond),
                instructions_if: self.compile_scope(&mut op.scope_if),
                instructions_else: self.compile_scope(&mut op.scope_else),
            }),
            gpu::Branch::Return => instructions.push(Instruction::Return),
            gpu::Branch::Break => instructions.push(Instruction::Break),
            gpu::Branch::RangeLoop(mut range_loop) => instructions.push(Instruction::RangeLoop {
                i: self.compile_variable(range_loop.i),
                start: self.compile_variable(range_loop.start),
                end: self.compile_variable(range_loop.end),
                instructions: self.compile_scope(&mut range_loop.scope),
            }),
            gpu::Branch::Loop(mut op) => instructions.push(Instruction::Loop {
                instructions: self.compile_scope(&mut op.scope),
            }),
        };
    }
    fn compile_procedure(
        &mut self,
        instructions: &mut Vec<Instruction>,
        proc: gpu::Procedure,
        scope: &mut gpu::Scope,
    ) {
        let mut compile = |scope: &mut gpu::Scope| {
            instructions.extend(self.compile_scope(scope));
        };

        match proc {
            gpu::Procedure::ReadGlobalWithLayout(proc) => {
                proc.expand(scope);
                compile(scope);
            }
            gpu::Procedure::ReadGlobal(proc) => {
                proc.expand(scope);
                compile(scope);
            }
            gpu::Procedure::WriteGlobal(proc) => {
                proc.expand(scope);
                compile(scope);
            }
            gpu::Procedure::ConditionalAssign(proc) => {
                proc.expand(scope);
                compile(scope);
            }
            gpu::Procedure::CheckedIndex(proc) => {
                proc.expand(scope);
                compile(scope);
            }
            gpu::Procedure::CheckedIndexAssign(proc) => {
                proc.expand(scope);
                compile(scope);
            }
            gpu::Procedure::IndexOffsetGlobalWithLayout(proc) => {
                proc.expand(scope);
                compile(scope);
            }
        }
    }

    fn compile_instruction(&mut self, value: gpu::Operator) -> Instruction {
        match value {
            gpu::Operator::Add(op) => Instruction::Add(self.compile_binary(op)),
            gpu::Operator::Mul(op) => Instruction::Mul(self.compile_binary(op)),
            gpu::Operator::Div(op) => Instruction::Div(self.compile_binary(op)),
            gpu::Operator::Sub(op) => Instruction::Sub(self.compile_binary(op)),
            gpu::Operator::Assign(op) => Instruction::Assign(self.compile_unary(op)),
            gpu::Operator::Index(op) => Instruction::Index(self.compile_binary(op)),
            gpu::Operator::UncheckedIndex(op) => Instruction::Index(self.compile_binary(op)),
            gpu::Operator::IndexAssign(op) => Instruction::IndexAssign(self.compile_binary(op)),
            gpu::Operator::UncheckedIndexAssign(op) => {
                Instruction::IndexAssign(self.compile_binary(op))
            }
            gpu::Operator::Modulo(op) => Instruction::Modulo(self.compile_binary(op)),
            gpu::Operator::Equal(op) => Instruction::Equal(self.compile_binary(op)),
            gpu::Operator::Lower(op) => Instruction::Lower(self.compile_binary(op)),
            gpu::Operator::Greater(op) => Instruction::Greater(self.compile_binary(op)),
            gpu::Operator::LowerEqual(op) => Instruction::LowerEqual(self.compile_binary(op)),
            gpu::Operator::GreaterEqual(op) => Instruction::GreaterEqual(self.compile_binary(op)),
            gpu::Operator::Abs(op) => Instruction::Abs(self.compile_unary(op)),
            gpu::Operator::Exp(op) => Instruction::Exp(self.compile_unary(op)),
            gpu::Operator::Log(op) => Instruction::Log(self.compile_unary(op)),
            gpu::Operator::Log1p(op) => Instruction::Log1p(self.compile_unary(op)),
            gpu::Operator::Cos(op) => Instruction::Cos(self.compile_unary(op)),
            gpu::Operator::Sin(op) => Instruction::Sin(self.compile_unary(op)),
            gpu::Operator::Tanh(op) => Instruction::Tanh(self.compile_unary(op)),
            gpu::Operator::Powf(op) => Instruction::Powf(self.compile_binary(op)),
            gpu::Operator::Sqrt(op) => Instruction::Sqrt(self.compile_unary(op)),
            gpu::Operator::Erf(op) => Instruction::Erf(self.compile_unary(op)),
            gpu::Operator::And(op) => Instruction::And(self.compile_binary(op)),
            gpu::Operator::Or(op) => Instruction::Or(self.compile_binary(op)),
            gpu::Operator::Not(op) => Instruction::Not(self.compile_unary(op)),
            gpu::Operator::Max(op) => Instruction::Max(self.compile_binary(op)),
            gpu::Operator::Min(op) => Instruction::Min(self.compile_binary(op)),
            gpu::Operator::NotEqual(op) => Instruction::NotEqual(self.compile_binary(op)),
            gpu::Operator::BitwiseAnd(op) => Instruction::BitwiseAnd(self.compile_binary(op)),
            gpu::Operator::BitwiseXor(op) => Instruction::BitwiseXor(self.compile_binary(op)),
            gpu::Operator::ShiftLeft(op) => Instruction::ShiftLeft(self.compile_binary(op)),
            gpu::Operator::ShiftRight(op) => Instruction::ShiftRight(self.compile_binary(op)),
            gpu::Operator::Clamp(op) => Instruction::Clamp {
                input: self.compile_variable(op.input),
                min_value: self.compile_variable(op.min_value),
                max_value: self.compile_variable(op.max_value),
                out: self.compile_variable(op.out),
            },
            gpu::Operator::Recip(op) => Instruction::Div(super::BinaryInstruction {
                lhs: super::Variable::ConstantScalar(
                    1.0,
                    Self::compile_elem(op.input.item().elem()),
                ),
                rhs: self.compile_variable(op.input),
                out: self.compile_variable(op.out),
            }),
            gpu::Operator::Floor(op) => Instruction::Floor(self.compile_unary(op)),
            gpu::Operator::Ceil(op) => Instruction::Ceil(self.compile_unary(op)),
            gpu::Operator::Remainder(_op) => todo!(),
        }
    }

    fn compile_binary(&mut self, value: gpu::BinaryOperator) -> super::BinaryInstruction {
        super::BinaryInstruction {
            lhs: self.compile_variable(value.lhs),
            rhs: self.compile_variable(value.rhs),
            out: self.compile_variable(value.out),
        }
    }

    fn compile_unary(&mut self, value: gpu::UnaryOperator) -> super::UnaryInstruction {
        super::UnaryInstruction {
            input: self.compile_variable(value.input),
            out: self.compile_variable(value.out),
        }
    }

    fn compile_variable(&mut self, value: gpu::Variable) -> super::Variable {
        match value {
            gpu::Variable::GlobalInputArray(index, item) => {
                super::Variable::GlobalInputArray(index, Self::compile_item(item))
            }
            gpu::Variable::GlobalScalar(index, elem) => {
                super::Variable::GlobalScalar(index, Self::compile_elem(elem), elem)
            }
            gpu::Variable::Local(index, item, scope_depth) => super::Variable::Local {
                index,
                item: Self::compile_item(item),
                scope_depth,
            },
            gpu::Variable::LocalScalar(index, elem, scope_depth) => super::Variable::LocalScalar {
                index,
                elem: Self::compile_elem(elem),
                scope_depth,
            },
            gpu::Variable::GlobalOutputArray(index, item) => {
                super::Variable::GlobalOutputArray(index, Self::compile_item(item))
            }
            gpu::Variable::ConstantScalar(index, elem) => {
                super::Variable::ConstantScalar(index, Self::compile_elem(elem))
            }
            gpu::Variable::SharedMemory(index, item, size) => {
                let item = Self::compile_item(item);
                if !self.shared_memories.iter().any(|s| s.index == index) {
                    self.shared_memories
                        .push(super::SharedMemory::new(index, item, size));
                }
                super::Variable::SharedMemory(index, item, size)
            }
            gpu::Variable::AbsolutePos => {
                self.id = true;
                super::Variable::Id
            }
            gpu::Variable::Rank => {
                self.rank = true;
                super::Variable::Rank
            }
            gpu::Variable::UnitPos => {
                self.invocation_index = true;
                super::Variable::LocalInvocationIndex
            }
            gpu::Variable::UnitPosX => super::Variable::LocalInvocationIdX,
            gpu::Variable::UnitPosY => super::Variable::LocalInvocationIdY,
            gpu::Variable::UnitPosZ => super::Variable::LocalInvocationIdZ,
            gpu::Variable::CubePosX => super::Variable::WorkgroupIdX,
            gpu::Variable::CubePosY => super::Variable::WorkgroupIdY,
            gpu::Variable::CubePosZ => super::Variable::WorkgroupIdZ,
            gpu::Variable::AbsolutePosX => {
                self.global_invocation_id.0 = true;
                super::Variable::GlobalInvocationIdX
            }
            gpu::Variable::AbsolutePosY => {
                self.global_invocation_id.1 = true;
                super::Variable::GlobalInvocationIdY
            }
            gpu::Variable::AbsolutePosZ => {
                self.global_invocation_id.2 = true;
                super::Variable::GlobalInvocationIdZ
            }
            gpu::Variable::CubeDimX => super::Variable::WorkgroupSizeX,
            gpu::Variable::CubeDimY => super::Variable::WorkgroupSizeY,
            gpu::Variable::CubeDimZ => super::Variable::WorkgroupSizeZ,
            gpu::Variable::CubeCountX => super::Variable::NumWorkgroupsX,
            gpu::Variable::CubeCountY => super::Variable::NumWorkgroupsY,
            gpu::Variable::CubeCountZ => super::Variable::NumWorkgroupsZ,
            gpu::Variable::LocalArray(id, item, depth, size) => {
                let item = Self::compile_item(item);
                if !self
                    .local_arrays
                    .iter()
                    .any(|s| s.index == id && s.depth == depth)
                {
                    self.local_arrays
                        .push(super::LocalArray::new(id, item, depth, size));
                }
                super::Variable::LocalArray(id, item, depth, size)
            }
            gpu::Variable::CubePos => todo!(),
            gpu::Variable::CubeDim => todo!(),
            gpu::Variable::CubeCount => todo!(),
            gpu::Variable::SubcubeDim => todo!(),
        }
    }

    fn compile_binding(binding: gpu::Binding) -> super::Binding {
        super::Binding {
            item: Self::compile_item(binding.item),
            size: binding.size,
        }
    }

    fn compile_item(item: gpu::Item) -> super::Item {
        match item.vectorization {
            4 => super::Item::Vec4(Self::compile_elem(item.elem)),
            3 => super::Item::Vec3(Self::compile_elem(item.elem)),
            2 => super::Item::Vec2(Self::compile_elem(item.elem)),
            1 => super::Item::Scalar(Self::compile_elem(item.elem)),
            _ => panic!("Vectorization factor unsupported {:?}", item.vectorization),
        }
    }

    fn compile_elem(value: gpu::Elem) -> super::Elem {
        match value {
            gpu::Elem::Float(kind) => match kind {
                gpu::FloatKind::F16 => super::Elem::F16,
                gpu::FloatKind::BF16 => super::Elem::BF16,
                gpu::FloatKind::F32 => super::Elem::F32,
                gpu::FloatKind::F64 => panic!("f64 isn't supported yet"),
            },
            gpu::Elem::Int(kind) => match kind {
                gpu::IntKind::I32 => super::Elem::I32,
                gpu::IntKind::I64 => panic!("i64 isn't supported yet"),
            },
            gpu::Elem::UInt => super::Elem::U32,
            gpu::Elem::Bool => super::Elem::Bool,
        }
    }
}
