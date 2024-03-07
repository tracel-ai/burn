use std::marker::PhantomData;

use burn_jit::gpu;

use crate::element::{FloatElement, IntElement};

#[derive(new, Clone, Debug, Default)]
pub struct CudaCompiler<F: FloatElement, I: IntElement> {
    shape: bool,
    stride: bool,
    num_inputs: usize,
    num_outputs: usize,
    _f: PhantomData<F>,
    _i: PhantomData<I>,
}

impl<F: FloatElement, I: IntElement> burn_jit::Compiler for CudaCompiler<F, I> {
    type Representation = super::ComputeShader;
    type Float = f32;
    type Int = i32;

    type FullPrecisionCompiler = CudaCompiler<f32, i32>;

    fn compile(shader: burn_jit::gpu::ComputeShader) -> Self::Representation {
        let mut compiler = Self::default();
        compiler.compile_shader(shader)
    }

    fn elem_size(elem: burn_jit::gpu::Elem) -> usize {
        Self::compile_elem(elem).size()
    }
}

impl<F: FloatElement, I: IntElement> CudaCompiler<F, I> {
    fn compile_shader(&mut self, mut value: gpu::ComputeShader) -> super::ComputeShader {
        self.num_inputs = value.inputs.len();
        self.num_outputs = value.outputs.len();

        let instructions = self.compile_scope(&mut value.body);
        let body = super::Body {
            instructions,
            rank: true,
            id: true,
            stride: true,
            shape: true,
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
            shared_memories: Vec::new(),
            workgroup_size: value.workgroup_size,
            body,
        }
    }

    fn compile_scope(&mut self, value: &mut gpu::Scope) -> Vec<super::Instruction> {
        let mut instructions = Vec::new();
        let processing = value.process();

        for var in processing.variables {
            instructions.push(super::Instruction::DeclareVariable {
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
        instructions: &mut Vec<super::Instruction>,
        operation: gpu::Operation,
        scope: &mut gpu::Scope,
    ) {
        match operation {
            gpu::Operation::Operator(op) => instructions.push(self.compile_instruction(op)),
            gpu::Operation::Procedure(proc) => self.compile_procedure(instructions, proc, scope),
            gpu::Operation::Metadata(op) => instructions.push(self.compile_metadata(op)),
            gpu::Operation::Branch(val) => self.compile_branch(instructions, val),
            gpu::Operation::Synchronization(val) => todo!(),
        }
    }

    fn compile_metadata(&mut self, metadata: gpu::Metadata) -> super::Instruction {
        match metadata {
            gpu::Metadata::Stride { dim, var, out } => {
                self.stride = true;
                let position = match var {
                    gpu::Variable::GlobalInputArray(idx, _) => idx as usize,
                    gpu::Variable::GlobalOutputArray(idx, _) => self.num_inputs + idx as usize,
                    _ => panic!("Only Input and Output have a stride, got: {:?}", var),
                };
                super::Instruction::Stride {
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
                super::Instruction::Shape {
                    dim: self.compile_variable(dim),
                    position,
                    out: self.compile_variable(out),
                }
            }
            gpu::Metadata::ArrayLength { var, out } => todo!(),
        }
    }

    fn compile_branch(&mut self, instructions: &mut Vec<super::Instruction>, branch: gpu::Branch) {
        match branch {
            gpu::Branch::If(mut op) => instructions.push(super::Instruction::If {
                cond: self.compile_variable(op.cond),
                instructions: self.compile_scope(&mut op.scope),
            }),
            gpu::Branch::IfElse(mut op) => instructions.push(super::Instruction::IfElse {
                cond: self.compile_variable(op.cond),
                instructions_if: self.compile_scope(&mut op.scope_if),
                instructions_else: self.compile_scope(&mut op.scope_else),
            }),
            gpu::Branch::Return => instructions.push(super::Instruction::Return),
            gpu::Branch::Break => instructions.push(super::Instruction::Break),
            gpu::Branch::RangeLoop(mut range_loop) => {
                instructions.push(super::Instruction::RangeLoop {
                    i: self.compile_variable(range_loop.i),
                    start: self.compile_variable(range_loop.start),
                    end: self.compile_variable(range_loop.end),
                    instructions: self.compile_scope(&mut range_loop.scope),
                })
            }
            gpu::Branch::Loop(mut op) => instructions.push(super::Instruction::Loop {
                instructions: self.compile_scope(&mut op.scope),
            }),
        };
    }
    fn compile_procedure(
        &mut self,
        instructions: &mut Vec<super::Instruction>,
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
            gpu::Procedure::IndexOffsetGlobalWithLayout(proc) => {
                proc.expand(scope);
                compile(scope);
            }
        }
    }

    fn compile_instruction(&mut self, value: gpu::Operator) -> super::Instruction {
        match value {
            gpu::Operator::Add(op) => super::Instruction::Add {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(op.out),
            },
            gpu::Operator::Mul(op) => super::Instruction::Mul {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(op.out),
            },
            gpu::Operator::Div(op) => super::Instruction::Div {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(op.out),
            },
            gpu::Operator::Sub(op) => super::Instruction::Sub {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(op.out),
            },
            gpu::Operator::Assign(op) => super::Instruction::Assign {
                input: self.compile_variable(op.input),
                out: self.compile_variable(op.out),
            },
            gpu::Operator::Index(op) => super::Instruction::Index {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(op.out),
            },
            gpu::Operator::IndexAssign(op) => super::Instruction::IndexAssign {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(op.out),
            },
            gpu::Operator::Modulo(op) => super::Instruction::Modulo {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(op.out),
            },
            gpu::Operator::Equal(op) => super::Instruction::Equal {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(op.out),
            },
            gpu::Operator::Lower(op) => super::Instruction::Lower {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(op.out),
            },
            gpu::Operator::Greater(op) => super::Instruction::Greater {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(op.out),
            },
            gpu::Operator::LowerEqual(op) => super::Instruction::LowerEqual {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(op.out),
            },
            gpu::Operator::GreaterEqual(op) => super::Instruction::GreaterEqual {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(op.out),
            },

            gpu::Operator::Abs(_) => todo!(),
            gpu::Operator::Exp(_) => todo!(),
            gpu::Operator::Log(_) => todo!(),
            gpu::Operator::Log1p(_) => todo!(),
            gpu::Operator::Cos(_) => todo!(),
            gpu::Operator::Sin(_) => todo!(),
            gpu::Operator::Tanh(_) => todo!(),
            gpu::Operator::Powf(_) => todo!(),
            gpu::Operator::Sqrt(_) => todo!(),
            gpu::Operator::Erf(op) => super::Instruction::Erf {
                input: self.compile_variable(op.input),
                out: self.compile_variable(op.out),
            },
            gpu::Operator::Recip(_) => todo!(),
            gpu::Operator::Clamp(_) => todo!(),
            gpu::Operator::And(_) => todo!(),
            gpu::Operator::Or(_) => todo!(),
            gpu::Operator::Not(_) => todo!(),
            gpu::Operator::Max(_) => todo!(),
            gpu::Operator::Min(_) => todo!(),
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
                todo!()
            }
            gpu::Variable::Id => super::Variable::Id,
            gpu::Variable::Rank => super::Variable::Rank,
            gpu::Variable::LocalInvocationIndex => super::Variable::LocalInvocationIndex,
            gpu::Variable::LocalInvocationIdX => super::Variable::LocalInvocationIdX,
            gpu::Variable::LocalInvocationIdY => super::Variable::LocalInvocationIdY,
            gpu::Variable::LocalInvocationIdZ => super::Variable::LocalInvocationIdZ,
            gpu::Variable::WorkgroupIdX => super::Variable::WorkgroupIdX,
            gpu::Variable::WorkgroupIdY => super::Variable::WorkgroupIdY,
            gpu::Variable::WorkgroupIdZ => super::Variable::WorkgroupIdZ,
            gpu::Variable::GlobalInvocationIdX => super::Variable::GlobalInvocationIdX,
            gpu::Variable::GlobalInvocationIdY => super::Variable::GlobalInvocationIdY,
            gpu::Variable::GlobalInvocationIdZ => super::Variable::GlobalInvocationIdZ,
            gpu::Variable::WorkgroupSizeX => super::Variable::WorkgroupSizeX,
            gpu::Variable::WorkgroupSizeY => super::Variable::WorkgroupSizeY,
            gpu::Variable::WorkgroupSizeZ => super::Variable::WorkgroupSizeZ,
            gpu::Variable::NumWorkgroupsX => super::Variable::NumWorkgroupsX,
            gpu::Variable::NumWorkgroupsY => super::Variable::NumWorkgroupsY,
            gpu::Variable::NumWorkgroupsZ => super::Variable::NumWorkgroupsZ,
        }
    }

    fn compile_binding(binding: gpu::Binding) -> super::Binding {
        super::Binding {
            item: Self::compile_item(binding.item),
            size: binding.size,
        }
    }

    fn compile_item(item: gpu::Item) -> super::Item {
        match item {
            gpu::Item::Vec4(elem) => super::Item::Vec4(Self::compile_elem(elem)),
            gpu::Item::Vec3(elem) => super::Item::Vec3(Self::compile_elem(elem)),
            gpu::Item::Vec2(elem) => super::Item::Vec2(Self::compile_elem(elem)),
            gpu::Item::Scalar(elem) => super::Item::Scalar(Self::compile_elem(elem)),
        }
    }

    fn compile_elem(value: gpu::Elem) -> super::Elem {
        match value {
            gpu::Elem::Float => F::cuda_elem(),
            gpu::Elem::Int => I::cuda_elem(),
            gpu::Elem::UInt => super::Elem::U32,
            gpu::Elem::Bool => super::Elem::Bool,
        }
    }
}
