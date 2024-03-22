use super::LocalArray;
use super::{shader::ComputeShader, Item, SharedMemory};
use crate::compiler::wgsl;
use crate::{FloatElement, IntElement};
use burn_jit::gpu;
use std::marker::PhantomData;

/// Wgsl Compiler.
#[derive(Clone)]
pub struct WgslCompiler<F: FloatElement, I: IntElement> {
    num_inputs: usize,
    num_outputs: usize,
    local_invocation_index: bool,
    local_invocation_id: bool,
    global_invocation_id: bool,
    workgroup_id: bool,
    rank: bool,
    id: bool,
    stride: bool,
    shape: bool,
    num_workgroups: bool,
    shared_memories: Vec<SharedMemory>,
    local_arrays: Vec<LocalArray>,
    _float: PhantomData<F>,
    _int: PhantomData<I>,
}

impl<F: FloatElement, I: IntElement> core::fmt::Debug for WgslCompiler<F, I> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("WgslCompiler")
    }
}

impl<F: FloatElement, I: IntElement> Default for WgslCompiler<F, I> {
    fn default() -> Self {
        Self {
            num_inputs: 0,
            num_outputs: 0,
            local_invocation_index: false,
            local_invocation_id: false,
            global_invocation_id: false,
            workgroup_id: false,
            rank: false,
            id: false,
            stride: false,
            shape: false,
            num_workgroups: false,
            shared_memories: Vec::default(),
            local_arrays: Vec::default(),
            _float: PhantomData,
            _int: PhantomData,
        }
    }
}

impl<F: FloatElement, I: IntElement> burn_jit::Compiler for WgslCompiler<F, I> {
    type Representation = ComputeShader;
    type Float = F;
    type Int = I;
    type FullPrecisionCompiler = WgslCompiler<f32, i32>;

    fn compile(shader: gpu::ComputeShader) -> Self::Representation {
        let mut compiler = Self::default();
        compiler.compile_shader(shader)
    }

    fn elem_size(elem: gpu::Elem) -> usize {
        Self::compile_elem(elem).size()
    }

    fn max_shared_memory_size() -> usize {
        8192
    }
}

impl<F: FloatElement, I: IntElement> WgslCompiler<F, I> {
    fn compile_shader(&mut self, mut value: gpu::ComputeShader) -> wgsl::ComputeShader {
        self.num_inputs = value.inputs.len();
        self.num_outputs = value.outputs.len();

        let instructions = self.compile_scope(&mut value.body);
        let extensions = register_extensions(&instructions);
        let body = wgsl::Body {
            instructions,
            rank: true,
            id: self.id,
            stride: self.stride,
            shape: self.shape,
        };

        wgsl::ComputeShader {
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
            shared_memories: self.shared_memories.clone(),
            local_arrays: self.local_arrays.clone(),
            workgroup_size: value.workgroup_size,
            global_invocation_id: self.global_invocation_id || self.id,
            local_invocation_index: self.local_invocation_index,
            local_invocation_id: self.local_invocation_id,
            num_workgroups: self.id || self.num_workgroups,
            workgroup_id: self.workgroup_id,
            body,
            extensions,
        }
    }

    fn compile_item(item: gpu::Item) -> Item {
        match item {
            gpu::Item::Vec4(elem) => wgsl::Item::Vec4(Self::compile_elem(elem)),
            gpu::Item::Vec3(elem) => wgsl::Item::Vec3(Self::compile_elem(elem)),
            gpu::Item::Vec2(elem) => wgsl::Item::Vec2(Self::compile_elem(elem)),
            gpu::Item::Scalar(elem) => wgsl::Item::Scalar(Self::compile_elem(elem)),
        }
    }

    fn compile_elem(value: gpu::Elem) -> wgsl::Elem {
        match value {
            gpu::Elem::Float => F::wgpu_elem(),
            gpu::Elem::Int => I::wgpu_elem(),
            gpu::Elem::UInt => wgsl::Elem::U32,
            gpu::Elem::Bool => wgsl::Elem::Bool,
        }
    }

    fn compile_variable(&mut self, value: gpu::Variable) -> wgsl::Variable {
        match value {
            gpu::Variable::GlobalInputArray(index, item) => {
                wgsl::Variable::GlobalInputArray(index, Self::compile_item(item))
            }
            gpu::Variable::GlobalScalar(index, elem) => {
                wgsl::Variable::GlobalScalar(index, Self::compile_elem(elem), elem)
            }
            gpu::Variable::Local(index, item, scope_depth) => wgsl::Variable::Local {
                index,
                item: Self::compile_item(item),
                scope_depth,
            },
            gpu::Variable::LocalScalar(index, elem, scope_depth) => wgsl::Variable::LocalScalar {
                index,
                elem: Self::compile_elem(elem),
                scope_depth,
            },
            gpu::Variable::GlobalOutputArray(index, item) => {
                wgsl::Variable::GlobalOutputArray(index, Self::compile_item(item))
            }
            gpu::Variable::ConstantScalar(index, elem) => {
                wgsl::Variable::ConstantScalar(index, Self::compile_elem(elem))
            }
            gpu::Variable::SharedMemory(index, item, size) => {
                let item = Self::compile_item(item);
                if !self.shared_memories.iter().any(|s| s.index == index) {
                    self.shared_memories
                        .push(SharedMemory::new(index, item, size));
                }
                wgsl::Variable::SharedMemory(index, item, size)
            }
            gpu::Variable::LocalArray(index, item, scope_depth, size) => {
                let item = Self::compile_item(item);
                if !self.local_arrays.iter().any(|s| s.index == index) {
                    self.local_arrays
                        .push(LocalArray::new(index, item, scope_depth, size));
                }
                wgsl::Variable::LocalArray(index, item, scope_depth, size)
            }
            gpu::Variable::Id => {
                self.id = true;
                wgsl::Variable::Id
            }
            gpu::Variable::Rank => {
                self.rank = true;
                wgsl::Variable::Rank
            }
            gpu::Variable::LocalInvocationIndex => {
                self.local_invocation_index = true;
                wgsl::Variable::LocalInvocationIndex
            }
            gpu::Variable::LocalInvocationIdX => {
                self.local_invocation_id = true;
                wgsl::Variable::LocalInvocationIdX
            }
            gpu::Variable::LocalInvocationIdY => {
                self.local_invocation_id = true;
                wgsl::Variable::LocalInvocationIdY
            }
            gpu::Variable::LocalInvocationIdZ => {
                self.local_invocation_id = true;
                wgsl::Variable::LocalInvocationIdZ
            }
            gpu::Variable::WorkgroupIdX => {
                self.workgroup_id = true;
                wgsl::Variable::WorkgroupIdX
            }
            gpu::Variable::WorkgroupIdY => {
                self.workgroup_id = true;
                wgsl::Variable::WorkgroupIdY
            }
            gpu::Variable::WorkgroupIdZ => {
                self.workgroup_id = true;
                wgsl::Variable::WorkgroupIdZ
            }
            gpu::Variable::GlobalInvocationIdX => {
                self.global_invocation_id = true;
                wgsl::Variable::GlobalInvocationIdX
            }
            gpu::Variable::GlobalInvocationIdY => {
                self.global_invocation_id = true;
                wgsl::Variable::GlobalInvocationIdY
            }
            gpu::Variable::GlobalInvocationIdZ => {
                self.global_invocation_id = true;
                wgsl::Variable::GlobalInvocationIdZ
            }
            gpu::Variable::WorkgroupSizeX => wgsl::Variable::WorkgroupSizeX,
            gpu::Variable::WorkgroupSizeY => wgsl::Variable::WorkgroupSizeY,
            gpu::Variable::WorkgroupSizeZ => wgsl::Variable::WorkgroupSizeZ,
            gpu::Variable::NumWorkgroupsX => {
                self.num_workgroups = true;
                wgsl::Variable::NumWorkgroupsX
            }
            gpu::Variable::NumWorkgroupsY => {
                self.num_workgroups = true;
                wgsl::Variable::NumWorkgroupsY
            }
            gpu::Variable::NumWorkgroupsZ => {
                self.num_workgroups = true;
                wgsl::Variable::NumWorkgroupsZ
            }
        }
    }

    fn compile_scope(&mut self, value: &mut gpu::Scope) -> Vec<wgsl::Instruction> {
        let mut instructions = Vec::new();
        let processing = value.process();

        for var in processing.variables {
            instructions.push(wgsl::Instruction::DeclareVariable {
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
        instructions: &mut Vec<wgsl::Instruction>,
        operation: gpu::Operation,
        scope: &mut gpu::Scope,
    ) {
        match operation {
            gpu::Operation::Operator(op) => instructions.push(self.compile_instruction(op)),
            gpu::Operation::Procedure(proc) => self.compile_procedure(instructions, proc, scope),
            gpu::Operation::Metadata(op) => instructions.push(self.compile_metadata(op)),
            gpu::Operation::Branch(val) => self.compile_branch(instructions, val),
            gpu::Operation::Synchronization(val) => self.compile_synchronization(instructions, val),
        }
    }

    fn compile_branch(&mut self, instructions: &mut Vec<wgsl::Instruction>, branch: gpu::Branch) {
        match branch {
            gpu::Branch::If(mut op) => instructions.push(wgsl::Instruction::If {
                cond: self.compile_variable(op.cond),
                instructions: self.compile_scope(&mut op.scope),
            }),
            gpu::Branch::IfElse(mut op) => instructions.push(wgsl::Instruction::IfElse {
                cond: self.compile_variable(op.cond),
                instructions_if: self.compile_scope(&mut op.scope_if),
                instructions_else: self.compile_scope(&mut op.scope_else),
            }),
            gpu::Branch::Return => instructions.push(wgsl::Instruction::Return),
            gpu::Branch::Break => instructions.push(wgsl::Instruction::Break),
            gpu::Branch::RangeLoop(mut range_loop) => {
                instructions.push(wgsl::Instruction::RangeLoop {
                    i: self.compile_variable(range_loop.i),
                    start: self.compile_variable(range_loop.start),
                    end: self.compile_variable(range_loop.end),
                    instructions: self.compile_scope(&mut range_loop.scope),
                })
            }
            gpu::Branch::Loop(mut op) => instructions.push(wgsl::Instruction::Loop {
                instructions: self.compile_scope(&mut op.scope),
            }),
        };
    }

    fn compile_synchronization(
        &mut self,
        instructions: &mut Vec<wgsl::Instruction>,
        synchronization: gpu::Synchronization,
    ) {
        match synchronization {
            gpu::Synchronization::WorkgroupBarrier => {
                instructions.push(wgsl::Instruction::WorkgroupBarrier)
            }
        };
    }

    fn compile_procedure(
        &mut self,
        instructions: &mut Vec<wgsl::Instruction>,
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

    fn compile_metadata(&mut self, metadata: gpu::Metadata) -> wgsl::Instruction {
        match metadata {
            gpu::Metadata::Stride { dim, var, out } => {
                self.stride = true;
                let position = match var {
                    gpu::Variable::GlobalInputArray(idx, _) => idx as usize,
                    gpu::Variable::GlobalOutputArray(idx, _) => self.num_inputs + idx as usize,
                    _ => panic!("Only Input and Output have a stride, got: {:?}", var),
                };
                wgsl::Instruction::Stride {
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
                wgsl::Instruction::Shape {
                    dim: self.compile_variable(dim),
                    position,
                    out: self.compile_variable(out),
                }
            }
            gpu::Metadata::ArrayLength { var, out } => wgsl::Instruction::ArrayLength {
                out: self.compile_variable(out),
                var: self.compile_variable(var),
            },
        }
    }

    fn compile_instruction(&mut self, value: gpu::Operator) -> wgsl::Instruction {
        match value {
            gpu::Operator::Max(op) => wgsl::Instruction::Max {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(op.out),
            },
            gpu::Operator::Min(op) => wgsl::Instruction::Min {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(op.out),
            },
            gpu::Operator::Add(op) => wgsl::Instruction::Add {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(op.out),
            },
            gpu::Operator::Index(op) => wgsl::Instruction::Index {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(op.out),
            },
            gpu::Operator::Modulo(op) => wgsl::Instruction::Modulo {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(op.out),
            },
            gpu::Operator::Sub(op) => wgsl::Instruction::Sub {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(op.out),
            },
            gpu::Operator::Mul(op) => wgsl::Instruction::Mul {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(op.out),
            },
            gpu::Operator::Div(op) => wgsl::Instruction::Div {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(op.out),
            },
            gpu::Operator::Abs(op) => wgsl::Instruction::Abs {
                input: self.compile_variable(op.input),
                out: self.compile_variable(op.out),
            },
            gpu::Operator::Exp(op) => wgsl::Instruction::Exp {
                input: self.compile_variable(op.input),
                out: self.compile_variable(op.out),
            },
            gpu::Operator::Ceil(op) => wgsl::Instruction::Ceil {
                input: self.compile_variable(op.input),
                out: self.compile_variable(op.out),
            },
            gpu::Operator::Log(op) => wgsl::Instruction::Log {
                input: self.compile_variable(op.input),
                out: self.compile_variable(op.out),
            },
            gpu::Operator::Log1p(op) => wgsl::Instruction::Log1p {
                input: self.compile_variable(op.input),
                out: self.compile_variable(op.out),
            },
            gpu::Operator::Cos(op) => wgsl::Instruction::Cos {
                input: self.compile_variable(op.input),
                out: self.compile_variable(op.out),
            },
            gpu::Operator::Sin(op) => wgsl::Instruction::Sin {
                input: self.compile_variable(op.input),
                out: self.compile_variable(op.out),
            },
            gpu::Operator::Tanh(op) => wgsl::Instruction::Tanh {
                input: self.compile_variable(op.input),
                out: self.compile_variable(op.out),
            },
            gpu::Operator::Powf(op) => wgsl::Instruction::Powf {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(op.out),
            },
            gpu::Operator::Sqrt(op) => wgsl::Instruction::Sqrt {
                input: self.compile_variable(op.input),
                out: self.compile_variable(op.out),
            },
            gpu::Operator::Erf(op) => wgsl::Instruction::Erf {
                input: self.compile_variable(op.input),
                out: self.compile_variable(op.out),
            },
            gpu::Operator::Recip(op) => wgsl::Instruction::Recip {
                input: self.compile_variable(op.input),
                out: self.compile_variable(op.out),
            },
            gpu::Operator::Equal(op) => wgsl::Instruction::Equal {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(op.out),
            },
            gpu::Operator::Lower(op) => wgsl::Instruction::Lower {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(op.out),
            },
            gpu::Operator::Clamp(op) => wgsl::Instruction::Clamp {
                input: self.compile_variable(op.input),
                min_value: self.compile_variable(op.min_value),
                max_value: self.compile_variable(op.max_value),
                out: self.compile_variable(op.out),
            },
            gpu::Operator::Greater(op) => wgsl::Instruction::Greater {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(op.out),
            },
            gpu::Operator::LowerEqual(op) => wgsl::Instruction::LowerEqual {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(op.out),
            },
            gpu::Operator::GreaterEqual(op) => wgsl::Instruction::GreaterEqual {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(op.out),
            },
            gpu::Operator::NotEqual(op) => wgsl::Instruction::NotEqual {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(op.out),
            },
            gpu::Operator::Assign(op) => wgsl::Instruction::Assign {
                input: self.compile_variable(op.input),
                out: self.compile_variable(op.out),
            },
            gpu::Operator::IndexAssign(op) => wgsl::Instruction::IndexAssign {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(op.out),
            },
            gpu::Operator::And(op) => wgsl::Instruction::And {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(op.out),
            },
            gpu::Operator::Or(op) => wgsl::Instruction::Or {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(op.out),
            },
            gpu::Operator::Not(op) => wgsl::Instruction::Not {
                input: self.compile_variable(op.input),
                out: self.compile_variable(op.out),
            },
            gpu::Operator::BitwiseAnd(op) => wgsl::Instruction::BitwiseAnd {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(op.out),
            },
            gpu::Operator::BitwiseXor(op) => wgsl::Instruction::BitwiseXor {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(op.out),
            },
            gpu::Operator::ShiftLeft(op) => wgsl::Instruction::ShiftLeft {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(op.out),
            },
            gpu::Operator::ShiftRight(op) => wgsl::Instruction::ShiftRight {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(op.out),
            },
        }
    }

    fn compile_location(value: gpu::Location) -> wgsl::Location {
        match value {
            gpu::Location::Storage => wgsl::Location::Storage,
            gpu::Location::Workgroup => wgsl::Location::Workgroup,
        }
    }

    fn compile_visibility(value: gpu::Visibility) -> wgsl::Visibility {
        match value {
            gpu::Visibility::Read => wgsl::Visibility::Read,
            gpu::Visibility::ReadWrite => wgsl::Visibility::ReadWrite,
        }
    }

    fn compile_binding(value: gpu::Binding) -> wgsl::Binding {
        wgsl::Binding {
            visibility: Self::compile_visibility(value.visibility),
            location: Self::compile_location(value.location),
            item: Self::compile_item(value.item),
            size: value.size,
        }
    }
}

fn register_extensions(instructions: &[wgsl::Instruction]) -> Vec<wgsl::Extension> {
    let mut extensions = Vec::new();

    let mut register_extension = |extension: wgsl::Extension| {
        if !extensions.contains(&extension) {
            extensions.push(extension);
        }
    };

    // Since not all instructions are native to WGSL, we need to add the custom ones.
    for instruction in instructions {
        match instruction {
            wgsl::Instruction::Powf { lhs: _, rhs, out } => {
                register_extension(wgsl::Extension::PowfPrimitive(out.item()));

                if rhs.is_always_scalar() {
                    register_extension(wgsl::Extension::PowfScalar(out.item()));
                } else {
                    register_extension(wgsl::Extension::Powf(out.item()));
                }
            }
            wgsl::Instruction::Erf { input, out: _ } => {
                register_extension(wgsl::Extension::Erf(input.item()));
            }
            #[cfg(target_os = "macos")]
            wgsl::Instruction::Tanh { input, out: _ } => {
                register_extension(wgsl::Extension::SafeTanh(input.item()))
            }
            _ => {}
        }
    }

    extensions
}
