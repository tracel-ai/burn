use super::{shader::ComputeShader, Item};
use crate::{
    codegen::{
        compiler,
        dialect::{gpu, wgsl},
    },
    FloatElement, IntElement,
};
use std::marker::PhantomData;

/// Wgsl Compiler.
#[derive(Clone)]
pub struct Compiler<F: FloatElement, I: IntElement> {
    num_inputs: usize,
    num_outputs: usize,
    invocation_index: bool,
    workgroup_id: bool,
    _float: PhantomData<F>,
    _int: PhantomData<I>,
}

impl<F: FloatElement, I: IntElement> core::fmt::Debug for Compiler<F, I> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("WgslCompiler")
    }
}

impl<F: FloatElement, I: IntElement> Default for Compiler<F, I> {
    fn default() -> Self {
        Self {
            num_inputs: 0,
            num_outputs: 0,
            invocation_index: false,
            workgroup_id: false,
            _float: PhantomData,
            _int: PhantomData,
        }
    }
}

impl<F: FloatElement, I: IntElement> compiler::Compiler for Compiler<F, I> {
    type Representation = ComputeShader;
    type Float = F;
    type Int = I;
    type FullPrecisionCompiler = Compiler<f32, i32>;

    fn compile(shader: gpu::ComputeShader) -> Self::Representation {
        let mut compiler = Self::default();
        compiler.compile_shader(shader)
    }

    fn elem_size(elem: gpu::Elem) -> usize {
        Self::compile_elem(elem).size()
    }
}

impl<F: FloatElement, I: IntElement> Compiler<F, I> {
    fn compile_shader(&mut self, mut value: gpu::ComputeShader) -> wgsl::ComputeShader {
        self.num_inputs = value.inputs.len();
        self.num_outputs = value.outputs.len();

        let body = self.compile_scope(&mut value.body);
        let extensions = register_extensions(&body);

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
            workgroup_size: value.workgroup_size,
            global_invocation_id: true,
            local_invocation_index: self.invocation_index,
            num_workgroups: true,
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
            gpu::Elem::Float => F::wgsl_elem(),
            gpu::Elem::Int => I::wgsl_elem(),
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
            gpu::Variable::Id => wgsl::Variable::Id,
            gpu::Variable::Rank => wgsl::Variable::Rank,
            gpu::Variable::InvocationIndex => {
                self.invocation_index = true;
                wgsl::Variable::LocalInvocationIndex
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
            gpu::Variable::GlobalInvocationIdX => wgsl::Variable::GlobalInvocationIdX,
            gpu::Variable::GlobalInvocationIdY => wgsl::Variable::GlobalInvocationIdY,
            gpu::Variable::GlobalInvocationIdZ => wgsl::Variable::GlobalInvocationIdZ,
        }
    }

    fn compile_scope(&mut self, value: &mut gpu::Scope) -> wgsl::Scope {
        let mut operations = Vec::new();
        let processing = value.process();

        for var in processing.variables {
            operations.push(wgsl::Instruction::DeclareVariable {
                var: self.compile_variable(var),
            });
        }

        processing
            .operations
            .into_iter()
            .for_each(|op| self.compile_operation(&mut operations, op, value));

        wgsl::Scope {
            operators: operations,
        }
    }

    fn compile_operation(
        &mut self,
        instructions: &mut Vec<wgsl::Instruction>,
        operation: gpu::Operation,
        scope: &mut gpu::Scope,
    ) {
        match operation {
            gpu::Operation::Operator(op) => instructions.push(self.compile_instruction(op)),
            gpu::Operation::Procedure(algo) => self.compile_algorithm(instructions, algo, scope),
            gpu::Operation::Metadata(op) => instructions.push(self.compile_metadata(op)),
            gpu::Operation::Loop(val) => instructions.push(self.compile_loop(val)),
            gpu::Operation::Branch(val) => self.compile_branch(instructions, val),
        }
    }

    fn compile_branch(&mut self, instructions: &mut Vec<wgsl::Instruction>, branch: gpu::Branch) {
        match branch {
            gpu::Branch::If(mut op) => instructions.push(wgsl::Instruction::If {
                cond: self.compile_variable(op.cond),
                instructions: self.compile_scope(&mut op.scope).operators,
            }),
            gpu::Branch::IfElse(mut op) => instructions.push(wgsl::Instruction::IfElse {
                cond: self.compile_variable(op.cond),
                instructions_if: self.compile_scope(&mut op.scope_if).operators,
                instructions_else: self.compile_scope(&mut op.scope_else).operators,
            }),
            gpu::Branch::Return => instructions.push(wgsl::Instruction::Return),
            gpu::Branch::Break => instructions.push(wgsl::Instruction::Break),
        };
    }

    fn compile_algorithm(
        &mut self,
        instructions: &mut Vec<wgsl::Instruction>,
        proc: gpu::Procedure,
        scope: &mut gpu::Scope,
    ) {
        let mut compile = |scope: &mut gpu::Scope| {
            let compiled = self.compile_scope(scope).operators;
            instructions.extend(compiled);
        };

        match proc {
            gpu::Procedure::ReadGlobalWithLayout(algo) => {
                algo.expand(scope);
                compile(scope);
            }
            gpu::Procedure::ReadGlobal(algo) => {
                algo.expand(scope);
                compile(scope);
            }
            gpu::Procedure::Matmul(algo) => {
                algo.expand(scope);
                compile(scope);
            }
            gpu::Procedure::WriteGlobal(algo) => {
                algo.expand(scope);
                compile(scope);
            }
            gpu::Procedure::ConditionalAssign(proc) => {
                proc.expand(scope);
                compile(scope);
            }
        }
    }

    fn compile_loop(&mut self, loop_val: gpu::Loop) -> wgsl::Instruction {
        match loop_val {
            gpu::Loop::Range(mut range_loop) => wgsl::Instruction::RangeLoop {
                i: self.compile_variable(range_loop.i),
                start: self.compile_variable(range_loop.start),
                end: self.compile_variable(range_loop.end),
                instructions: self.compile_scope(&mut range_loop.scope).operators,
            },
        }
    }

    fn compile_metadata(&mut self, metadata: gpu::Metadata) -> wgsl::Instruction {
        match metadata {
            gpu::Metadata::Stride { dim, var, out } => {
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
        }
    }

    fn compile_instruction(&mut self, value: gpu::Operator) -> wgsl::Instruction {
        match value {
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
            gpu::Operator::Assign(op) => wgsl::Instruction::Assign {
                input: self.compile_variable(op.input),
                out: self.compile_variable(op.out),
            },
            gpu::Operator::IndexAssign(op) => wgsl::Instruction::IndexAssign {
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

fn register_extensions(body: &wgsl::Scope) -> Vec<wgsl::Extension> {
    let mut extensions = Vec::new();

    let mut register_extension = |extension: wgsl::Extension| {
        if !extensions.contains(&extension) {
            extensions.push(extension);
        }
    };

    // Since not all operators are native to WGSL, we need to add the custom ones.
    for op in body.operators.iter() {
        match op {
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
