use super::{shader::ComputeShader, Item};
use crate::{
    codegen::{
        compiler,
        dialect::{
            gpu::{
                self,
                algorithm::{generate_read_global, generate_read_global_with_layout},
            },
            wgsl,
        },
    },
    FloatElement, IntElement,
};
use std::marker::PhantomData;

/// Wgsl Compiler.
#[derive(Clone)]
pub struct Compiler<F: FloatElement, I: IntElement> {
    num_inputs: usize,
    num_outputs: usize,
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
            global_invocation_id: value.global_invocation_id,
            num_workgroups: value.num_workgroups,
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

    fn compile_variable(value: gpu::Variable) -> wgsl::Variable {
        match value {
            gpu::Variable::Input(index, item) => {
                wgsl::Variable::Input(index, Self::compile_item(item))
            }
            gpu::Variable::Scalar(index, item) => {
                let elem = item.elem();
                wgsl::Variable::Scalar(index, Self::compile_item(item), elem)
            }
            gpu::Variable::Local(index, item, scope_depth) => wgsl::Variable::Local {
                index,
                item: Self::compile_item(item),
                scope_depth,
            },
            gpu::Variable::Output(index, item) => {
                wgsl::Variable::Output(index, Self::compile_item(item))
            }
            gpu::Variable::Constant(index, item) => {
                wgsl::Variable::Constant(index, Self::compile_item(item))
            }
            gpu::Variable::Id => wgsl::Variable::Id,
            gpu::Variable::Rank => wgsl::Variable::Rank,
        }
    }

    fn compile_scope(&self, value: &mut gpu::Scope) -> wgsl::Scope {
        let mut operations = Vec::new();
        let processing = value.process();

        for var in processing.variables {
            operations.push(wgsl::Instruction::DeclareVariable {
                var: Self::compile_variable(var),
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
        &self,
        instructions: &mut Vec<wgsl::Instruction>,
        operation: gpu::Operation,
        scope: &mut gpu::Scope,
    ) {
        match operation {
            gpu::Operation::Operator(op) => instructions.push(self.compile_instruction(op)),
            gpu::Operation::Algorithm(algo) => self.compile_algorithm(instructions, algo, scope),
            gpu::Operation::Metadata(op) => instructions.push(self.compile_metadata(op)),
            gpu::Operation::Loop(val) => instructions.push(self.compile_loop(val)),
        }
    }

    fn compile_algorithm(
        &self,
        instructions: &mut Vec<wgsl::Instruction>,
        algo: gpu::Algorithm,
        scope: &mut gpu::Scope,
    ) {
        let mut compile = |scope: &mut gpu::Scope| {
            let compiled = self.compile_scope(scope).operators;
            instructions.extend(compiled);
        };

        match algo {
            gpu::Algorithm::ReadGlobalWithLayout(algo) => {
                generate_read_global_with_layout(scope, algo);
                compile(scope);
            }
            gpu::Algorithm::ReadGlobal(algo) => {
                generate_read_global(scope, algo);
                compile(scope);
            }
        }
    }

    fn compile_loop(&self, loop_val: gpu::Loop) -> wgsl::Instruction {
        match loop_val {
            gpu::Loop::Range(mut range_loop) => {
                // Start and end are from the parent scope.
                let start = Self::compile_variable(range_loop.start);
                let end = Self::compile_variable(range_loop.end);

                // i is from the loop scope.
                let i = Self::compile_variable(range_loop.i);

                let instructions = self.compile_scope(&mut range_loop.scope).operators;

                wgsl::Instruction::RangeLoop {
                    i,
                    start,
                    end,
                    instructions,
                }
            }
        }
    }

    fn compile_metadata(&self, metadata: gpu::Metadata) -> wgsl::Instruction {
        match metadata {
            gpu::Metadata::Stride { dim, var, out } => {
                let position = match var {
                    gpu::Variable::Input(idx, _) => idx as usize,
                    gpu::Variable::Output(idx, _) => self.num_inputs + idx as usize,
                    _ => panic!("Only Input and Output have a stride, got: {:?}", var),
                };
                wgsl::Instruction::Stride {
                    dim: Self::compile_variable(dim),
                    position,
                    out: Self::compile_variable(out),
                }
            }
            gpu::Metadata::Shape { dim, var, out } => {
                let position = match var {
                    gpu::Variable::Input(idx, _) => idx as usize,
                    gpu::Variable::Output(idx, _) => self.num_inputs + idx as usize,
                    _ => panic!("Only Input and Output have a shape."),
                };
                wgsl::Instruction::Shape {
                    dim: Self::compile_variable(dim),
                    position,
                    out: Self::compile_variable(out),
                }
            }
        }
    }

    fn compile_instruction(&self, value: gpu::Operator) -> wgsl::Instruction {
        match value {
            gpu::Operator::Add(op) => wgsl::Instruction::Add {
                lhs: Self::compile_variable(op.lhs),
                rhs: Self::compile_variable(op.rhs),
                out: Self::compile_variable(op.out),
            },
            gpu::Operator::Index(op) => wgsl::Instruction::Index {
                lhs: Self::compile_variable(op.lhs),
                rhs: Self::compile_variable(op.rhs),
                out: Self::compile_variable(op.out),
            },
            gpu::Operator::Modulo(op) => wgsl::Instruction::Modulo {
                lhs: Self::compile_variable(op.lhs),
                rhs: Self::compile_variable(op.rhs),
                out: Self::compile_variable(op.out),
            },
            gpu::Operator::Sub(op) => wgsl::Instruction::Sub {
                lhs: Self::compile_variable(op.lhs),
                rhs: Self::compile_variable(op.rhs),
                out: Self::compile_variable(op.out),
            },
            gpu::Operator::Mul(op) => wgsl::Instruction::Mul {
                lhs: Self::compile_variable(op.lhs),
                rhs: Self::compile_variable(op.rhs),
                out: Self::compile_variable(op.out),
            },
            gpu::Operator::Div(op) => wgsl::Instruction::Div {
                lhs: Self::compile_variable(op.lhs),
                rhs: Self::compile_variable(op.rhs),
                out: Self::compile_variable(op.out),
            },
            gpu::Operator::Abs(op) => wgsl::Instruction::Abs {
                input: Self::compile_variable(op.input),
                out: Self::compile_variable(op.out),
            },
            gpu::Operator::Exp(op) => wgsl::Instruction::Exp {
                input: Self::compile_variable(op.input),
                out: Self::compile_variable(op.out),
            },
            gpu::Operator::Log(op) => wgsl::Instruction::Log {
                input: Self::compile_variable(op.input),
                out: Self::compile_variable(op.out),
            },
            gpu::Operator::Log1p(op) => wgsl::Instruction::Log1p {
                input: Self::compile_variable(op.input),
                out: Self::compile_variable(op.out),
            },
            gpu::Operator::Cos(op) => wgsl::Instruction::Cos {
                input: Self::compile_variable(op.input),
                out: Self::compile_variable(op.out),
            },
            gpu::Operator::Sin(op) => wgsl::Instruction::Sin {
                input: Self::compile_variable(op.input),
                out: Self::compile_variable(op.out),
            },
            gpu::Operator::Tanh(op) => wgsl::Instruction::Tanh {
                input: Self::compile_variable(op.input),
                out: Self::compile_variable(op.out),
            },
            gpu::Operator::Powf(op) => wgsl::Instruction::Powf {
                lhs: Self::compile_variable(op.lhs),
                rhs: Self::compile_variable(op.rhs),
                out: Self::compile_variable(op.out),
            },
            gpu::Operator::Sqrt(op) => wgsl::Instruction::Sqrt {
                input: Self::compile_variable(op.input),
                out: Self::compile_variable(op.out),
            },
            gpu::Operator::Erf(op) => wgsl::Instruction::Erf {
                input: Self::compile_variable(op.input),
                out: Self::compile_variable(op.out),
            },
            gpu::Operator::Recip(op) => wgsl::Instruction::Recip {
                input: Self::compile_variable(op.input),
                out: Self::compile_variable(op.out),
            },
            gpu::Operator::Equal(op) => wgsl::Instruction::Equal {
                lhs: Self::compile_variable(op.lhs),
                rhs: Self::compile_variable(op.rhs),
                out: Self::compile_variable(op.out),
            },
            gpu::Operator::Lower(op) => wgsl::Instruction::Lower {
                lhs: Self::compile_variable(op.lhs),
                rhs: Self::compile_variable(op.rhs),
                out: Self::compile_variable(op.out),
            },
            gpu::Operator::Clamp(op) => wgsl::Instruction::Clamp {
                input: Self::compile_variable(op.input),
                min_value: Self::compile_variable(op.min_value),
                max_value: Self::compile_variable(op.max_value),
                out: Self::compile_variable(op.out),
            },
            gpu::Operator::Greater(op) => wgsl::Instruction::Greater {
                lhs: Self::compile_variable(op.lhs),
                rhs: Self::compile_variable(op.rhs),
                out: Self::compile_variable(op.out),
            },
            gpu::Operator::LowerEqual(op) => wgsl::Instruction::LowerEqual {
                lhs: Self::compile_variable(op.lhs),
                rhs: Self::compile_variable(op.rhs),
                out: Self::compile_variable(op.out),
            },
            gpu::Operator::GreaterEqual(op) => wgsl::Instruction::GreaterEqual {
                lhs: Self::compile_variable(op.lhs),
                rhs: Self::compile_variable(op.rhs),
                out: Self::compile_variable(op.out),
            },
            gpu::Operator::ConditionalAssign(op) => wgsl::Instruction::ConditionalAssign {
                cond: Self::compile_variable(op.cond),
                lhs: Self::compile_variable(op.lhs),
                rhs: Self::compile_variable(op.rhs),
                out: Self::compile_variable(op.out),
            },
            gpu::Operator::AssignGlobal(op) => wgsl::Instruction::AssignGlobal {
                input: Self::compile_variable(op.input),
                out: Self::compile_variable(op.out),
            },
            gpu::Operator::AssignLocal(op) => wgsl::Instruction::AssignLocal {
                input: Self::compile_variable(op.input),
                out: Self::compile_variable(op.out),
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
            wgsl::Instruction::Powf { lhs: _, rhs, out } => match rhs {
                wgsl::Variable::Scalar(_, _, _) => {
                    register_extension(wgsl::Extension::PowfScalar(*out.item()));
                }
                _ => {
                    register_extension(wgsl::Extension::Powf(*out.item()));
                }
            },
            wgsl::Instruction::Erf { input, out: _ } => {
                register_extension(wgsl::Extension::Erf(*input.item()));
            }
            #[cfg(target_os = "macos")]
            wgsl::Instruction::Tanh { input, out: _ } => {
                register_extension(wgsl::Extension::SafeTanh(*input.item()))
            }
            _ => {}
        }
    }

    extensions
}
