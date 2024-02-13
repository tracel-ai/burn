use super::{shader::ComputeShader, Item};
use crate::{
    codegen::{
        compiler,
        dialect::{gpu, wgsl},
    },
    FloatElement, IntElement,
};
use std::{marker::PhantomData, rc::Rc};

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
    fn compile_shader(&mut self, value: gpu::ComputeShader) -> wgsl::ComputeShader {
        self.num_inputs = value.inputs.len();
        self.num_outputs = value.outputs.len();

        let body = self.compile_scope(value.body, None);
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

    fn compile_variable(value: gpu::Variable, prefix: &Rc<String>) -> wgsl::Variable {
        match value {
            gpu::Variable::Input(index, item) => {
                wgsl::Variable::Input(index, Self::compile_item(item))
            }
            gpu::Variable::Scalar(index, item) => {
                let elem = item.elem();
                wgsl::Variable::Scalar(index, Self::compile_item(item), elem)
            }
            gpu::Variable::Local(index, item) => wgsl::Variable::Local {
                prefix: prefix.clone(),
                index,
                item: Self::compile_item(item),
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

    fn compile_scope(&self, value: gpu::Scope, prefix: Option<&Rc<String>>) -> wgsl::Scope {
        let prefix = match prefix {
            Some(val) => Rc::new(val.to_string() + value.prefix.as_str()),
            None => Rc::new(value.prefix),
        };

        let mut operations = Vec::new();
        value
            .operations
            .into_iter()
            .for_each(|op| self.compile_operation(&mut operations, op, &prefix));

        wgsl::Scope {
            operators: operations,
        }
    }

    fn compile_operation(
        &self,
        instructions: &mut Vec<wgsl::Instruction>,
        operation: gpu::Operation,
        prefix: &Rc<String>,
    ) {
        match operation {
            gpu::Operation::Operator(op) => instructions.push(self.compile_instruction(op, prefix)),
            gpu::Operation::Algorithm(_) => {}
            gpu::Operation::Metadata(op) => instructions.push(self.compile_metadata(op, prefix)),
            gpu::Operation::Loop(val) => instructions.push(self.compile_loop(val, prefix)),
        }
    }

    fn compile_loop(&self, loop_val: gpu::Loop, prefix: &Rc<String>) -> wgsl::Instruction {
        match loop_val {
            gpu::Loop::Range(range_loop) => {
                // Start and end are from the parent scope.
                let start = Self::compile_variable(range_loop.start, prefix);
                let end = Self::compile_variable(range_loop.end, prefix);

                let prefix_loop = prefix.to_string() + range_loop.scope.prefix.as_str();
                let prefix_loop = Rc::new(prefix_loop);
                // i is from the loop scope.
                let i = Self::compile_variable(range_loop.i, &prefix_loop);

                let mut instructions = Vec::new();
                range_loop
                    .scope
                    .operations
                    .into_iter()
                    .for_each(|op| self.compile_operation(&mut instructions, op, &prefix_loop));

                wgsl::Instruction::RangeLoop {
                    i,
                    start,
                    end,
                    instructions,
                }
            }
        }
    }

    fn compile_metadata(&self, metadata: gpu::Metadata, prefix: &Rc<String>) -> wgsl::Instruction {
        match metadata {
            gpu::Metadata::Rank { out } => wgsl::Instruction::Rank {
                out: Self::compile_variable(out, prefix),
            },
            gpu::Metadata::Stride { dim, var, out } => {
                let position = match var {
                    gpu::Variable::Input(idx, _) => idx as usize,
                    gpu::Variable::Output(idx, _) => self.num_inputs + idx as usize,
                    _ => panic!("Only Input and Output have a stride."),
                };
                wgsl::Instruction::Stride {
                    dim: Self::compile_variable(dim, prefix),
                    position,
                    out: Self::compile_variable(out, prefix),
                }
            }
            gpu::Metadata::Shape { dim, var, out } => {
                let position = match var {
                    gpu::Variable::Input(idx, _) => idx as usize,
                    gpu::Variable::Output(idx, _) => self.num_inputs + idx as usize,
                    _ => panic!("Only Input and Output have a shape."),
                };
                wgsl::Instruction::Shape {
                    dim: Self::compile_variable(dim, prefix),
                    position,
                    out: Self::compile_variable(out, prefix),
                }
            }
        }
    }

    fn compile_instruction(&self, value: gpu::Operator, prefix: &Rc<String>) -> wgsl::Instruction {
        match value {
            gpu::Operator::Add(op) => wgsl::Instruction::Add {
                lhs: Self::compile_variable(op.lhs, &prefix),
                rhs: Self::compile_variable(op.rhs, &prefix),
                out: Self::compile_variable(op.out, &prefix),
            },
            gpu::Operator::Modulo(op) => wgsl::Instruction::Modulo {
                lhs: Self::compile_variable(op.lhs, &prefix),
                rhs: Self::compile_variable(op.rhs, &prefix),
                out: Self::compile_variable(op.out, &prefix),
            },
            gpu::Operator::Sub(op) => wgsl::Instruction::Sub {
                lhs: Self::compile_variable(op.lhs, &prefix),
                rhs: Self::compile_variable(op.rhs, &prefix),
                out: Self::compile_variable(op.out, &prefix),
            },
            gpu::Operator::Mul(op) => wgsl::Instruction::Mul {
                lhs: Self::compile_variable(op.lhs, &prefix),
                rhs: Self::compile_variable(op.rhs, &prefix),
                out: Self::compile_variable(op.out, &prefix),
            },
            gpu::Operator::Div(op) => wgsl::Instruction::Div {
                lhs: Self::compile_variable(op.lhs, &prefix),
                rhs: Self::compile_variable(op.rhs, &prefix),
                out: Self::compile_variable(op.out, &prefix),
            },
            gpu::Operator::Abs(op) => wgsl::Instruction::Abs {
                input: Self::compile_variable(op.input, &prefix),
                out: Self::compile_variable(op.out, &prefix),
            },
            gpu::Operator::Exp(op) => wgsl::Instruction::Exp {
                input: Self::compile_variable(op.input, &prefix),
                out: Self::compile_variable(op.out, &prefix),
            },
            gpu::Operator::Log(op) => wgsl::Instruction::Log {
                input: Self::compile_variable(op.input, &prefix),
                out: Self::compile_variable(op.out, &prefix),
            },
            gpu::Operator::Log1p(op) => wgsl::Instruction::Log1p {
                input: Self::compile_variable(op.input, &prefix),
                out: Self::compile_variable(op.out, &prefix),
            },
            gpu::Operator::Cos(op) => wgsl::Instruction::Cos {
                input: Self::compile_variable(op.input, &prefix),
                out: Self::compile_variable(op.out, &prefix),
            },
            gpu::Operator::Sin(op) => wgsl::Instruction::Sin {
                input: Self::compile_variable(op.input, &prefix),
                out: Self::compile_variable(op.out, &prefix),
            },
            gpu::Operator::Tanh(op) => wgsl::Instruction::Tanh {
                input: Self::compile_variable(op.input, &prefix),
                out: Self::compile_variable(op.out, &prefix),
            },
            gpu::Operator::Powf(op) => wgsl::Instruction::Powf {
                lhs: Self::compile_variable(op.lhs, &prefix),
                rhs: Self::compile_variable(op.rhs, &prefix),
                out: Self::compile_variable(op.out, &prefix),
            },
            gpu::Operator::Sqrt(op) => wgsl::Instruction::Sqrt {
                input: Self::compile_variable(op.input, &prefix),
                out: Self::compile_variable(op.out, &prefix),
            },
            gpu::Operator::Erf(op) => wgsl::Instruction::Erf {
                input: Self::compile_variable(op.input, &prefix),
                out: Self::compile_variable(op.out, &prefix),
            },
            gpu::Operator::Recip(op) => wgsl::Instruction::Recip {
                input: Self::compile_variable(op.input, &prefix),
                out: Self::compile_variable(op.out, &prefix),
            },
            gpu::Operator::Equal(op) => wgsl::Instruction::Equal {
                lhs: Self::compile_variable(op.lhs, &prefix),
                rhs: Self::compile_variable(op.rhs, &prefix),
                out: Self::compile_variable(op.out, &prefix),
            },
            gpu::Operator::Lower(op) => wgsl::Instruction::Lower {
                lhs: Self::compile_variable(op.lhs, &prefix),
                rhs: Self::compile_variable(op.rhs, &prefix),
                out: Self::compile_variable(op.out, &prefix),
            },
            gpu::Operator::Clamp(op) => wgsl::Instruction::Clamp {
                input: Self::compile_variable(op.input, &prefix),
                min_value: Self::compile_variable(op.min_value, &prefix),
                max_value: Self::compile_variable(op.max_value, &prefix),
                out: Self::compile_variable(op.out, &prefix),
            },
            gpu::Operator::Greater(op) => wgsl::Instruction::Greater {
                lhs: Self::compile_variable(op.lhs, &prefix),
                rhs: Self::compile_variable(op.rhs, &prefix),
                out: Self::compile_variable(op.out, &prefix),
            },
            gpu::Operator::LowerEqual(op) => wgsl::Instruction::LowerEqual {
                lhs: Self::compile_variable(op.lhs, &prefix),
                rhs: Self::compile_variable(op.rhs, &prefix),
                out: Self::compile_variable(op.out, &prefix),
            },
            gpu::Operator::GreaterEqual(op) => wgsl::Instruction::GreaterEqual {
                lhs: Self::compile_variable(op.lhs, &prefix),
                rhs: Self::compile_variable(op.rhs, &prefix),
                out: Self::compile_variable(op.out, &prefix),
            },
            gpu::Operator::ConditionalAssign(op) => wgsl::Instruction::ConditionalAssign {
                cond: Self::compile_variable(op.cond, &prefix),
                lhs: Self::compile_variable(op.lhs, &prefix),
                rhs: Self::compile_variable(op.rhs, &prefix),
                out: Self::compile_variable(op.out, &prefix),
            },
            gpu::Operator::AssignGlobal(op) => wgsl::Instruction::AssignGlobal {
                input: Self::compile_variable(op.input, &prefix),
                out: Self::compile_variable(op.out, &prefix),
            },
            gpu::Operator::AssignLocal(op) => wgsl::Instruction::AssignLocal {
                input: Self::compile_variable(op.input, &prefix),
                out: Self::compile_variable(op.out, &prefix),
            },
            gpu::Operator::ReadGlobal(op) => wgsl::Instruction::ReadGlobal {
                variable: Self::compile_variable(op.variable, &prefix),
            },
            gpu::Operator::ReadGlobalWithLayout(op) => wgsl::Instruction::ReadGlobalWithLayout {
                variable: Self::compile_variable(op.variable, &prefix),
                tensor_read_pos: op.tensor_read_pos,
                tensor_layout_pos: op.tensor_layout_pos,
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
