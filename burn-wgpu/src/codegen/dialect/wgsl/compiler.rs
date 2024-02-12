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
        Self::compile_shader(shader)
    }

    fn elem_size(elem: gpu::Elem) -> usize {
        Self::compile_elem(elem).size()
    }
}

impl<F: FloatElement, I: IntElement> Compiler<F, I> {
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
            gpu::Variable::Local(index, item) => {
                wgsl::Variable::Local(index, Self::compile_item(item))
            }
            gpu::Variable::Output(index, item) => {
                wgsl::Variable::Output(index, Self::compile_item(item))
            }
            gpu::Variable::Constant(index, item) => {
                wgsl::Variable::Constant(index, Self::compile_item(item))
            }
        }
    }

    fn compile_body(value: gpu::Body) -> wgsl::Body {
        wgsl::Body {
            operators: value
                .operators
                .into_iter()
                .map(Self::compile_operation)
                .collect(),
        }
    }

    fn compile_operation(value: gpu::Operation) -> wgsl::Operation {
        match value {
            gpu::Operation::Add(op) => wgsl::Operation::Add {
                lhs: Self::compile_variable(op.lhs),
                rhs: Self::compile_variable(op.rhs),
                out: Self::compile_variable(op.out),
            },
            gpu::Operation::Sub(op) => wgsl::Operation::Sub {
                lhs: Self::compile_variable(op.lhs),
                rhs: Self::compile_variable(op.rhs),
                out: Self::compile_variable(op.out),
            },
            gpu::Operation::Mul(op) => wgsl::Operation::Mul {
                lhs: Self::compile_variable(op.lhs),
                rhs: Self::compile_variable(op.rhs),
                out: Self::compile_variable(op.out),
            },
            gpu::Operation::Div(op) => wgsl::Operation::Div {
                lhs: Self::compile_variable(op.lhs),
                rhs: Self::compile_variable(op.rhs),
                out: Self::compile_variable(op.out),
            },
            gpu::Operation::Abs(op) => wgsl::Operation::Abs {
                input: Self::compile_variable(op.input),
                out: Self::compile_variable(op.out),
            },
            gpu::Operation::Exp(op) => wgsl::Operation::Exp {
                input: Self::compile_variable(op.input),
                out: Self::compile_variable(op.out),
            },
            gpu::Operation::Log(op) => wgsl::Operation::Log {
                input: Self::compile_variable(op.input),
                out: Self::compile_variable(op.out),
            },
            gpu::Operation::Log1p(op) => wgsl::Operation::Log1p {
                input: Self::compile_variable(op.input),
                out: Self::compile_variable(op.out),
            },
            gpu::Operation::Cos(op) => wgsl::Operation::Cos {
                input: Self::compile_variable(op.input),
                out: Self::compile_variable(op.out),
            },
            gpu::Operation::Sin(op) => wgsl::Operation::Sin {
                input: Self::compile_variable(op.input),
                out: Self::compile_variable(op.out),
            },
            gpu::Operation::Tanh(op) => wgsl::Operation::Tanh {
                input: Self::compile_variable(op.input),
                out: Self::compile_variable(op.out),
            },
            gpu::Operation::Powf(op) => wgsl::Operation::Powf {
                lhs: Self::compile_variable(op.lhs),
                rhs: Self::compile_variable(op.rhs),
                out: Self::compile_variable(op.out),
            },
            gpu::Operation::Sqrt(op) => wgsl::Operation::Sqrt {
                input: Self::compile_variable(op.input),
                out: Self::compile_variable(op.out),
            },
            gpu::Operation::Erf(op) => wgsl::Operation::Erf {
                input: Self::compile_variable(op.input),
                out: Self::compile_variable(op.out),
            },
            gpu::Operation::Recip(op) => wgsl::Operation::Recip {
                input: Self::compile_variable(op.input),
                out: Self::compile_variable(op.out),
            },
            gpu::Operation::Equal(op) => wgsl::Operation::Equal {
                lhs: Self::compile_variable(op.lhs),
                rhs: Self::compile_variable(op.rhs),
                out: Self::compile_variable(op.out),
            },
            gpu::Operation::Lower(op) => wgsl::Operation::Lower {
                lhs: Self::compile_variable(op.lhs),
                rhs: Self::compile_variable(op.rhs),
                out: Self::compile_variable(op.out),
            },
            gpu::Operation::Clamp(op) => wgsl::Operation::Clamp {
                input: Self::compile_variable(op.input),
                min_value: Self::compile_variable(op.min_value),
                max_value: Self::compile_variable(op.max_value),
                out: Self::compile_variable(op.out),
            },
            gpu::Operation::Greater(op) => wgsl::Operation::Greater {
                lhs: Self::compile_variable(op.lhs),
                rhs: Self::compile_variable(op.rhs),
                out: Self::compile_variable(op.out),
            },
            gpu::Operation::LowerEqual(op) => wgsl::Operation::LowerEqual {
                lhs: Self::compile_variable(op.lhs),
                rhs: Self::compile_variable(op.rhs),
                out: Self::compile_variable(op.out),
            },
            gpu::Operation::GreaterEqual(op) => wgsl::Operation::GreaterEqual {
                lhs: Self::compile_variable(op.lhs),
                rhs: Self::compile_variable(op.rhs),
                out: Self::compile_variable(op.out),
            },
            gpu::Operation::ConditionalAssign(op) => wgsl::Operation::ConditionalAssign {
                cond: Self::compile_variable(op.cond),
                lhs: Self::compile_variable(op.lhs),
                rhs: Self::compile_variable(op.rhs),
                out: Self::compile_variable(op.out),
            },
            gpu::Operation::AssignGlobal(op) => wgsl::Operation::AssignGlobal {
                input: Self::compile_variable(op.input),
                out: Self::compile_variable(op.out),
            },
            gpu::Operation::AssignLocal(op) => wgsl::Operation::AssignLocal {
                input: Self::compile_variable(op.input),
                out: Self::compile_variable(op.out),
            },
            gpu::Operation::ReadGlobal(op) => wgsl::Operation::ReadGlobal {
                variable: Self::compile_variable(op.variable),
            },
            gpu::Operation::ReadGlobalWithLayout(op) => wgsl::Operation::ReadGlobalWithLayout {
                variable: Self::compile_variable(op.variable),
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

    fn compile_shader(value: gpu::ComputeShader) -> wgsl::ComputeShader {
        let body = Self::compile_body(value.body);
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
}

fn register_extensions(body: &wgsl::Body) -> Vec<wgsl::Extension> {
    let mut extensions = Vec::new();

    let mut register_extension = |extension: wgsl::Extension| {
        if !extensions.contains(&extension) {
            extensions.push(extension);
        }
    };

    // Since not all operators are native to WGSL, we need to add the custom ones.
    for op in body.operators.iter() {
        match op {
            wgsl::Operation::Powf { lhs: _, rhs, out } => match rhs {
                wgsl::Variable::Scalar(_, _, _) => {
                    register_extension(wgsl::Extension::PowfScalar(*out.item()));
                }
                _ => {
                    register_extension(wgsl::Extension::Powf(*out.item()));
                }
            },
            wgsl::Operation::Erf { input, out: _ } => {
                register_extension(wgsl::Extension::Erf(*input.item()));
            }
            #[cfg(target_os = "macos")]
            wgsl::Operation::Tanh { input, out: _ } => {
                register_extension(wgsl::Extension::SafeTanh(*input.item()))
            }
            _ => {}
        }
    }

    extensions
}
