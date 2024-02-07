use super::{shader::WgslComputeShader, WgslItem};
use crate::{
    codegen::{
        compiler::Compiler,
        dialect::{gpu, wgsl},
    },
    element::WgpuElement,
};
use std::marker::PhantomData;

pub struct WgslCompiler<F: WgpuElement, I: WgpuElement> {
    _float: PhantomData<F>,
    _int: PhantomData<I>,
}

impl<F: WgpuElement, I: WgpuElement> Compiler for WgslCompiler<F, I> {
    type Representation = WgslComputeShader;

    fn compile(shader: gpu::ComputeShader) -> Self::Representation {
        Self::compile_shader(shader)
    }
}

impl<F: WgpuElement, I: WgpuElement> WgslCompiler<F, I> {
    fn compile_item(item: gpu::Item) -> WgslItem {
        match item {
            gpu::Item::Vec4(elem) => wgsl::WgslItem::Vec4(Self::compile_elem(elem)),
            gpu::Item::Vec3(elem) => wgsl::WgslItem::Vec3(Self::compile_elem(elem)),
            gpu::Item::Vec2(elem) => wgsl::WgslItem::Vec2(Self::compile_elem(elem)),
            gpu::Item::Scalar(elem) => wgsl::WgslItem::Scalar(Self::compile_elem(elem)),
        }
    }

    fn compile_elem(value: gpu::Elem) -> wgsl::WgslElem {
        match value {
            gpu::Elem::Float => wgsl::WgslElem::F32,
            gpu::Elem::Int => wgsl::WgslElem::I32,
            gpu::Elem::UInt => wgsl::WgslElem::U32,
            gpu::Elem::Bool => wgsl::WgslElem::Bool,
        }
    }

    fn compile_variable(value: gpu::Variable) -> wgsl::WgslVariable {
        match value {
            gpu::Variable::Input(index, item) => {
                wgsl::WgslVariable::Input(index, Self::compile_item(item))
            }
            gpu::Variable::Scalar(index, item) => {
                let elem = item.elem();
                wgsl::WgslVariable::Scalar(index, Self::compile_item(item), elem)
            }
            gpu::Variable::Local(index, item) => {
                wgsl::WgslVariable::Local(index, Self::compile_item(item))
            }
            gpu::Variable::Output(index, item) => {
                wgsl::WgslVariable::Output(index, Self::compile_item(item))
            }
            gpu::Variable::Constant(index, item) => {
                wgsl::WgslVariable::Constant(index, Self::compile_item(item))
            }
        }
    }

    fn compile_body(value: gpu::Body) -> wgsl::WgslBody {
        wgsl::WgslBody {
            operators: value
                .operators
                .into_iter()
                .map(Self::compile_operation)
                .collect(),
        }
    }

    fn compile_operation(value: gpu::Operation) -> wgsl::WgslOperation {
        match value {
            gpu::Operation::Add(op) => wgsl::WgslOperation::Add {
                lhs: Self::compile_variable(op.lhs),
                rhs: Self::compile_variable(op.rhs),
                out: Self::compile_variable(op.out),
            },
            gpu::Operation::Sub(op) => wgsl::WgslOperation::Sub {
                lhs: Self::compile_variable(op.lhs),
                rhs: Self::compile_variable(op.rhs),
                out: Self::compile_variable(op.out),
            },
            gpu::Operation::Mul(op) => wgsl::WgslOperation::Mul {
                lhs: Self::compile_variable(op.lhs),
                rhs: Self::compile_variable(op.rhs),
                out: Self::compile_variable(op.out),
            },
            gpu::Operation::Div(op) => wgsl::WgslOperation::Div {
                lhs: Self::compile_variable(op.lhs),
                rhs: Self::compile_variable(op.rhs),
                out: Self::compile_variable(op.out),
            },
            gpu::Operation::Abs(op) => wgsl::WgslOperation::Abs {
                input: Self::compile_variable(op.input),
                out: Self::compile_variable(op.out),
            },
            gpu::Operation::Exp(op) => wgsl::WgslOperation::Exp {
                input: Self::compile_variable(op.input),
                out: Self::compile_variable(op.out),
            },
            gpu::Operation::Log(op) => wgsl::WgslOperation::Log {
                input: Self::compile_variable(op.input),
                out: Self::compile_variable(op.out),
            },
            gpu::Operation::Log1p(op) => wgsl::WgslOperation::Log1p {
                input: Self::compile_variable(op.input),
                out: Self::compile_variable(op.out),
            },
            gpu::Operation::Cos(op) => wgsl::WgslOperation::Cos {
                input: Self::compile_variable(op.input),
                out: Self::compile_variable(op.out),
            },
            gpu::Operation::Sin(op) => wgsl::WgslOperation::Sin {
                input: Self::compile_variable(op.input),
                out: Self::compile_variable(op.out),
            },
            gpu::Operation::Tanh(op) => wgsl::WgslOperation::Tanh {
                input: Self::compile_variable(op.input),
                out: Self::compile_variable(op.out),
            },
            gpu::Operation::Powf(op) => wgsl::WgslOperation::Powf {
                lhs: Self::compile_variable(op.lhs),
                rhs: Self::compile_variable(op.rhs),
                out: Self::compile_variable(op.out),
            },
            gpu::Operation::Sqrt(op) => wgsl::WgslOperation::Sqrt {
                input: Self::compile_variable(op.input),
                out: Self::compile_variable(op.out),
            },
            gpu::Operation::Erf(op) => wgsl::WgslOperation::Erf {
                input: Self::compile_variable(op.input),
                out: Self::compile_variable(op.out),
            },
            gpu::Operation::Recip(op) => wgsl::WgslOperation::Recip {
                input: Self::compile_variable(op.input),
                out: Self::compile_variable(op.out),
            },
            gpu::Operation::Equal(op) => wgsl::WgslOperation::Equal {
                lhs: Self::compile_variable(op.lhs),
                rhs: Self::compile_variable(op.rhs),
                out: Self::compile_variable(op.out),
            },
            gpu::Operation::Lower(op) => wgsl::WgslOperation::Lower {
                lhs: Self::compile_variable(op.lhs),
                rhs: Self::compile_variable(op.rhs),
                out: Self::compile_variable(op.out),
            },
            gpu::Operation::Clamp(op) => wgsl::WgslOperation::Clamp {
                input: Self::compile_variable(op.input),
                min_value: Self::compile_variable(op.min_value),
                max_value: Self::compile_variable(op.max_value),
                out: Self::compile_variable(op.out),
            },
            gpu::Operation::Greater(op) => wgsl::WgslOperation::Greater {
                lhs: Self::compile_variable(op.lhs),
                rhs: Self::compile_variable(op.rhs),
                out: Self::compile_variable(op.out),
            },
            gpu::Operation::LowerEqual(op) => wgsl::WgslOperation::LowerEqual {
                lhs: Self::compile_variable(op.lhs),
                rhs: Self::compile_variable(op.rhs),
                out: Self::compile_variable(op.out),
            },
            gpu::Operation::GreaterEqual(op) => wgsl::WgslOperation::GreaterEqual {
                lhs: Self::compile_variable(op.lhs),
                rhs: Self::compile_variable(op.rhs),
                out: Self::compile_variable(op.out),
            },
            gpu::Operation::ConditionalAssign(op) => wgsl::WgslOperation::ConditionalAssign {
                cond: Self::compile_variable(op.cond),
                lhs: Self::compile_variable(op.lhs),
                rhs: Self::compile_variable(op.rhs),
                out: Self::compile_variable(op.out),
            },
            gpu::Operation::AssignGlobal(op) => wgsl::WgslOperation::AssignGlobal {
                input: Self::compile_variable(op.input),
                out: Self::compile_variable(op.out),
            },
            gpu::Operation::AssignLocal(op) => wgsl::WgslOperation::AssignLocal {
                input: Self::compile_variable(op.input),
                out: Self::compile_variable(op.out),
            },
            gpu::Operation::ReadGlobal(op) => wgsl::WgslOperation::ReadGlobal {
                variable: Self::compile_variable(op.variable),
            },
            gpu::Operation::ReadGlobalWithLayout(op) => wgsl::WgslOperation::ReadGlobalWithLayout {
                variable: Self::compile_variable(op.variable),
                tensor_read_pos: op.tensor_read_pos,
                tensor_layout_pos: op.tensor_layout_pos,
            },
        }
    }

    fn compile_location(value: gpu::Location) -> wgsl::WgslLocation {
        match value {
            gpu::Location::Storage => wgsl::WgslLocation::Storage,
            gpu::Location::Workgroup => wgsl::WgslLocation::Workgroup,
        }
    }

    fn compile_visibility(value: gpu::Visibility) -> wgsl::WgslVisibility {
        match value {
            gpu::Visibility::Read => wgsl::WgslVisibility::Read,
            gpu::Visibility::ReadWrite => wgsl::WgslVisibility::ReadWrite,
        }
    }

    fn compile_binding(value: gpu::Binding) -> wgsl::WgslBinding {
        wgsl::WgslBinding {
            visibility: Self::compile_visibility(value.visibility),
            location: Self::compile_location(value.location),
            item: Self::compile_item(value.item),
            size: value.size,
        }
    }

    fn compile_shader(value: gpu::ComputeShader) -> wgsl::WgslComputeShader {
        let body = Self::compile_body(value.body);
        let extensions = register_extensions(&body);

        wgsl::WgslComputeShader {
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

fn register_extensions(body: &wgsl::WgslBody) -> Vec<wgsl::WgslExtension> {
    let mut extensions = Vec::new();

    let mut register_extension = |extension: wgsl::WgslExtension| {
        if !extensions.contains(&extension) {
            extensions.push(extension);
        }
    };

    // Since not all operators are native to WGSL, we need to add the custom ones.
    for op in body.operators.iter() {
        match op {
            wgsl::WgslOperation::Powf { lhs: _, rhs, out } => match rhs {
                wgsl::WgslVariable::Scalar(_, _, _) => {
                    register_extension(wgsl::WgslExtension::PowfScalar(*out.item()));
                }
                _ => {
                    register_extension(wgsl::WgslExtension::Powf(*out.item()));
                }
            },
            wgsl::WgslOperation::Erf { input, out: _ } => {
                register_extension(wgsl::WgslExtension::Erf(*input.item()));
            }
            #[cfg(target_os = "macos")]
            wgsl::WgslOperation::Tanh { input, out: _ } => {
                register_function(wgsl::WgslExtension::SafeTanh(*input.item()))
            }
            _ => {}
        }
    }

    extensions
}
