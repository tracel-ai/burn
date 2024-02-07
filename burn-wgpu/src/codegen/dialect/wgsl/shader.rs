use super::base::WgslVariable;
use super::operations::WgslOperation;
use super::{base::WgslItem, body::WgslBody};
use crate::codegen::dialect::gpu::{self, WorkgroupSize};
use crate::codegen::dialect::wgsl::WgslExtension;
use std::fmt::Display;

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum WgslLocation {
    Storage,
    #[allow(dead_code)]
    Workgroup,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum WgslVisibility {
    Read,
    ReadWrite,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct WgslBinding {
    pub location: WgslLocation,
    pub visibility: WgslVisibility,
    pub item: WgslItem,
    pub size: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct WgslComputeShader {
    pub inputs: Vec<WgslBinding>,
    pub outputs: Vec<WgslBinding>,
    pub named: Vec<(String, WgslBinding)>,
    pub workgroup_size: WorkgroupSize,
    pub global_invocation_id: bool,
    pub num_workgroups: bool,
    pub body: WgslBody,
    pub extensions: Vec<WgslExtension>,
}

impl Display for WgslComputeShader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Self::format_bindings(f, "input", &self.inputs, 0)?;
        Self::format_bindings(f, "output", &self.outputs, self.inputs.len())?;

        for (i, (name, binding)) in self.named.iter().enumerate() {
            Self::format_binding(
                f,
                name.as_str(),
                binding,
                self.inputs.len() + self.outputs.len() + i,
            )?;
        }

        f.write_fmt(format_args!(
            "const WORKGROUP_SIZE_X = {}u;
const WORKGROUP_SIZE_Y = {}u;
const WORKGROUP_SIZE_Z = {}u;\n",
            self.workgroup_size.x, self.workgroup_size.y, self.workgroup_size.z
        ))?;

        f.write_fmt(format_args!(
            "
@compute
@workgroup_size({}, {}, {})
fn main(
",
            self.workgroup_size.x, self.workgroup_size.y, self.workgroup_size.z
        ))?;

        if self.global_invocation_id {
            f.write_str("    @builtin(global_invocation_id) global_id: vec3<u32>,\n")?;
        }

        if self.num_workgroups {
            f.write_str("    @builtin(num_workgroups) num_workgroups: vec3<u32>,\n")?;
        }

        f.write_fmt(format_args!(
            ") {{
    {}
}}",
            self.body
        ))?;

        for extension in self.extensions.iter() {
            f.write_fmt(format_args!("{extension}\n\n"))?;
        }

        Ok(())
    }
}

impl WgslComputeShader {
    fn format_bindings(
        f: &mut core::fmt::Formatter<'_>,
        prefix: &str,
        bindings: &[WgslBinding],
        num_entry: usize,
    ) -> core::fmt::Result {
        for (i, binding) in bindings.iter().enumerate() {
            Self::format_binding(
                f,
                format!("{prefix}_{i}_global").as_str(),
                binding,
                num_entry + i,
            )?;
        }

        Ok(())
    }

    fn format_binding(
        f: &mut core::fmt::Formatter<'_>,
        name: &str,
        binding: &WgslBinding,
        num_entry: usize,
    ) -> core::fmt::Result {
        let ty = match binding.size {
            Some(size) => format!("array<{}, {}>", binding.item, size),
            None => format!("array<{}>", binding.item),
        };

        f.write_fmt(format_args!(
            "@group(0)
@binding({})
var<{}, {}> {}: {};
\n",
            num_entry, binding.location, binding.visibility, name, ty
        ))?;

        Ok(())
    }
}

impl Display for WgslLocation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WgslLocation::Storage => f.write_str("storage"),
            WgslLocation::Workgroup => f.write_str("workgroup"),
        }
    }
}

impl Display for WgslVisibility {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WgslVisibility::Read => f.write_str("read"),
            WgslVisibility::ReadWrite => f.write_str("read_write"),
        }
    }
}

fn register_extensions(body: &WgslBody) -> Vec<WgslExtension> {
    let mut extensions = Vec::new();

    let mut register_extension = |extension: WgslExtension| {
        if !extensions.contains(&extension) {
            extensions.push(extension);
        }
    };

    // Since not all operators are native to WGSL, we need to add the custom ones.
    for op in body.operators.iter() {
        match op {
            WgslOperation::Powf { lhs: _, rhs, out } => match rhs {
                WgslVariable::Scalar(_, _) => {
                    register_extension(WgslExtension::PowfScalar(*out.item()));
                }
                _ => {
                    register_extension(WgslExtension::Powf(*out.item()));
                }
            },
            WgslOperation::Erf { input, out: _ } => {
                register_extension(WgslExtension::Erf(*input.item()));
            }
            #[cfg(target_os = "macos")]
            WgslOperation::Tanh { input, out: _ } => {
                register_function(WgslExtension::SafeTanh(*input.item()))
            }
            _ => {}
        }
    }

    extensions
}

impl From<gpu::Location> for WgslLocation {
    fn from(value: gpu::Location) -> Self {
        match value {
            gpu::Location::Storage => WgslLocation::Storage,
            gpu::Location::Workgroup => WgslLocation::Workgroup,
        }
    }
}

impl From<gpu::Visibility> for WgslVisibility {
    fn from(value: gpu::Visibility) -> Self {
        match value {
            gpu::Visibility::Read => WgslVisibility::Read,
            gpu::Visibility::ReadWrite => WgslVisibility::ReadWrite,
        }
    }
}

impl From<gpu::Binding> for WgslBinding {
    fn from(value: gpu::Binding) -> Self {
        Self {
            visibility: value.visibility.into(),
            location: value.location.into(),
            item: value.item.into(),
            size: value.size,
        }
    }
}

impl From<gpu::ComputeShader> for WgslComputeShader {
    fn from(value: gpu::ComputeShader) -> Self {
        let body = value.body.into();
        let extensions = register_extensions(&body);

        Self {
            inputs: value.inputs.into_iter().map(From::from).collect(),
            outputs: value.outputs.into_iter().map(From::from).collect(),
            named: value
                .named
                .into_iter()
                .map(|(name, binding)| (name, binding.into()))
                .collect(),
            workgroup_size: value.workgroup_size,
            global_invocation_id: value.global_invocation_id,
            num_workgroups: value.num_workgroups,
            body,
            extensions,
        }
    }
}
