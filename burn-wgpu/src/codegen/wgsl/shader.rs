use super::base::WgslVariable;
use super::operations::WgslOperation;
use super::{base::WgslItem, body::WgslBody};
use crate::codegen::wgsl::extension::WgslExtension;
use crate::codegen::ComputeShader;
use crate::kernel::WORKGROUP_DEFAULT;
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

#[derive(new, Debug, PartialEq, Eq, Clone, Copy)]
pub struct WorkgroupSize {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl Default for WorkgroupSize {
    fn default() -> Self {
        Self {
            x: WORKGROUP_DEFAULT as u32,
            y: WORKGROUP_DEFAULT as u32,
            z: 1,
        }
    }
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

impl From<ComputeShader> for WgslComputeShader {
    fn from(value: ComputeShader) -> Self {
        todo!()
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
            WgslOperation::Powf {
                lhs: _,
                rhs,
                out: _,
            } => match rhs {
                WgslVariable::Scalar(_, _) => {
                    register_extension(WgslExtension::PowfScalar(*rhs.item()));
                }
                _ => {
                    register_extension(WgslExtension::Powf(*rhs.item()));
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
