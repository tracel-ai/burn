use super::{Body, Extension, Item};
use crate::codegen::dialect::gpu::WorkgroupSize;
use std::fmt::Display;

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum Location {
    Storage,
    #[allow(dead_code)]
    Workgroup,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum Visibility {
    Read,
    ReadWrite,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Binding {
    pub location: Location,
    pub visibility: Visibility,
    pub item: Item,
    pub size: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct ComputeShader {
    pub inputs: Vec<Binding>,
    pub outputs: Vec<Binding>,
    pub named: Vec<(String, Binding)>,
    pub workgroup_size: WorkgroupSize,
    pub global_invocation_id: bool,
    pub num_workgroups: bool,
    pub body: Body,
    pub extensions: Vec<Extension>,
}

impl Display for ComputeShader {
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

impl ComputeShader {
    fn format_bindings(
        f: &mut core::fmt::Formatter<'_>,
        prefix: &str,
        bindings: &[Binding],
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
        binding: &Binding,
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

impl Display for Location {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Location::Storage => f.write_str("storage"),
            Location::Workgroup => f.write_str("workgroup"),
        }
    }
}

impl Display for Visibility {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Visibility::Read => f.write_str("read"),
            Visibility::ReadWrite => f.write_str("read_write"),
        }
    }
}
