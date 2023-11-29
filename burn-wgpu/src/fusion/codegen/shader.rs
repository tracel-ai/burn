use super::{Body, Function};
use crate::kernel::{DynamicKernelSource, SourceTemplate, WORKGROUP_DEFAULT};
use std::{
    collections::hash_map::DefaultHasher,
    fmt::Display,
    hash::{Hash, Hasher},
};

#[derive(Hash, PartialEq, Eq)]
pub enum Location {
    Storage,
    #[allow(dead_code)]
    Workgroup,
}

#[derive(Hash, PartialEq, Eq)]
pub enum Visibility {
    Read,
    ReadWrite,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq, Copy)]
pub enum Elem {
    F32,
    #[allow(dead_code)]
    I32,
    U32,
    Bool,
}

#[derive(Hash, PartialEq, Eq)]
pub struct Binding {
    pub location: Location,
    pub visibility: Visibility,
    pub elem: Elem,
    pub size: Option<usize>,
}

#[derive(Hash, PartialEq, Eq)]
pub struct WorkgroupSize {
    pub x: usize,
    pub y: usize,
    pub z: usize,
}

impl Default for WorkgroupSize {
    fn default() -> Self {
        Self {
            x: WORKGROUP_DEFAULT,
            y: WORKGROUP_DEFAULT,
            z: 1,
        }
    }
}

#[derive(Hash)]
pub struct ComputeShader {
    pub inputs: Vec<Binding>,
    pub outputs: Vec<Binding>,
    pub named: Vec<(String, Binding)>,
    pub workgroup_size: WorkgroupSize,
    pub global_invocation_id: bool,
    pub num_workgroups: bool,
    pub body: Body,
    pub functions: Vec<Function>,
}

impl DynamicKernelSource for ComputeShader {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(self.to_string())
    }

    fn id(&self) -> String {
        let mut s = DefaultHasher::new();
        self.hash(&mut s);

        s.finish().to_string()
    }
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

        for function in self.functions.iter() {
            f.write_fmt(format_args!("{function}\n\n"))?;
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
            Some(size) => format!("array<{}, {}>", binding.elem, size),
            None => format!("array<{}>", binding.elem),
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

impl Display for Elem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Elem::F32 => f.write_str("f32"),
            Elem::I32 => f.write_str("i32"),
            Elem::U32 => f.write_str("u32"),
            Elem::Bool => f.write_str("bool"),
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
