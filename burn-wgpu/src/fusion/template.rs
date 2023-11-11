use crate::kernel::{DynamicKernelSource, SourceTemplate, WORKGROUP_DEFAULT};
use std::{
    collections::hash_map::DefaultHasher,
    fmt::Display,
    hash::{Hash, Hasher},
};

#[derive(Hash, PartialEq, Eq)]
pub enum Location {
    Storage,
    Workgroup,
}

#[derive(Hash, PartialEq, Eq)]
pub enum Visibility {
    Read,
    ReadWrite,
}

#[derive(Hash, PartialEq, Eq)]
pub enum Elem {
    F32,
    I32,
    U32,
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

pub struct WgslTempate {
    pub inputs: Vec<Binding>,
    pub outputs: Vec<Binding>,
    pub info: Option<Binding>,
    pub workgroup_sizes: WorkgroupSize,
    pub global_invocation_id: bool,
    pub num_workgroups: bool,
    pub body: Box<dyn DynamicKernelSource>,
}

impl WgslTempate {
    fn format_bindings(prefix: &str, bindings: &[Binding], num_entry: usize) -> String {
        let mut output = String::new();

        for (i, binding) in bindings.iter().enumerate() {
            output += Self::format_binding(
                format!("{prefix}_{i}_global").as_str(),
                binding,
                num_entry + i,
            )
            .as_str();
        }

        output
    }

    fn format_binding(name: &str, binding: &Binding, num_entry: usize) -> String {
        let ty = match binding.size {
            Some(size) => format!("array<{}; {}>", binding.elem, size),
            None => format!("array<{}>", binding.elem),
        };

        format!(
            "@group(0)
@binding({})
var<{}, {}> {}: {};
\n",
            num_entry, binding.location, binding.visibility, name, ty
        )
    }
}

impl DynamicKernelSource for WgslTempate {
    fn source(&self) -> SourceTemplate {
        let mut source_template = String::new();
        source_template += Self::format_bindings("input", &self.inputs, 0).as_str();
        source_template +=
            Self::format_bindings("output", &self.outputs, self.inputs.len()).as_str();

        if let Some(info) = &self.info {
            source_template +=
                Self::format_binding("info", info, self.inputs.len() + self.outputs.len()).as_str();
        }

        source_template += format!(
            "const WORKGROUP_SIZE_X = {}u;
const WORKGROUP_SIZE_Y = {}u;
const WORKGROUP_SIZE_Z = {}u;\n",
            self.workgroup_sizes.x, self.workgroup_sizes.y, self.workgroup_sizes.z
        )
        .as_str();
        source_template += format!(
            "
@compute
@workgroup_size({}, {}, {})
fn main(
",
            self.workgroup_sizes.x, self.workgroup_sizes.y, self.workgroup_sizes.z
        )
        .as_str();

        if self.global_invocation_id {
            source_template += "    @builtin(global_invocation_id) global_id: vec3<u32>,\n";
        }

        if self.num_workgroups {
            source_template += "    @builtin(num_workgroups) num_workgroups: vec3<u32>,\n";
        }

        source_template += ") {
    {{ body }}
}";

        SourceTemplate::new(source_template).register("body", self.body.source().complete())
    }

    fn id(&self) -> String {
        let mut s = DefaultHasher::new();
        self.inputs.hash(&mut s);
        self.outputs.hash(&mut s);
        self.info.hash(&mut s);
        self.workgroup_sizes.hash(&mut s);
        self.global_invocation_id.hash(&mut s);
        self.num_workgroups.hash(&mut s);

        self.body.id() + &s.finish().to_string()
    }
}
