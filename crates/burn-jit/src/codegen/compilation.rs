#[cfg(feature = "fusion")]
use crate::fusion::JitFusionHandle;

use super::{
    dialect::gpu::{self},
    Compiler,
};
use crate::{
    codegen::dialect::gpu::{
        Binding, ComputeShader, Elem, Item, Location, ReadingStrategy, Variable, Vectorization,
        Visibility, WorkgroupSize,
    },
    Runtime,
};

/// The compilation struct allows you to create a [compute shader](ComputeShader) based on
/// [compilation info](CompilationInfo) and [compilation settings](CompilationSettings).
#[derive(Clone)]
pub struct Compilation {
    info: CompilationInfo,
    input_bindings: Vec<Binding>,
    output_bindings: Vec<Binding>,
    named_bindings: Vec<(String, Binding)>,
}

/// The information necessary to compile a [compute shader](ComputeShader).
#[derive(Clone)]
pub struct CompilationInfo {
    pub inputs: Vec<InputInfo>,
    pub outputs: Vec<OutputInfo>,
    pub scope: gpu::Scope,
}

/// Simply indicate the output that can be replaced by the input.
#[derive(new, Clone, Copy, Debug)]
pub struct InplaceMapping {
    /// Input position.
    pub pos_input: usize,
    /// Output position.
    pub pos_output: usize,
}

#[derive(Default, Clone)]
pub struct CompilationSettings {
    pub mappings: Vec<InplaceMapping>,
    vectorization: Option<Vectorization>,
    workgroup_size: WorkgroupSize,
    reading_strategy: Vec<(u16, ReadingStrategy)>,
}

impl core::fmt::Display for CompilationSettings {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The goal of this implementation is to generate the shortest representation
        // that won't clash with any other compilation settings. This is crucial since we rely on
        // this representation to know when to compile a new version of a kernel.
        //
        // Each main section starts with a letter that can't be used by other main sections:
        //
        // * Mapping:          m
        //   * Input:  i
        //   * Output: o
        //
        // * Reading Strategy: r
        //   * Output layout: o
        //   * Plain:         p
        //
        // * Vectorization:    v
        // * Workgroup Size X: x
        // * Workgroup Size Y: y
        // * Workgroup Size Z: z
        f.write_str("m")?;
        for mapping in self.mappings.iter() {
            f.write_fmt(format_args!(
                "i{}o{}",
                mapping.pos_input, mapping.pos_output
            ))?;
        }

        f.write_str("r")?;

        for (input, strategy) in self.reading_strategy.iter() {
            match strategy {
                ReadingStrategy::OutputLayout => f.write_fmt(format_args!("i{}o", input)),
                ReadingStrategy::Plain => f.write_fmt(format_args!("i{}p", input)),
            }?;
        }

        match self.vectorization {
            Some(vectorization) => match vectorization {
                Vectorization::Vec4 => f.write_str("v4"),
                Vectorization::Vec3 => f.write_str("v3"),
                Vectorization::Vec2 => f.write_str("v2"),
                Vectorization::Scalar => f.write_str("v1"),
            }?,
            None => f.write_str("vn")?,
        };

        f.write_fmt(format_args!(
            "x{}y{}z{}",
            self.workgroup_size.x, self.workgroup_size.y, self.workgroup_size.x
        ))
    }
}

impl CompilationSettings {
    /// Compile the shader with vectorization enabled.
    #[allow(dead_code)]
    pub fn vectorize(mut self, vectorization: Vectorization) -> Self {
        self.vectorization = Some(vectorization);
        self
    }

    /// Compile the shader with inplace enabled by the given [mapping](InplaceMapping).
    ///
    /// Notes:
    ///
    /// You should favor using `dynamic_settings` when using fusion, since the mapping is going to
    /// be created from the runtime information.
    pub fn inplace(mut self, mappings: Vec<InplaceMapping>) -> Self {
        self.mappings = mappings;
        self
    }

    #[cfg(feature = "fusion")]
    /// Apply dynamic settings based on the runtime information captured by the `burn-fusion`
    /// project.
    ///
    /// Two optimizations are done here:
    ///
    /// 1. Find and remove unnecessary broadcasting procedures based on runtime tensor layouts.
    ///
    /// 2. (Optional) Find which inputs can be used inplaced based on runtime tensor layouts and captured tensor
    ///    descriptions. This is enabled only when stateful is set to true.
    pub fn dynamic_settings<R: Runtime>(
        self,
        info: &CompilationInfo,
        inputs: &[&burn_tensor::repr::TensorDescription],
        outputs: &[&burn_tensor::repr::TensorDescription],
        handles_inputs: &[JitFusionHandle<R>],
        stateful: bool,
    ) -> Self {
        let mut settings = self;

        if stateful {
            settings = settings.dynamic_inplace(info, inputs, outputs, handles_inputs);
        }

        settings.dynamic_reading_strategy(info, inputs, outputs, handles_inputs)
    }

    #[cfg(feature = "fusion")]
    fn dynamic_inplace<R: Runtime>(
        self,
        info: &CompilationInfo,
        inputs: &[&burn_tensor::repr::TensorDescription],
        outputs: &[&burn_tensor::repr::TensorDescription],
        handles_inputs: &[JitFusionHandle<R>],
    ) -> Self {
        let mut potential_inplace = inputs
            .iter()
            .zip(info.inputs.iter())
            .enumerate()
            .filter_map(|(pos, (desc, input))| {
                match desc.status {
                    burn_tensor::repr::TensorStatus::ReadOnly => return None,
                    burn_tensor::repr::TensorStatus::NotInit => return None,
                    burn_tensor::repr::TensorStatus::ReadWrite => (),
                };

                let handle = &handles_inputs[pos];

                if handle.handle.can_mut() && is_contiguous(&handle.strides) {
                    Some((pos, desc, input))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();

        let mappings = outputs
            .iter()
            .zip(info.outputs.iter())
            .enumerate()
            .filter_map(|(pos, (desc, output))| {
                if potential_inplace.is_empty() {
                    return None;
                }

                for (index, (_, desc_input, input)) in potential_inplace.iter().enumerate() {
                    if desc.shape == desc_input.shape && input.item() == output.item() {
                        let (pos_input, _desc, _info) = potential_inplace.remove(index);
                        return Some(InplaceMapping::new(pos_input, pos));
                    }
                }

                None
            })
            .collect();

        self.inplace(mappings)
    }

    #[cfg(feature = "fusion")]
    fn dynamic_reading_strategy<R: Runtime>(
        mut self,
        info: &CompilationInfo,
        inputs: &[&burn_tensor::repr::TensorDescription],
        outputs: &[&burn_tensor::repr::TensorDescription],
        handles_inputs: &[JitFusionHandle<R>],
    ) -> Self {
        // First output is chosen for the layout reference.
        // but all outputs should have the same shape anyways.
        let layout_shape = &outputs[0].shape;

        for (input_id, strategy) in info.scope.read_globals() {
            if let ReadingStrategy::Plain = strategy {
                continue;
            };

            let index = input_id as usize;
            let handle = &handles_inputs[index];
            let description_input = &inputs[index];

            if &description_input.shape != layout_shape {
                continue;
            }

            if is_contiguous(&handle.strides) {
                self.reading_strategy
                    .push((input_id, ReadingStrategy::Plain));
            }
        }
        self
    }

    /// Set the grid size.
    #[allow(dead_code)] // Only used for fusion for now.
    pub fn workgroup_size(mut self, workgroup_size: WorkgroupSize) -> Self {
        self.workgroup_size = workgroup_size;
        self
    }
}

#[allow(dead_code)]
fn is_contiguous(strides: &[usize]) -> bool {
    let mut current = 0;

    for stride in strides.iter().rev() {
        if current > *stride {
            return false;
        }
        current = *stride;
    }

    true
}

/// Information related to an input.
#[derive(Clone, Debug)]
pub enum InputInfo {
    Array { item: Item, visibility: Visibility },
    Scalar { elem: Elem, size: usize },
}

impl InputInfo {
    /// The item type of the input.
    #[allow(dead_code)]
    pub fn item(&self) -> Item {
        match self {
            InputInfo::Array {
                item,
                visibility: _,
            } => *item,
            InputInfo::Scalar { elem, size: _ } => Item::Scalar(*elem),
        }
    }
}

impl OutputInfo {
    /// The item type of the input.
    #[allow(dead_code)]
    pub fn item(&self) -> Item {
        match self {
            OutputInfo::ArrayWrite {
                item,
                local: _,
                position: _,
            } => *item,
            OutputInfo::InputArrayWrite {
                item,
                input: _,
                local: _,
                position: _,
            } => *item,
            OutputInfo::Array { item } => *item,
        }
    }
}

/// Information related to an output.
#[derive(Clone, Debug)]
pub enum OutputInfo {
    /// Write the local variable to a new array.
    ///
    /// This will create a new binding in the [compute shader](ComputeShader).
    ArrayWrite {
        item: Item,
        local: u16,
        position: Variable,
    },
    /// Write the local variable to an existing input binding.
    InputArrayWrite {
        item: Item,
        input: u16,
        local: u16,
        position: Variable,
    },
    /// Simply register the output, but don't automatically add a write to it.
    ///
    /// Useful when a [procedure](gpu::Procedure) writes to the output using
    /// [operations](gpu::Operation).
    Array { item: Item },
}

impl OutputInfo {
    #[allow(dead_code)]
    pub fn elem_size<R: Runtime>(&self) -> usize {
        let elem = match self {
            OutputInfo::ArrayWrite {
                item,
                local: _,
                position: _,
            } => bool_elem(item.elem()),
            OutputInfo::InputArrayWrite {
                item,
                input: _,
                local: _,
                position: _,
            } => bool_elem(item.elem()),
            OutputInfo::Array { item } => bool_elem(item.elem()),
        };
        <R::Compiler as Compiler>::elem_size(elem)
    }
}

impl Compilation {
    /// Starts a new compilation.
    pub fn new(info: CompilationInfo) -> Self {
        Self {
            info,
            input_bindings: Default::default(),
            output_bindings: Default::default(),
            named_bindings: Default::default(),
        }
    }

    /// Performs the compilation with the provided [settings](CompilationSettings).
    pub fn compile(mut self, mut settings: CompilationSettings) -> ComputeShader {
        if let Some(vectorization) = settings.vectorization {
            self.info.scope.vectorize(vectorization);
        }

        self.register_inputs(&settings);
        self.register_outputs(&mut settings);

        let inputs = self.input_bindings;
        let outputs = self.output_bindings;
        let mut named = Vec::with_capacity(2);

        named.push((
            "info".to_string(),
            Binding {
                item: Item::Scalar(Elem::UInt),
                visibility: Visibility::Read,
                location: Location::Storage,
                size: None, // We avoid putting the length here since it will force a new kernel
                            // for each tensor rank.
            },
        ));

        for (name, binding) in self.named_bindings.into_iter() {
            named.push((name, binding));
        }

        ComputeShader {
            inputs,
            outputs,
            named,
            workgroup_size: settings.workgroup_size,
            body: self.info.scope,
        }
    }

    fn register_inputs(&mut self, settings: &CompilationSettings) {
        for (id, strategy) in settings.reading_strategy.iter() {
            self.info.scope.update_read(*id, *strategy);
        }

        for input in self.info.inputs.drain(..) {
            match input {
                InputInfo::Array { item, visibility } => {
                    let item = if let Some(vectorization) = settings.vectorization {
                        item.vectorize(vectorization)
                    } else {
                        item
                    };

                    self.input_bindings.push(Binding {
                        item: bool_item(item),
                        visibility,
                        location: Location::Storage,
                        size: None,
                    });
                }
                InputInfo::Scalar { elem, size } => {
                    let elem = bool_elem(elem);

                    self.named_bindings.push((
                        format!("scalars_{}", elem),
                        Binding {
                            item: Item::Scalar(elem),
                            visibility: Visibility::Read,
                            location: Location::Storage,
                            size: Some(size),
                        },
                    ));
                }
            }
        }
    }

    fn register_outputs(&mut self, settings: &mut CompilationSettings) {
        let mut index = 0;

        if !settings.mappings.is_empty() {
            let mut mappings = Vec::new();
            core::mem::swap(&mut settings.mappings, &mut mappings);

            for mapping in mappings {
                self.register_inplace_mapping(mapping);
            }
        }

        for array in self.info.outputs.drain(..) {
            match array {
                OutputInfo::ArrayWrite {
                    item,
                    local,
                    position,
                } => {
                    let item = if let Some(vectorization) = settings.vectorization {
                        item.vectorize(vectorization)
                    } else {
                        item
                    };
                    let elem_adapted = bool_item(item);

                    self.output_bindings.push(Binding {
                        item: elem_adapted,
                        visibility: Visibility::ReadWrite,
                        location: Location::Storage,
                        size: None,
                    });
                    self.info.scope.write_global(
                        Variable::Local(local, item, self.info.scope.depth),
                        Variable::GlobalOutputArray(index, elem_adapted),
                        position,
                    );
                    index += 1;
                }
                OutputInfo::InputArrayWrite {
                    item,
                    input,
                    local,
                    position,
                } => {
                    let item = if let Some(vectorization) = settings.vectorization {
                        item.vectorize(vectorization)
                    } else {
                        item
                    };

                    self.info.scope.write_global(
                        Variable::Local(local, item, self.info.scope.depth),
                        Variable::GlobalInputArray(input, bool_item(item)),
                        position,
                    );
                }
                OutputInfo::Array { item } => {
                    let item = if let Some(vectorization) = settings.vectorization {
                        item.vectorize(vectorization)
                    } else {
                        item
                    };
                    let elem_adapted = bool_item(item);

                    self.output_bindings.push(Binding {
                        item: elem_adapted,
                        visibility: Visibility::ReadWrite,
                        location: Location::Storage,
                        size: None,
                    });

                    index += 1;
                }
            }
        }
    }

    fn register_inplace_mapping(&mut self, mapping: InplaceMapping) {
        let output = match self.info.outputs.get_mut(mapping.pos_output) {
            Some(output) => output,
            None => panic!("No output found."),
        };

        let (item, local, position) = match output {
            OutputInfo::ArrayWrite { item, local, position } => (item, local, position),
            OutputInfo::InputArrayWrite {
                item: _,
                input,
                local: _,
                position: _,
            } => {
                assert_eq!(
                    *input, mapping.pos_input as u16,
                    "Can't use different inputs for the same output."
                );
                return;
            }
            OutputInfo::Array { item: _ } => panic!("Can't register an inplace operation for an array that isn't using a defined writing strategy."),
        };

        let item = match self.input_bindings.get_mut(mapping.pos_input) {
            Some(binding) => {
                // Update input visibility.
                binding.visibility = Visibility::ReadWrite;
                // Inputs modified inplace should be read without any specified layout.
                self.info
                    .scope
                    .update_read(mapping.pos_input as u16, gpu::ReadingStrategy::Plain);

                // Use the same item as the input.
                //
                // The output can be different (i.e inplace boolean operations on float bindings).
                binding.item
            }
            None => *item,
        };

        // Update the output.
        *output = OutputInfo::InputArrayWrite {
            item,
            input: mapping.pos_input as u16,
            local: *local,
            position: *position,
        };
    }
}

fn bool_item(ty: Item) -> Item {
    match ty {
        Item::Vec4(elem) => Item::Vec4(bool_elem(elem)),
        Item::Vec3(elem) => Item::Vec3(bool_elem(elem)),
        Item::Vec2(elem) => Item::Vec2(bool_elem(elem)),
        Item::Scalar(elem) => Item::Scalar(bool_elem(elem)),
    }
}

pub fn bool_elem(elem: Elem) -> Elem {
    match elem {
        // U32 are used for bool tensors
        Elem::Bool => Elem::UInt,
        _ => elem,
    }
}
