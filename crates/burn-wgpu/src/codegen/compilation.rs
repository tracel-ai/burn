use burn_fusion::TensorDescription;

use super::{dialect::gpu, Compiler};
use crate::{
    codegen::dialect::gpu::{
        Binding, ComputeShader, Elem, Item, Location, ReadingStrategy, Variable, Vectorization,
        Visibility, WorkgroupSize,
    },
    fusion::JitFusionHandle,
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
    pub mappings: Vec<InplaceMapping>,
}

/// Simply indicate the output that can be replaced by the input.
#[derive(new, Clone, Copy, Debug)]
pub struct InplaceMapping {
    /// Input position.
    pub pos_input: usize, // TODO: Add multiple possible inputs for each output and let the
    // compilation process handle which input is used with partial dynamic
    // inplace operations.
    /// Output position.
    pub pos_output: usize,
}

#[derive(Default, Clone, Debug)]
pub struct CompilationSettings {
    pub partial_inplace_mapping: Vec<InplaceMapping>,
    full_inplace: bool,
    vectorization: Vectorization,
    workgroup_size: WorkgroupSize,
    reading_strategy: Vec<(u16, ReadingStrategy)>,
}

impl CompilationSettings {
    /// Compile the shader with vectorization enabled.
    #[allow(dead_code)]
    pub fn vectorize(mut self, vectorization: Vectorization) -> Self {
        self.vectorization = vectorization;
        self
    }

    /// Compile the shader with inplace enabled.
    ///
    /// Notes:
    ///
    /// This won't guarantee that the shader will use input arrays as outputs, since it is only
    /// possible when [inplace mappings](InplaceMapping) are provided as [compilation info](CompilationInfo)
    pub fn inplace(mut self, inplace_enabled: bool) -> Self {
        self.full_inplace = inplace_enabled;
        self
    }

    pub fn dynamic_settings<R: Runtime>(
        self,
        info: &CompilationInfo,
        inputs: &[&TensorDescription],
        outputs: &[&TensorDescription],
        handles_inputs: &[JitFusionHandle<R>],
    ) -> Self {
        self.reading_strategy(info, inputs, outputs, handles_inputs)
            .inplace_partial(&info.mappings, handles_inputs)
    }

    fn reading_strategy<R: Runtime>(
        mut self,
        info: &CompilationInfo,
        inputs: &[&TensorDescription],
        outputs: &[&TensorDescription],
        handles_inputs: &[JitFusionHandle<R>],
    ) -> Self {
        let layout_ref = match info.scope.layout_ref {
            Some(val) => val,
            None => return self,
        };

        let layout_description = match layout_ref {
            Variable::GlobalInputArray(id, _) => &inputs[id as usize],
            Variable::GlobalOutputArray(id, _) => &outputs[id as usize],
            _ => return self,
        };

        for (input_id, strategy) in info.scope.read_globals() {
            if let ReadingStrategy::Plain = strategy {
                continue;
            };

            let index = input_id as usize;
            let handle = &handles_inputs[index];
            let description_input = &inputs[index];

            if description_input.shape != layout_description.shape {
                continue;
            }

            let mut is_contiguous = true;
            let mut current = 0;

            for stride in handle.strides.iter().rev() {
                if current > *stride {
                    is_contiguous = false;
                }
                current = *stride;
            }

            if is_contiguous {
                self.reading_strategy
                    .push((input_id, ReadingStrategy::Plain));
            }
        }
        self
    }

    /// Compile the shader with partial inplace mappings.
    fn inplace_partial<R: Runtime>(
        mut self,
        mappings: &[InplaceMapping],
        handles_inputs: &[JitFusionHandle<R>],
    ) -> Self {
        for mapping in mappings.iter() {
            let handle = &handles_inputs[mapping.pos_input];

            if !handle.handle.can_mut() {
                continue;
            }

            let mut is_contiguous = true;
            let mut current = 0;

            for stride in handle.strides.iter().rev() {
                if current > *stride {
                    is_contiguous = false;
                }
                current = *stride;
            }

            if is_contiguous {
                self.partial_inplace_mapping.push(mapping.clone());
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

/// Information related to an input.
#[derive(Clone)]
pub enum InputInfo {
    Array { item: Item, visibility: Visibility },
    Scalar { elem: Elem, size: usize },
}

/// Information related to an output.
#[derive(Clone)]
pub enum OutputInfo {
    /// Write the local variable to a new array.
    ///
    /// This will create a new binding in the [compute shader](ComputeShader).
    ArrayWrite { item: Item, local: u16 },
    /// Write the local variable to an existing input binding.
    InputArrayWrite { item: Item, input: u16, local: u16 },
    /// Simply register the output, but don't automatically add a write to it.
    ///
    /// Useful when a [procedure](gpu::Procedure) writes to the output using
    /// [operations](gpu::Operation).
    Array { item: Item },
}

impl OutputInfo {
    pub fn elem_size<R: Runtime>(&self) -> usize {
        let elem = match self {
            OutputInfo::ArrayWrite { item, local: _ } => bool_elem(item.elem()),
            OutputInfo::InputArrayWrite {
                item,
                input: _,
                local: _,
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
        self.info.scope.vectorize(settings.vectorization);

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
                    let item = item.vectorize(settings.vectorization);

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

        if settings.full_inplace {
            let mut mappings = Vec::new();
            core::mem::swap(&mut self.info.mappings, &mut mappings);

            for mapping in mappings {
                self.register_inplace_mapping(mapping);
            }
        } else if !settings.partial_inplace_mapping.is_empty() {
            let mut mappings = Vec::new();
            core::mem::swap(&mut settings.partial_inplace_mapping, &mut mappings);

            for mapping in mappings {
                self.register_inplace_mapping(mapping);
            }
        }

        for array in self.info.outputs.drain(..) {
            match array {
                OutputInfo::ArrayWrite { item, local } => {
                    let item = item.vectorize(settings.vectorization);
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
                    );
                    index += 1;
                }
                OutputInfo::InputArrayWrite { item, input, local } => {
                    let item = item.vectorize(settings.vectorization);

                    self.info.scope.write_global(
                        Variable::Local(local, item, self.info.scope.depth),
                        Variable::GlobalInputArray(input, bool_item(item)),
                    );
                }
                OutputInfo::Array { item } => {
                    let item = item.vectorize(settings.vectorization);
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
            None => return, // No output to update.
        };

        let (item, local) = match output {
            OutputInfo::ArrayWrite { item, local } => (item, local),
            OutputInfo::InputArrayWrite {
                item: _,
                input: _,
                local: _,
            } => return,
            OutputInfo::Array { item: _ } => return,
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
