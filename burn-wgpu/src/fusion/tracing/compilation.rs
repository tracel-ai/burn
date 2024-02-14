use crate::codegen::dialect::gpu::{
    Binding, ComputeShader, Elem, Item, Location, Variable, Vectorization, Visibility,
    WorkgroupSize,
};
use crate::codegen::{InplaceMapping, Input, Output};
use crate::fusion::tracing::CompilingInfo;

#[derive(Clone)]
pub struct Compilation {
    info: CompilingInfo,
    input_bindings: Vec<Binding>,
    output_bindings: Vec<Binding>,
    named_bindings: Vec<(String, Binding)>,
}

#[derive(Default)]
pub struct CompilationSettings {
    vectorization: Vectorization,
    inplace_available: bool,
    workgroup_size: WorkgroupSize,
}

impl CompilationSettings {
    #[allow(dead_code)]
    pub fn vectorize(mut self, vectorization: Vectorization) -> Self {
        self.vectorization = vectorization;
        self
    }
    #[allow(dead_code)]
    pub fn inplace(mut self, available: bool) -> Self {
        self.inplace_available = available;
        self
    }
    #[allow(dead_code)] // Only used for fusion for now.
    pub fn workgroup_size(mut self, workgroup_size: WorkgroupSize) -> Self {
        self.workgroup_size = workgroup_size;
        self
    }
}

impl Compilation {
    pub fn new(info: CompilingInfo) -> Self {
        Self {
            info,
            input_bindings: Default::default(),
            output_bindings: Default::default(),
            named_bindings: Default::default(),
        }
    }

    pub fn compile(mut self, settings: CompilationSettings) -> ComputeShader {
        self.info.scope.vectorize(settings.vectorization);

        self.register_inputs(&settings);
        self.register_outputs(&settings);

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
            num_workgroups: true,
            global_invocation_id: true,
        }
    }

    fn register_inputs(&mut self, settings: &CompilationSettings) {
        for input in self.info.inputs.drain(..) {
            match input {
                Input::Array {
                    item,
                    visibility,
                    strategy: _,
                } => {
                    let item = item.vectorize(settings.vectorization);

                    self.input_bindings.push(Binding {
                        item: bool_item(item),
                        visibility: visibility,
                        location: Location::Storage,
                        size: None,
                    });
                }
                Input::Scalar { elem, size } => {
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

    fn register_outputs(&mut self, settings: &CompilationSettings) {
        let mut index = 0;

        if settings.inplace_available {
            for mapping in self.info.mappings.iter() {
                match self.info.outputs.get_mut(mapping.position_output) {
                    Some(output) => match output {
                        Output::Array { item, local } => {
                            *output = Output::Input {
                                item: *item,
                                input: mapping.position_input as u16,
                                local: *local,
                            };
                        }
                        Output::Input {
                            item: _,
                            input: _,
                            local: _,
                        } => continue,
                    },
                    None => continue,
                }

                if let Some(binding) = self.input_bindings.get_mut(mapping.position_input) {
                    binding.visibility = Visibility::ReadWrite
                }
            }
        }

        for array in self.info.outputs.drain(..) {
            match array {
                Output::Array { item, local } => {
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
                        Variable::Output(index, elem_adapted),
                    );
                    index += 1;
                }
                Output::Input { item, input, local } => {
                    let item = item.vectorize(settings.vectorization);

                    self.info.scope.write_global(
                        Variable::Local(local, item, self.info.scope.depth),
                        Variable::Input(input, bool_item(item)),
                    );
                }
            }
        }
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

fn bool_elem(elem: Elem) -> Elem {
    match elem {
        // U32 are used for bool tensors
        Elem::Bool => Elem::UInt,
        _ => elem,
    }
}
