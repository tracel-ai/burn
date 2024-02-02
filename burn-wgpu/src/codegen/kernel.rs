use crate::codegen::{
    Binding, Body, ComputeShader, Elem, Function, Location, Operator, Variable, Visibility,
    WorkgroupSize,
};
use crate::compute::{StaticKernel, WgpuComputeClient, WgpuHandle};
use crate::element::WgpuElement;
use crate::kernel::{elemwise_workgroup, StaticKernelSource, WORKGROUP_DEFAULT};
use std::marker::PhantomData;

use super::Item;

/// Kernel creation input phase, see [kernel codegen](ElemWiseKernelCodegen) for more details.
pub struct InputPhase;
/// Kernel creation body phase, see [kernel codegen](ElemWiseKernelCodegen) for more details.
pub struct BodyPhase;
/// Kernel creation output phase, see [kernel codegen](ElemWiseKernelCodegen) for more details.
pub struct OutputPhase;
/// Kernel compilation phase, see [kernel codegen](ElemWiseKernelCodegen) for more details.
pub struct CompilationPhase;

#[derive(new, Clone, Copy)]
pub struct InplaceMapping {
    pub position_input: usize,
    pub position_output: usize,
}

/// Define a vectorization scheme.
#[allow(dead_code)]
#[derive(Copy, Clone)]
pub enum Vectorization {
    /// Use vec4 for vectorization.
    Vec4,
    /// Use vec3 for vectorization.
    Vec3,
    /// Use vec2 for vectorization.
    Vec2,
    /// Don't vectorize.
    Scalar,
}

/// Allows to create custom wgsl kernels based on configured inputs, body and outputs.
///
/// This type has 4 phases that must be executed in order, but no worry the type system won't allow
/// you to make mistakes.
///
///   1. [Input Phase](InputPhase)
///     This phase focuses on registering the input arrays and scalars that are going to be used by
///     the kernel.
///   2. [Body Phase](BodyPhase)
///     After the input phase is done, all the operations that happen in the body must be
///     registered.
///   3. [Output Phase](OutputPhase)
///     This step focuses on registering all output arrays or inputs that the kernel needs to write to.
///   4. [Compilation Phase](CompilationPhase)
///     Now that all other phases are completed, we can actually compile the kernel.
pub struct ElemWiseKernelCodegen<Phase = InputPhase> {
    operations: Vec<Operator>,
    input_bindings: Vec<Binding>,
    output_bindings: Vec<Binding>,
    named_bindings: Vec<(String, Binding)>,
    functions: Vec<Function>,
    vectorization: Vectorization,
    mappings_inplace: Vec<InplaceMapping>,
    workgroup_size: WorkgroupSize,
    _phase: PhantomData<Phase>,
}

pub enum Input {
    Array {
        item: Item,
        visibility: Visibility,
        strategy: ReadingStrategy,
    },
    Scalar {
        elem: Elem,
        size: usize,
    },
}

pub enum ReadingStrategy {
    /// Each element will be read in a way to be compatible with the output layout.
    OutputLayout,
    /// Keep the current layout.
    Plain,
}

#[derive(Clone)]
pub enum Output {
    Array { item: Item, local: u16 },
    Input { item: Item, input: u16, local: u16 },
}

impl Default for ElemWiseKernelCodegen<InputPhase> {
    fn default() -> Self {
        Self {
            operations: Vec::new(),
            input_bindings: Vec::new(),
            output_bindings: Vec::new(),
            named_bindings: Vec::new(),
            functions: Vec::new(),
            vectorization: Vectorization::Scalar,
            mappings_inplace: Vec::new(),
            workgroup_size: WorkgroupSize::default(),
            _phase: PhantomData,
        }
    }
}

impl ElemWiseKernelCodegen<InputPhase> {
    pub fn new() -> Self {
        Self::default()
    }

    #[allow(dead_code)]
    pub fn vectorize(mut self, vectorization: Vectorization) -> Self {
        self.vectorization = vectorization;
        self
    }

    #[allow(dead_code)]
    pub fn inplace(mut self, mappings: &[InplaceMapping]) -> Self {
        self.mappings_inplace = mappings.to_vec();
        self
    }

    /// Register the inputs used by the kernel.
    pub fn inputs(mut self, inputs: &[Input]) -> ElemWiseKernelCodegen<BodyPhase> {
        let mut index: u16 = 0;

        for input in inputs {
            match input {
                Input::Array {
                    item,
                    visibility,
                    strategy,
                } => {
                    let item = item.vectorize(self.vectorization);

                    self.input_bindings.push(Binding {
                        item: bool_item(item),
                        visibility: *visibility,
                        location: Location::Storage,
                        size: None,
                    });

                    match strategy {
                        ReadingStrategy::OutputLayout => {
                            self.operations.push(Operator::ReadGlobalWithLayout {
                                variable: Variable::Input(index, item),
                                tensor_read_pos: index as usize,
                                tensor_layout_pos: 0, // Will set the right value during the output
                                                      // phase.
                            });
                        }
                        ReadingStrategy::Plain => {
                            self.operations.push(Operator::ReadGlobal {
                                variable: Variable::Input(index, item),
                            });
                        }
                    }

                    index += 1;
                }
                Input::Scalar { elem, size } => {
                    let elem = bool_elem(*elem);

                    self.named_bindings.push((
                        format!("scalars_{}", elem),
                        Binding {
                            item: Item::Scalar(elem),
                            visibility: Visibility::Read,
                            location: Location::Storage,
                            size: Some(*size),
                        },
                    ));
                }
            }
        }

        ElemWiseKernelCodegen {
            operations: self.operations,
            input_bindings: self.input_bindings,
            output_bindings: self.output_bindings,
            named_bindings: self.named_bindings,
            functions: self.functions,
            vectorization: self.vectorization,
            mappings_inplace: self.mappings_inplace,
            workgroup_size: self.workgroup_size,
            _phase: PhantomData,
        }
    }
}

impl ElemWiseKernelCodegen<BodyPhase> {
    /// Register the [operators](Operator) that the kernel must execute in the order provided.
    pub fn body(mut self, operators: &[Operator]) -> ElemWiseKernelCodegen<OutputPhase> {
        let mut register_function = |function: Function| {
            if !self.functions.contains(&function) {
                self.functions.push(function);
            }
        };

        // Since not all operators are native to WGSL, we need to add the custom ones.
        for ops in operators.iter() {
            match ops {
                Operator::Powf {
                    lhs: _,
                    rhs,
                    out: _,
                } => match rhs {
                    Variable::Scalar(_, _) => {
                        register_function(Function::PowfScalar(
                            Item::Scalar(Elem::F32).vectorize(self.vectorization),
                        ));
                    }
                    _ => {
                        register_function(Function::Powf(
                            Item::Scalar(Elem::F32).vectorize(self.vectorization),
                        ));
                    }
                },
                Operator::Erf { input: _, out: _ } => {
                    register_function(Function::Erf(
                        Item::Scalar(Elem::F32).vectorize(self.vectorization),
                    ));
                }
                #[cfg(target_os = "macos")]
                Operator::Tanh { input: _, out: _ } => register_function(Function::SafeTanh(
                    Item::Scalar(Elem::F32).vectorize(self.vectorization),
                )),
                _ => {}
            }
            self.operations.push(ops.vectorize(self.vectorization));
        }

        ElemWiseKernelCodegen {
            operations: self.operations,
            input_bindings: self.input_bindings,
            output_bindings: self.output_bindings,
            named_bindings: self.named_bindings,
            vectorization: self.vectorization,
            functions: self.functions,
            mappings_inplace: self.mappings_inplace,
            workgroup_size: self.workgroup_size,
            _phase: PhantomData,
        }
    }
}

impl ElemWiseKernelCodegen<OutputPhase> {
    /// Register the outputs with their local variable index.
    ///
    /// Note that the index corresponds to the registered [operator](Operator) number at the
    /// [body phase](BodyPhase).
    /// So the 4th operator registered creates the local variable 3 (N-1, since the 1th index is 0).
    pub fn outputs(mut self, outputs: &[Output]) -> ElemWiseKernelCodegen<CompilationPhase> {
        let mut index = 0;
        let mut position_out = 0;

        let mut outputs = outputs.to_vec();

        for mapping in self.mappings_inplace.iter() {
            match outputs.get_mut(mapping.position_output) {
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

        for array in &outputs {
            match array {
                Output::Array { item, local } => {
                    let item = item.vectorize(self.vectorization);
                    let elem_adapted = bool_item(item);

                    self.output_bindings.push(Binding {
                        item: elem_adapted,
                        visibility: Visibility::ReadWrite,
                        location: Location::Storage,
                        size: None,
                    });
                    self.operations.push(Operator::AssignGlobal {
                        input: Variable::Local(*local, item),
                        out: Variable::Output(index, elem_adapted),
                    });
                    index += 1;

                    if index == 1 {
                        position_out = self.input_bindings.len(); // First output when we have a
                                                                  // new array for the output.
                    }
                }
                Output::Input { item, input, local } => {
                    let item = item.vectorize(self.vectorization);

                    self.operations.push(Operator::AssignGlobal {
                        input: Variable::Local(*local, item),
                        out: Variable::Input(*input, bool_item(item)),
                    });
                    position_out = *input as usize; // Input number when we use inplace operation.
                }
            }
        }

        // We set the output number that will be used for the stride definition.
        for i in 0..self.input_bindings.len() {
            if let Some(Operator::ReadGlobalWithLayout {
                variable: _,
                tensor_read_pos: _,
                tensor_layout_pos,
            }) = self.operations.get_mut(i)
            {
                {
                    *tensor_layout_pos = position_out;
                }
            };
        }

        ElemWiseKernelCodegen {
            operations: self.operations,
            input_bindings: self.input_bindings,
            output_bindings: self.output_bindings,
            named_bindings: self.named_bindings,
            functions: self.functions,
            vectorization: self.vectorization,
            mappings_inplace: self.mappings_inplace,
            workgroup_size: self.workgroup_size,
            _phase: PhantomData,
        }
    }
}

impl ElemWiseKernelCodegen<CompilationPhase> {
    pub fn workgroup_size(mut self, workgroup_size: WorkgroupSize) -> Self {
        self.workgroup_size = workgroup_size;
        self
    }

    /// Compile the kernel into a [compute shader](ComputeShader).
    pub fn compile(self) -> ComputeShader {
        let inputs = self.input_bindings;
        let outputs = self.output_bindings;
        let mut named = Vec::with_capacity(2);

        named.push((
            "info".to_string(),
            Binding {
                item: Item::Scalar(Elem::U32),
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
            workgroup_size: self.workgroup_size,
            body: Body::new(self.operations),
            num_workgroups: true,
            global_invocation_id: true,
            functions: self.functions,
        }
    }
}

#[derive(new)]
pub struct StaticHandle<'a> {
    handle: &'a WgpuHandle,
    strides: &'a [usize],
    shape: &'a [usize],
}

/// The position of the input or output to calculate the number of workgroups to launch.
pub enum WorkgroupLaunch {
    Input { pos: usize },
    Output { pos: usize },
}

/// Execute a static kernel.
///
///
/// The limitation from this method is that you can't launch a kernel with multiple types of
/// scalar.
pub fn execute_static<K, E: WgpuElement>(
    inputs: &[StaticHandle],
    outputs: &[StaticHandle],
    scalar_elems: Option<&[E]>,
    launch: WorkgroupLaunch,
    client: WgpuComputeClient,
) where
    K: StaticKernelSource + 'static,
{
    let mut info = Vec::new();
    let mut handles = Vec::with_capacity(inputs.len() + outputs.len() + 2);

    // Inner function to fill the info buffer.
    let mut register_info_tensor = |strides: &[usize], shape: &[usize]| {
        if info.is_empty() {
            info.push(strides.len() as u32);
        }

        for s in strides.iter() {
            info.push(*s as u32);
        }
        for s in shape.iter() {
            info.push(*s as u32);
        }
    };

    let mut num_elems_output = 0;

    // We start by registering the inputs.
    for (i, input) in inputs.iter().enumerate() {
        if let WorkgroupLaunch::Input { pos } = &launch {
            if i == *pos {
                num_elems_output = calculate_num_elems_dyn_rank(input.shape);
            }
        };
        register_info_tensor(input.strides, input.shape);
        handles.push(input.handle);
    }

    // Then we follow with the outputs.
    for (i, output) in outputs.iter().enumerate() {
        if let WorkgroupLaunch::Output { pos } = &launch {
            if i == *pos {
                num_elems_output = calculate_num_elems_dyn_rank(output.shape);
            }
        };
        register_info_tensor(output.strides, output.shape);
        handles.push(output.handle);
    }

    let info = &client.create(bytemuck::cast_slice(&info));
    handles.push(info);

    // Finally we finish with the named bindings.
    let mut scalars = None;
    if let Some(values) = &scalar_elems {
        scalars = Some(client.create(bytemuck::cast_slice(values)));
    }

    if let Some(scalars) = scalars.as_ref() {
        handles.push(scalars);
    }

    let workgroup = elemwise_workgroup(num_elems_output, WORKGROUP_DEFAULT);
    let kernel = Box::new(StaticKernel::<K>::new(workgroup));

    client.execute(kernel, &handles);
}

pub(crate) fn calculate_num_elems_dyn_rank(shape: &[usize]) -> usize {
    let mut num_elems = 1;
    for i in shape.iter() {
        num_elems *= i;
    }
    num_elems
}

impl Item {
    fn vectorize(&self, vectorize: Vectorization) -> Item {
        match vectorize {
            Vectorization::Vec4 => Item::Vec4(self.elem()),
            Vectorization::Vec3 => Item::Vec3(self.elem()),
            Vectorization::Vec2 => Item::Vec2(self.elem()),
            Vectorization::Scalar => Item::Scalar(self.elem()),
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
        Elem::Bool => Elem::U32,
        _ => elem,
    }
}
