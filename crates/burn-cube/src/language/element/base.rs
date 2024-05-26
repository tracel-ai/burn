use crate::{
    dialect::{ComputeShader, Elem, Item, Variable},
    Compilation, CompilationInfo, CompilationSettings, CubeContext, InputInfo, KernelLauncher,
    OutputInfo, Runtime,
};
use alloc::rc::Rc;
use std::collections::HashMap;

/// Types used in a cube function must implement this trait
///
/// Variables whose values will be known at runtime must
/// have ExpandElement as associated type
/// Variables whose values will be known at compile time
/// must have the primitive type as associated type
///
/// Note: Cube functions should be written using CubeTypes,
/// so that the code generated uses the associated ExpandType.
/// This allows Cube code to not necessitate cloning, which is cumbersome
/// in algorithmic code. The necessary cloning will automatically appear in
/// the generated code.
pub trait CubeType {
    type ExpandType: Clone;
}

pub struct KernelBuilder {
    pub context: CubeContext,
    inputs: Vec<InputInfo>,
    outputs: Vec<OutputInfo>,
    settings: CompilationSettings,
    indices: HashMap<Elem, usize>,
    num_input: u16,
    num_output: u16,
}

impl Default for KernelBuilder {
    fn default() -> Self {
        Self {
            context: CubeContext::root(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            settings: CompilationSettings::default(),
            indices: HashMap::new(),
            num_input: 0,
            num_output: 0,
        }
    }
}

impl KernelBuilder {
    pub fn scalar(&mut self, elem: Elem) -> ExpandElement {
        let index = match self.indices.get_mut(&elem) {
            Some(index) => match self.inputs.get_mut(*index).unwrap() {
                InputInfo::Scalar { elem: _, size } => {
                    *size = *size + 1;
                    *size as u16 - 1
                }
                _ => panic!("Should be a scalar."),
            },
            None => {
                self.indices.insert(elem, self.inputs.len());
                self.inputs.push(InputInfo::Scalar { size: 1, elem });
                0
            }
        };

        self.context.scalar(index, elem)
    }

    pub fn output_array(&mut self, item: Item) -> ExpandElement {
        self.outputs.push(OutputInfo::Array { item });
        let variable = self.context.output(self.num_output, item);
        self.num_output += 1;

        variable
    }

    pub fn input_array(&mut self, item: Item) -> ExpandElement {
        self.inputs.push(InputInfo::Array {
            item,
            visibility: crate::dialect::Visibility::Read,
        });
        let variable = self.context.input(self.num_input, item);
        self.num_input += 1;
        variable
    }

    pub fn compile(self) -> ComputeShader {
        let info = CompilationInfo {
            scope: self.context.into_scope(),
            inputs: self.inputs,
            outputs: self.outputs,
        };

        Compilation::new(info).compile(self.settings)
    }
}

pub trait ArgSettings<R: Runtime>: Send + Sync {
    fn register(&self, launcher: &mut KernelLauncher<R>);
}

pub trait LaunchArg {
    type RuntimeArg<'a, R: Runtime>: ArgSettings<R>;

    fn compile_input(builder: &mut KernelBuilder) -> ExpandElement;
    fn compile_output(builder: &mut KernelBuilder) -> ExpandElement;
}

/// Reference to a JIT variable
#[derive(Clone, Debug)]
pub enum ExpandElement {
    /// Variable kept in the variable pool.
    Managed(Rc<Variable>),
    /// Variable not kept in the variable pool.
    Plain(Variable),
}

impl ExpandElement {
    pub fn can_mut(&self) -> bool {
        match self {
            ExpandElement::Managed(var) => {
                if let Variable::Local(_, _, _) = var.as_ref() {
                    Rc::strong_count(var) <= 2
                } else {
                    false
                }
            }
            ExpandElement::Plain(_) => false,
        }
    }
}

impl core::ops::Deref for ExpandElement {
    type Target = Variable;

    fn deref(&self) -> &Self::Target {
        match self {
            ExpandElement::Managed(var) => var.as_ref(),
            ExpandElement::Plain(var) => var,
        }
    }
}

impl From<ExpandElement> for Variable {
    fn from(value: ExpandElement) -> Self {
        match value {
            ExpandElement::Managed(var) => *var,
            ExpandElement::Plain(var) => var,
        }
    }
}
