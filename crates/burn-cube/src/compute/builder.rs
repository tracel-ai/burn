use crate::dialect::{Elem, Item};
use crate::{Compilation, CompilationInfo, CubeContext, ExpandElement, InputInfo, OutputInfo};
use std::collections::HashMap;

/// Prepare a kernel for [compilation](crate::Compilation).
pub struct KernelBuilder {
    pub context: CubeContext,
    inputs: Vec<InputInfo>,
    outputs: Vec<OutputInfo>,
    indices: HashMap<Elem, usize>,
    num_input: u16,
    num_output: u16,
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

    pub fn build(self) -> Compilation {
        let info = CompilationInfo {
            scope: self.context.into_scope(),
            inputs: self.inputs,
            outputs: self.outputs,
        };

        Compilation::new(info)
    }
}

impl Default for KernelBuilder {
    fn default() -> Self {
        Self {
            context: CubeContext::root(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            indices: HashMap::new(),
            num_input: 0,
            num_output: 0,
        }
    }
}
