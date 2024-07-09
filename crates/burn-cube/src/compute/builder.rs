use crate::ir::{Elem, Item, Visibility};
use crate::prelude::KernelDefinition;
use crate::KernelSettings;
use crate::{
    frontend::{CubeContext, ExpandElement},
    InputInfo, KernelExpansion, KernelIntegrator, OutputInfo,
};
use std::collections::HashMap;

/// Prepare a kernel to create a [kernel definition](crate::KernelDefinition).
pub struct KernelBuilder {
    /// Cube [context](CubeContext).
    pub context: CubeContext,
    inputs: Vec<InputInfo>,
    outputs: Vec<OutputInfo>,
    indices: HashMap<Elem, usize>,
    num_input: u16,
    num_output: u16,
}

impl KernelBuilder {
    /// Register a scalar and return the [element](ExpandElement) to be used for kernel expansion.
    pub fn scalar(&mut self, elem: Elem) -> ExpandElement {
        let index = match self.indices.get_mut(&elem) {
            Some(index) => match self.inputs.get_mut(*index).unwrap() {
                InputInfo::Scalar { elem: _, size } => {
                    *size += 1;
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

    /// Register an output array and return the [element](ExpandElement) to be used for kernel expansion.
    pub fn output_tensor(&mut self, item: Item) -> ExpandElement {
        self.outputs.push(OutputInfo::Array { item });
        let variable = self.context.output(self.num_output, item);
        self.num_output += 1;

        variable
    }

    /// Register an input array and return the [element](ExpandElement) to be used for kernel expansion.
    pub fn input_tensor(&mut self, item: Item) -> ExpandElement {
        self.inputs.push(InputInfo::Array {
            item,
            visibility: Visibility::Read,
        });
        let variable = self.context.input(self.num_input, item);
        self.num_input += 1;
        variable
    }

    /// Register an output array and return the [element](ExpandElement) to be used for kernel expansion.
    pub fn output_array(&mut self, item: Item) -> ExpandElement {
        self.outputs.push(OutputInfo::Array { item });
        let variable = self.context.output(self.num_output, item);
        self.num_output += 1;

        variable
    }

    /// Register an input array and return the [element](ExpandElement) to be used for kernel expansion.
    pub fn input_array(&mut self, item: Item) -> ExpandElement {
        self.inputs.push(InputInfo::Array {
            item,
            visibility: Visibility::Read,
        });
        let variable = self.context.input(self.num_input, item);
        self.num_input += 1;
        variable
    }

    /// Build the [kernel definition](KernelDefinition).
    pub fn build(self, settings: KernelSettings) -> KernelDefinition {
        KernelIntegrator::new(KernelExpansion {
            scope: self.context.into_scope(),
            inputs: self.inputs,
            outputs: self.outputs,
        })
        .integrate(settings)
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
