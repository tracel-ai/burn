use crate::{
    compute::DynamicKernel,
    kernel::{elemwise_workgroup, DynamicKernelSource, SourceTemplate, WORKGROUP_DEFAULT},
    FloatElement, GraphicsApi, IntElement, Wgpu,
};
use burn_fusion::{
    graph::{NumericOpsDescription, TensorOpsDescription},
    FusionBackend, FusionOps, FusionProperties, FusionStatus, HandleContainer, TensorDescription,
    TensorId,
};
use burn_tensor::Element;
use core::fmt::Display;
use core::hash::Hash;
use hashbrown::{HashMap, HashSet};
use std::{collections::hash_map::DefaultHasher, hash::Hasher, sync::Arc};

use super::{Binding, WgslTempate};

pub struct FloatElementWiseFusionOps {
    inputs: Vec<TensorDescription>,
    temps: HashMap<TensorId, u16>,
    tensors: HashMap<TensorId, TensorDescription>,
    ops: Vec<Operator>,
    properties: FusionProperties,
}

impl Default for FloatElementWiseFusionOps {
    fn default() -> Self {
        Self {
            inputs: Vec::new(),
            temps: HashMap::new(),
            tensors: HashMap::new(),
            ops: Vec::new(),
            properties: FusionProperties::default(),
        }
    }
}

impl<G: GraphicsApi + 'static, F: FloatElement, I: IntElement> FusionOps<Wgpu<G, F, I>>
    for FloatElementWiseFusionOps
{
    fn register(&mut self, ops: Arc<TensorOpsDescription<Wgpu<G, F, I>>>) -> FusionStatus {
        match ops.as_ref() {
            TensorOpsDescription::NumericOpsFloat(ops) => {
                if !self.register_numeric(ops) {
                    return FusionStatus::Closed(self.properties.clone());
                }
            }
            _ => return FusionStatus::Closed(self.properties.clone()),
        };

        self.properties.score += 1;
        self.properties.ready = self.ops.len() > 1;

        self.tensors
            .values()
            .for_each(|tensor| match tensor.status {
                burn_fusion::TensorStatus::ReadWrite => {}
                _ => self.properties.ready = false,
            });

        return FusionStatus::Open(self.properties.clone());
    }

    fn execute(&mut self, handles: &mut HandleContainer<Wgpu<G, F, I>>) {
        // BUILD THE OUTPUTS
        let mut outputs = Vec::new();

        if let Some(ops) = self.ops.last() {
            match ops {
                Operator::Add { lhs, rhs, out } => match out {
                    Ident::Input(index) => {
                        let output = self.inputs.get(*index as usize).unwrap();
                        outputs.push(output);
                    }
                    Ident::Temp(index) => {
                        let output = self.temps.iter().find(|(id, i)| index == *i).unwrap();
                        let output = self.tensors.get(output.0).unwrap();
                        outputs.push(output);
                    }
                    Ident::Output(_) => todo!(),
                },
                Operator::Sub { lhs, rhs, out } => todo!(),
                Operator::Assign { input, out } => todo!(),
            }
        }

        for tensor in self.tensors.values() {
            match tensor.status {
                burn_fusion::TensorStatus::ReadOnly => {
                    outputs.push(tensor);
                }
                _ => {}
            }
        }

        for (i, output) in outputs.iter().enumerate() {
            let input = self.temps.get(&output.id).unwrap();

            self.ops.push(Operator::Assign {
                input: Ident::Temp(*input),
                out: Ident::Output(i as u16),
            });
        }

        // BUILD THE BINDINGS, INFO and HANDLES.
        let mut bindings = Vec::new();
        let mut info = Vec::new();
        let mut kernel_handles = Vec::new();
        let mut client = None;

        // REGISTER INPUTS
        for input in self.inputs.iter() {
            if info.is_empty() {
                info.push(input.shape.len()); // Rank
            }

            let mut handle = handles.get_handle(&input);
            info.append(&mut handle.strides);
            info.append(&mut input.shape.clone());
            kernel_handles.push(handle.handle);

            if let None = client {
                client = Some(handle.client.clone());
            }

            bindings.push(Binding {
                elem: super::Elem::F32,
                visibility: super::Visibility::Read,
                location: super::Location::Storage,
                size: None,
            });
        }

        let mut client = client.unwrap();

        // REGISTER OUTPUTS
        let mut num_elems = 1;
        for output in outputs {
            if num_elems == 1 {
                for i in output.shape.iter() {
                    num_elems *= i;
                }
            }
            let handle = client.empty(num_elems * core::mem::size_of::<f32>());
            // TODO: handles.register_handle(handle);
            let strides = dyn_strides(&output.shape);

            info.append(&mut strides);
            info.append(&mut output.shape.clone());
            kernel_handles.push(handle);

            bindings.push(Binding {
                elem: super::Elem::F32,
                visibility: super::Visibility::ReadWrite,
                location: super::Location::Storage,
                size: None,
            });
        }

        // REGISTER INFO
        bindings.push(Binding {
            elem: super::Elem::U32,
            visibility: super::Visibility::Read,
            location: super::Location::Storage,
            size: None,
        });

        let kernel = WgslTempate {
            bindings,
            workgroup_sizes: super::WorkgroupSize::default(),
            body: Box::new(ElemWiseBody {
                operations: self.ops.clone(),
            }),
            num_workgroups: true,
            global_invocation_id: true,
        };

        let workgroup = elemwise_workgroup(num_elems, WORKGROUP_DEFAULT);

        let handles = kernel_handles.iter().collect::<Vec<_>>();

        client
            .unwrap()
            .execute(Box::new(DynamicKernel::new(kernel, workgroup)), &handles);
    }

    fn reset(&mut self) {
        self.properties = FusionProperties::default();
        self.tensors.clear();
        self.ops.clear();
    }

    fn len(&self) -> usize {
        self.ops.len()
    }
}

impl FloatElementWiseFusionOps {
    fn input_to_ident(&mut self, tensor: &TensorDescription) -> Ident {
        let ident = match self.tensors.contains_key(&tensor.id) {
            false => {
                let ident = Ident::Input(self.inputs.len() as u16);
                self.inputs.push(tensor.clone());
                ident
            }
            true => match self.temps.get(&tensor.id) {
                Some(index) => Ident::Temp(*index),
                None => {
                    let input = self
                        .inputs
                        .iter()
                        .enumerate()
                        .find(|(_, input)| input.id == tensor.id)
                        .unwrap();
                    Ident::Input(input.0 as u16)
                }
            },
        };
        self.tensors.insert(tensor.id.clone(), tensor.clone());

        ident
    }

    fn output_to_ident(&mut self, tensor: &TensorDescription) -> Ident {
        let temp = self.ops.len() as u16;
        self.temps.insert(tensor.id.clone(), temp);
        self.tensors.insert(tensor.id.clone(), tensor.clone());
        Ident::Temp(temp)
    }

    fn register_numeric<B: FusionBackend, E: Element>(
        &mut self,
        ops: &NumericOpsDescription<B, E>,
    ) -> bool {
        match ops {
            NumericOpsDescription::Add(desc, _) => {
                let lhs = self.input_to_ident(&desc.lhs);
                let rhs = self.input_to_ident(&desc.rhs);
                let out = self.output_to_ident(&desc.out);

                self.ops.push(Operator::Add { lhs, rhs, out });

                return true;
            }
            _ => false,
        }
    }
}

#[derive(Hash, Clone)]
enum Ident {
    Input(u16),
    Temp(u16),
    Output(u16),
}

#[derive(Hash, Clone)]
enum Operator {
    Add { lhs: Ident, rhs: Ident, out: Ident },
    Sub { lhs: Ident, rhs: Ident, out: Ident },
    Assign { input: Ident, out: Ident },
}

impl Display for Ident {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Ident::Input(number) => f.write_str(format!("input_{number}").as_str()),
            Ident::Temp(number) => f.write_str(format!("temp_{number}").as_str()),
            Ident::Output(number) => {
                f.write_str(format!("output_{number}_global[index_{number}]").as_str())
            }
        }
    }
}

impl Display for Operator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Operator::Add { lhs, rhs, out } => f.write_str(&format!("let {out} = {lhs} + {rhs};")),
            Operator::Sub { lhs, rhs, out } => f.write_str(&format!("let {out} = {lhs} - {rhs};")),
            Operator::Assign { input, out } => f.write_str(&format!("let {out} = {input};")),
        }
    }
}

#[derive(Hash)]
pub struct ElemWiseBody {
    operations: Vec<Operator>,
}

impl Display for ElemWiseBody {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for ops in self.operations.iter() {
            f.write_str(ops.to_string().as_str())?;
            f.write_str("\n")?;
        }

        Ok(())
    }
}

impl DynamicKernelSource for ElemWiseBody {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(self.to_string())
    }

    fn id(&self) -> String {
        let mut s = DefaultHasher::new();
        self.hash(&mut s);

        s.finish().to_string()
    }
}
fn dyn_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![0; shape.len()];

    let mut current = 1;
    shape.iter().enumerate().rev().for_each(|(index, val)| {
        strides[index] = current;
        current *= val;
    });

    strides
}
