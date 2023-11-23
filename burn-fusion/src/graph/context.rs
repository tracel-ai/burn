use super::{
    BaseOpsDescription, ReshapeDescription, SliceOpsDescription, SwapDimsDescription,
    TensorOpsDescription,
};
use crate::{FusionBackend, HandleContainer, TensorDescription, TensorId};
use std::collections::HashMap;

#[derive(new)]
pub struct Context<'a, 'b, B: FusionBackend> {
    pub tensors: &'a HashMap<TensorId, TensorDescription>,
    pub handles: &'b mut HandleContainer<B>,
}

#[derive(Default)]
pub(crate) struct LocalGraphConverter {
    tensors_local2global: HashMap<TensorId, TensorDescription>,
    tensors_global2local: HashMap<TensorId, TensorDescription>,
    /// Only useful to create new shape ID.
    /// You should use tensor descriptions to retrieve the proper shape.
    shapes_local2global: HashMap<usize, usize>,
}
impl LocalGraphConverter {
    pub(crate) fn context<'a, 'b, B: FusionBackend>(
        &'a self,
        handles: &'b mut HandleContainer<B>,
    ) -> Context<'a, 'b, B> {
        Context {
            handles,
            tensors: &self.tensors_local2global,
        }
    }
    pub(crate) fn clear(&mut self) {
        self.tensors_local2global.clear();
        self.tensors_global2local.clear();
        self.shapes_local2global.clear();
    }
}

impl TensorOpsDescription {
    pub(crate) fn to_local(&self, converter: &mut LocalGraphConverter) -> Self {
        match self {
            TensorOpsDescription::BaseOpsFloat(ops) => {
                TensorOpsDescription::BaseOpsFloat(ops.to_local(converter))
            }
            TensorOpsDescription::BaseOpsInt(ops) => {
                TensorOpsDescription::BaseOpsInt(ops.to_local(converter))
            }
            TensorOpsDescription::BaseOpsBool(ops) => {
                TensorOpsDescription::BaseOpsBool(ops.to_local(converter))
            }
            TensorOpsDescription::NumericOpsFloat(_) => todo!(),
            TensorOpsDescription::NumericOpsInt(_) => todo!(),
            TensorOpsDescription::BoolOps(_) => todo!(),
            TensorOpsDescription::IntOps(_) => todo!(),
            TensorOpsDescription::FloatOps(_) => todo!(),
            TensorOpsDescription::ModuleOps(_) => todo!(),
        }
    }
}

impl BaseOpsDescription {
    pub(crate) fn to_local(&self, converter: &mut LocalGraphConverter) -> Self {
        match self {
            BaseOpsDescription::ToDevice(desc) => {
                BaseOpsDescription::ToDevice(desc.to_local(converter))
            }
            BaseOpsDescription::Reshape(desc) => BaseOpsDescription::Reshape(ReshapeDescription {
                input: desc.input.to_local(converter),
                out: desc.out.to_local(converter),
            }),
            BaseOpsDescription::SwapDims(desc) => {
                BaseOpsDescription::SwapDims(SwapDimsDescription {
                    input: desc.input.to_local(converter),
                    out: desc.out.to_local(converter),
                    dim1: desc.dim1,
                    dim2: desc.dim2,
                })
            }
            BaseOpsDescription::Slice(desc) => BaseOpsDescription::Slice(SliceOpsDescription {
                tensor: desc.tensor.to_local(converter),
                ranges: desc.ranges.clone(),
                out: desc.out.to_local(converter),
            }),
            BaseOpsDescription::SliceAssign(desc) => {
                BaseOpsDescription::SliceAssign(super::SliceAssignOpsDescription {
                    tensor: desc.tensor.to_local(converter),
                    ranges: desc.ranges.clone(),
                    value: desc.value.to_local(converter),
                    out: desc.out.to_local(converter),
                })
            }
            BaseOpsDescription::Equal(desc) => {
                BaseOpsDescription::Equal(super::BinaryOpsDescription {
                    lhs: desc.lhs.to_local(converter),
                    rhs: desc.rhs.to_local(converter),
                    out: desc.out.to_local(converter),
                })
            }
            BaseOpsDescription::Repeat(desc) => {
                BaseOpsDescription::Repeat(super::RepeatOpsDescription {
                    tensor: desc.tensor.to_local(converter),
                    dim: desc.dim,
                    times: desc.times,
                    out: desc.out.to_local(converter),
                })
            }
            BaseOpsDescription::Cat(desc) => BaseOpsDescription::Cat(super::CatOpsDescription {
                tensors: desc
                    .tensors
                    .iter()
                    .map(|tensor| tensor.to_local(converter))
                    .collect(),
                dim: desc.dim,
                out: desc.out.to_local(converter),
            }),
        }
    }
}

impl TensorDescription {
    pub(crate) fn to_local(&self, converter: &mut LocalGraphConverter) -> Self {
        if let Some(value) = converter.tensors_global2local.get(&self.id) {
            return value.clone();
        }

        let local_id = TensorId::new(converter.tensors_local2global.len() as u64);
        let mut local_shape = Vec::with_capacity(self.shape.len());

        for dim in self.shape.iter() {
            if let Some(dim) = converter.shapes_local2global.get(dim) {
                local_shape.push(*dim);
            } else {
                let dim_new = converter.shapes_local2global.len();
                local_shape.push(dim_new);
                converter.shapes_local2global.insert(*dim, dim_new);
            }
        }

        let local_tensor = TensorDescription {
            id: local_id.clone(),
            shape: local_shape,
            status: self.status.clone(),
        };

        converter
            .tensors_local2global
            .insert(local_id, self.clone());
        converter
            .tensors_global2local
            .insert(self.id.clone(), local_tensor.clone());

        local_tensor
    }
}

#[cfg(test)]
mod tests {
    use crate::TensorStatus;

    use super::*;

    #[test]
    fn tensor_description_to_local() {
        let tensor1 = TensorDescription {
            id: TensorId::new(500),
            shape: vec![512, 32, 2048],
            status: TensorStatus::ReadOnly,
        };
        let tensor2 = TensorDescription {
            id: TensorId::new(501),
            shape: vec![512, 128, 2048],
            status: TensorStatus::ReadOnly,
        };
        let mut converter = LocalGraphConverter::default();
        let tensor1_local = tensor1.to_local(&mut converter);
        let tensor2_local = tensor2.to_local(&mut converter);

        assert_eq!(
            tensor1_local,
            TensorDescription {
                id: TensorId::new(0),
                shape: vec![0, 1, 2],
                status: TensorStatus::ReadOnly
            }
        );
        assert_eq!(
            tensor2_local,
            TensorDescription {
                id: TensorId::new(1),
                shape: vec![0, 3, 2],
                status: TensorStatus::ReadOnly
            }
        );
    }
}
