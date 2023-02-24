use std::marker::PhantomData;

use burn_tensor::backend::Backend;

use crate::{
    grads::Gradients,
    graph::ops::{Ops, OpsID, OpsMap, OpsMetadata, OpsMetadataRef, Requirement},
    ADBackendDecorator,
};

use burn_tensor::ops::*;

#[derive(Debug, Clone)]
pub struct ADTensor<B: Backend, const D: usize> {
    pub primitive: B::TensorPrimitive<D>,
    pub metadata: OpsMetadataRef,
    pub(crate) map: OpsMap<B>,
}

pub type Elem<B> = <ADBackendDecorator<B> as Backend>::Elem;
pub type BoolTensor<B, const D: usize> = <ADBackendDecorator<B> as Backend>::BoolTensorPrimitive<D>;
pub type IntTensor<B, const D: usize> =
    <<ADBackendDecorator<B> as Backend>::IntegerBackend as Backend>::TensorPrimitive<D>;

impl<const D: usize, B> core::ops::Add<Self> for ADTensor<B, D>
where
    B: Backend,
{
    type Output = Self;

    fn add(self, other: Self) -> Self {
        ADBackendDecorator::add(self, other)
    }
}

impl<B: Backend, const D: usize> Zeros for ADTensor<B, D> {
    fn zeros(&self) -> Self {
        todo!()
    }
}

impl<B: Backend, const D: usize> Ones for ADTensor<B, D> {
    fn ones(&self) -> Self {
        todo!()
    }
}

#[derive(new, Debug)]
struct NewTensor<B: Backend> {
    metadata: OpsMetadataRef,
    phantom: PhantomData<B>,
}

impl<B: Backend> Ops<B> for NewTensor<B> {
    fn backward(self: Box<Self>, _grads: &mut Gradients<B>) {}

    fn metadata(&self) -> OpsMetadataRef {
        self.metadata.clone()
    }
}

impl<B: Backend, const D: usize> ADTensor<B, D> {
    pub fn new(primitive: B::TensorPrimitive<D>) -> Self {
        let id = OpsID::new();
        let metadata = OpsMetadata::new(vec![], 0, id.clone(), Requirement::Grad);
        let tensor = Self {
            primitive,
            metadata: metadata.into(),
            map: OpsMap::new(),
        };
        let ops = NewTensor::new(tensor.metadata.clone());

        tensor.register_ops(ops)
    }
    pub fn from_binary_ops<const DLHS: usize, const DRHS: usize>(
        lhs: ADTensor<B, DLHS>,
        rhs: ADTensor<B, DRHS>,
        output: B::TensorPrimitive<D>,
    ) -> Self {
        let order = usize::max(lhs.metadata.order, rhs.metadata.order) + 1;
        let id = OpsID::new();
        let map = lhs.map.merge(&rhs.map);
        let parents = vec![lhs.metadata.id.clone(), rhs.metadata.id.clone()];
        let metadata = OpsMetadata::new(
            parents,
            order,
            id,
            lhs.metadata.infer_requirement(&rhs.metadata),
        );

        Self {
            primitive: output,
            metadata: metadata.into(),
            map,
        }
    }
    pub fn from_unary_ops<const DLHS: usize>(
        lhs: ADTensor<B, DLHS>,
        output: B::TensorPrimitive<D>,
    ) -> Self {
        let order = lhs.metadata.order + 1;
        let id = OpsID::new();
        let parents = vec![lhs.metadata.id.clone()];
        let metadata = OpsMetadata::new(
            parents,
            order,
            id,
            lhs.metadata.requirement.infer(&Requirement::None),
        );

        Self {
            primitive: output,
            metadata: metadata.into(),
            map: lhs.map,
        }
    }
    pub fn register_ops<O: Ops<B> + 'static>(self, ops: O) -> Self {
        self.map.register(&self.metadata.id, Box::new(ops));
        self
    }
}
