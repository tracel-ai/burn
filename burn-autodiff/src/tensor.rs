use std::marker::PhantomData;

use burn_tensor::backend::Backend;

use crate::{
    grads::Gradients,
    graph::{
        ops::{Backward, Graph, Metadata, MetadataRef, OpsID},
        Requirement,
    },
    ADBackendDecorator,
};

use burn_tensor::ops::*;

#[derive(Debug, Clone)]
pub struct ADTensor<B: Backend, const D: usize> {
    pub primitive: B::TensorPrimitive<D>,
    pub metadata: MetadataRef,
    pub(crate) graph: Graph<B>,
}

#[derive(new, Debug, Clone)]
pub struct BackwardTensor<B: Backend, const D: usize> {
    pub primitive: B::TensorPrimitive<D>,
    pub metadata: MetadataRef,
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
    metadata: MetadataRef,
    phantom: PhantomData<B>,
}

impl<B: Backend> Backward<B> for NewTensor<B> {
    fn backward(self: Box<Self>, _grads: &mut Gradients<B>) {}

    fn metadata(&self) -> MetadataRef {
        self.metadata.clone()
    }
}

impl<B: Backend, const D: usize> ADTensor<B, D> {
    pub fn new(primitive: B::TensorPrimitive<D>) -> Self {
        let id = OpsID::new();
        let metadata = Metadata::new(vec![], 0, id, Requirement::None);
        let tensor = Self {
            primitive,
            metadata: metadata.into(),
            graph: Graph::new(),
        };
        tensor.require_grad()
    }

    pub fn require_grad(mut self) -> Self {
        match self.metadata.requirement {
            Requirement::Grad => self,
            Requirement::GradInBackward => {
                panic!("Can't require grad to a non leaf tensor")
            }
            Requirement::None => {
                let metadata =
                    Metadata::new(vec![], 0, self.metadata.id.clone(), Requirement::Grad);
                self.metadata = metadata.into();
                let ops = NewTensor::new(self.metadata.clone());
                self.register_ops(ops)
            }
        }
    }

    pub fn from_ops<const N: usize>(
        nodes: &[MetadataRef; N],
        output: B::TensorPrimitive<D>,
        graphs: [Graph<B>; N],
        requirement: Requirement,
    ) -> Self {
        let graph = graphs
            .into_iter()
            .reduce(|acc, graph| graph.merge(acc))
            .unwrap_or(Graph::new());

        let order = nodes
            .iter()
            .map(|metadata| metadata.order)
            .reduce(|acc, order| usize::max(acc, order))
            .unwrap_or(0)
            + 1;

        let metadata = Metadata::new(
            nodes.iter().map(|metadata| metadata.id.clone()).collect(),
            order,
            OpsID::new(),
            requirement,
        );

        Self {
            primitive: output,
            metadata: metadata.into(),
            graph,
        }
    }

    pub fn to_backward(&self) -> BackwardTensor<B, D> {
        BackwardTensor::new(self.primitive.clone(), self.metadata.clone())
    }

    pub fn register_ops<O: Backward<B> + 'static>(mut self, ops: O) -> Self {
        self.graph = self.graph.register(&self.metadata.id, Box::new(ops));
        self
    }
}
