use std::marker::PhantomData;

use burn_tensor::backend::Backend;

use crate::{
    grads::Gradients,
    graph::ops::{Backward, Graph, Metadata, MetadataRef, OpsID, Requirement},
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
        ops: &[MetadataRef; N],
        output: B::TensorPrimitive<D>,
        graphs: [Graph<B>; N],
    ) -> Self {
        let id = OpsID::new();
        let mut graph = graphs.get(0).unwrap().clone();
        for i in 1..N {
            graph = graph.merge(&graphs[i]);
        }

        let parents = ops.iter().map(|metadata| metadata.id.clone()).collect();
        let mut order = 0;
        let mut requirement = Requirement::None;

        ops.iter().for_each(|metadata| {
            requirement = metadata.requirement.infer(&requirement);
            order = usize::max(metadata.order, order);
        });
        order += 1;

        let metadata = Metadata::new(parents, order, id, requirement);

        Self {
            primitive: output,
            metadata: metadata.into(),
            graph,
        }
    }

    pub fn from_binary_ops(
        lhs: MetadataRef,
        rhs: MetadataRef,
        output: B::TensorPrimitive<D>,
        lhs_graph: Graph<B>,
        rhs_graph: Graph<B>,
    ) -> Self {
        let order = usize::max(lhs.order, rhs.order) + 1;
        let id = OpsID::new();
        let map = lhs_graph.merge(&rhs_graph);
        let parents = vec![lhs.id.clone(), rhs.id.clone()];
        let metadata = Metadata::new(parents, order, id, lhs.infer_requirement(&rhs));

        Self {
            primitive: output,
            metadata: metadata.into(),
            graph: map,
        }
    }
    pub fn from_unary_ops(
        tensor: MetadataRef,
        output: B::TensorPrimitive<D>,
        graph: Graph<B>,
    ) -> Self {
        let order = tensor.order + 1;
        let id = OpsID::new();
        let parents = vec![tensor.id.clone()];
        let metadata = Metadata::new(
            parents,
            order,
            id,
            tensor.requirement.infer(&Requirement::None),
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

    pub fn to_backward_if_required(&self) -> Option<BackwardTensor<B, D>> {
        match self.metadata.requirement {
            Requirement::None => None,
            _ => Some(self.to_backward()),
        }
    }

    pub fn to_metadata_if_required(&self) -> Option<MetadataRef> {
        match self.metadata.requirement {
            Requirement::None => None,
            _ => Some(self.metadata.clone()),
        }
    }

    pub fn register_ops<O: Backward<B> + 'static>(self, ops: O) -> Self {
        self.graph.register(&self.metadata.id, Box::new(ops));
        self
    }
}

pub trait GetMetadata {
    fn metadata(&self) -> MetadataRef;
}

impl<B: Backend, const D: usize> GetMetadata for BackwardTensor<B, D> {
    fn metadata(&self) -> MetadataRef {
        self.metadata.clone()
    }
}

impl GetMetadata for MetadataRef {
    fn metadata(&self) -> MetadataRef {
        self.clone()
    }
}

/// May clone the type if necessary.
pub fn clone_if_shared<T1, T2, T3: Clone>(
    lhs: &Option<T1>,
    rhs: &Option<T2>,
    maybe_cloned: T3,
) -> (Option<T3>, Option<T3>) {
    if lhs.is_some() && rhs.is_none() {
        return (Some(maybe_cloned), None);
    }
    if lhs.is_none() && rhs.is_some() {
        return (None, Some(maybe_cloned));
    }

    if lhs.is_none() && rhs.is_none() {
        return (None, None);
    }

    (Some(maybe_cloned.clone()), Some(maybe_cloned))
}
