use std::marker::PhantomData;

use burn_tensor::backend::Backend;

use crate::{
    grads::Gradients,
    graph::{
        Node, NodeID, NodeRef, Requirement, {Graph, Step},
    },
    ADBackendDecorator,
};

use burn_tensor::ops::*;

#[derive(Debug, Clone)]
pub struct ADTensor<B: Backend, const D: usize> {
    pub primitive: B::TensorPrimitive<D>,
    pub node: NodeRef,
    pub(crate) graph: Graph<B>,
}

#[derive(new, Debug, Clone)]
pub struct BackwardTensor<B: Backend, const D: usize> {
    pub primitive: B::TensorPrimitive<D>,
    pub node: NodeRef,
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
struct RootStep<B: Backend> {
    node: NodeRef,
    phantom: PhantomData<B>,
}

impl<B: Backend> Step<B> for RootStep<B> {
    fn step(self: Box<Self>, _grads: &mut Gradients<B>) {
        // Nothing to do
    }

    fn node(&self) -> NodeRef {
        self.node.clone()
    }
}

impl<B: Backend, const D: usize> ADTensor<B, D> {
    pub fn new(primitive: B::TensorPrimitive<D>) -> Self {
        let id = NodeID::new();
        let node = Node::new(vec![], 0, id, Requirement::None);
        let tensor = Self {
            primitive,
            node: node.into(),
            graph: Graph::new(),
        };
        tensor.require_grad()
    }

    pub fn require_grad(mut self) -> Self {
        match self.node.requirement {
            Requirement::Grad => self,
            Requirement::GradInBackward => {
                panic!("Can't require grad to a non leaf tensor")
            }
            Requirement::None => {
                let node = Node::new(vec![], 0, self.node.id.clone(), Requirement::Grad);
                self.node = node.into();
                let ops = RootStep::new(self.node.clone());
                self.register_ops(ops)
            }
        }
    }

    pub fn from_ops<const N: usize>(
        nodes: &[NodeRef; N],
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
            .map(|node| node.order)
            .reduce(usize::max)
            .unwrap_or(0)
            + 1;

        let node = Node::new(
            nodes.iter().map(|node| node.id.clone()).collect(),
            order,
            NodeID::new(),
            requirement,
        );

        Self {
            primitive: output,
            node: node.into(),
            graph,
        }
    }

    pub fn to_backward(&self) -> BackwardTensor<B, D> {
        BackwardTensor::new(self.primitive.clone(), self.node.clone())
    }

    pub fn register_ops<O: Step<B> + 'static>(mut self, ops: O) -> Self {
        self.graph = self.graph.register(&self.node.id, Box::new(ops));
        self
    }
}
