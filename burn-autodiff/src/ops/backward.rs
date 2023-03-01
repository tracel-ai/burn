use super::{Ops, OpsPrep};
use crate::{
    grads::Gradients,
    graph::{Graph, NodeRef, Requirement},
    utils::duplicate,
};
use burn_tensor::backend::Backend;

/// Trait for all operations.
///
/// # Notes
///
/// Concrete types implementing this trait should not have any state.
/// If a state is necessary during the backward pass,
/// they should be declared with the associated type 'State'.
pub trait Backward<B, const D: usize, const N: usize>: Send + Sync + std::fmt::Debug
where
    Self: Sized + 'static,
    B: Backend,
{
    /// Associated type to compute the backward pass.
    type State: Clone + Send + Sync + std::fmt::Debug + 'static;

    /// The backward pass.
    fn backward(self, ops: Ops<Self::State, N>, grads: &mut Gradients);

    /// Prepare the backward ops.
    fn prepare(
        self,
        nodes: [NodeRef; N],
        graphs: [Graph; N],
    ) -> OpsPrep<Self, B, Self::State, D, N> {
        let requirement = Requirement::from_nodes(&nodes);
        OpsPrep::new(nodes, graphs, requirement, self)
    }
}

/// Execute a binary operation during the backward step.
pub fn binary<B, const D_OUT: usize, const D_LHS: usize, const D_RHS: usize, FLhs, FRhs>(
    parents: [Option<NodeRef>; 2],
    node: NodeRef,
    grads: &mut Gradients,
    func_lhs: FLhs,
    func_rhs: FRhs,
) where
    B: Backend,
    FLhs: FnOnce(B::TensorPrimitive<D_OUT>) -> B::TensorPrimitive<D_LHS>,
    FRhs: FnOnce(B::TensorPrimitive<D_OUT>) -> B::TensorPrimitive<D_RHS>,
{
    let [grad_4lhs, grad_4rhs] = duplicate(&parents, Some(grads.consume::<B, D_OUT>(&node)));
    let [node_lhs, node_rhs] = parents;

    if let Some(node) = node_lhs {
        let grad = func_lhs(grad_4lhs.unwrap());
        grads.register::<B, D_LHS>(node, grad)
    }

    if let Some(node) = node_rhs {
        let grad = func_rhs(grad_4rhs.unwrap());
        grads.register::<B, D_RHS>(node, grad)
    }
}

/// Execute a unary operation during the backward step.
pub fn unary<B, const D_OUT: usize, const D_IN: usize, F>(
    parents: [Option<NodeRef>; 1],
    node: NodeRef,
    grads: &mut Gradients,
    func: F,
) where
    B: Backend,
    F: FnOnce(B::TensorPrimitive<D_OUT>) -> B::TensorPrimitive<D_IN>,
{
    let [parent_node] = parents;
    let grad = grads.consume::<B, D_OUT>(&node);

    if let Some(node) = parent_node {
        let grad = func(grad);
        grads.register::<B, D_IN>(node, grad)
    }
}

/// Execute a unary operation during the backward step where the input backend
/// is different from the output backend.
pub fn unary_different_backend<BIn, BOut, const D_OUT: usize, const D_IN: usize, F>(
    parents: [Option<NodeRef>; 1],
    node: NodeRef,
    grads: &mut Gradients,
    func: F,
) where
    BIn: Backend,
    BOut: Backend,
    F: FnOnce(BOut::TensorPrimitive<D_OUT>) -> BIn::TensorPrimitive<D_IN>,
{
    let [parent_node] = parents;
    let grad = grads.consume::<BOut, D_OUT>(&node);

    if let Some(node) = parent_node {
        let grad = func(grad);
        grads.register::<BIn, D_IN>(node, grad)
    }
}
