use crate::graph::converter::Forward2BackwardGraphConverter;
use crate::graph::node::{BackwardNode, BackwardNodeRef, BackwardNodeState, ForwardNodeRef};
use crate::graph::ops::{
    BackwardRecordedOps, BackwardRecordedOpsRef, ForwardRecordedOps, RecordedOpsParentRef,
};
use crate::tensor::backend::Backend;
use crate::tensor::{backend::autodiff::ADTensor, ops::*};
use std::convert::TryInto;
use std::sync::Arc;

#[derive(new, Debug)]
pub struct ForwardCatOps<const D: usize, B: Backend> {
    nodes: Vec<ForwardNodeRef<B::TensorPrimitive<D>>>,
    dim: usize,
}

#[derive(new, Debug)]
pub struct BackwardCatOps<const D: usize, B: Backend> {
    nodes: Vec<BackwardNodeRef<B::TensorPrimitive<D>>>,
    dim: usize,
}

impl<const D: usize, B: Backend> ForwardRecordedOps<B::TensorPrimitive<D>> for ForwardCatOps<D, B> {
    fn to_backward(
        &self,
        graph: &mut Forward2BackwardGraphConverter,
    ) -> BackwardRecordedOpsRef<B::TensorPrimitive<D>> {
        Arc::new(BackwardCatOps::<D, B>::new(
            self.nodes
                .iter()
                .map(|node| {
                    let ops: BackwardNode<B::TensorPrimitive<D>> =
                        BackwardNode::from_node(node, graph);
                    Arc::new(ops)
                })
                .collect(),
            self.dim,
        ))
    }
}

impl<const D: usize, B: Backend> BackwardRecordedOps<B::TensorPrimitive<D>>
    for BackwardCatOps<D, B>
{
    fn backward_step(&self, state: &BackwardNodeState<B::TensorPrimitive<D>>) {
        let grad = state.grad();
        let indexes: Vec<_> = grad.shape().dims.iter().map(|v| 0..*v).collect();
        let indexes: [std::ops::Range<usize>; D] = indexes.try_into().unwrap();

        self.nodes.iter().enumerate().for_each(|(i, node)| {
            let mut indexes = indexes.clone();
            indexes[self.dim] = i..i + 1;
            node.state.update_grad(grad.index(indexes));
        });
    }

    fn backward_parents(&self) -> Vec<RecordedOpsParentRef> {
        self.nodes
            .iter()
            .map(|node| {
                let ops: RecordedOpsParentRef = node.clone();
                ops
            })
            .collect()
    }
}

impl<B: Backend, const D: usize> TensorOpsCat<B::Elem, D> for ADTensor<D, B> {
    fn cat(tensors: Vec<&Self>, dim: usize) -> Self {
        let nodes: Vec<_> = tensors.iter().map(|t| t.node.clone()).collect();
        let order = nodes.iter().map(|node| node.order).max().unwrap() + 1;

        let tensors_inner: Vec<B::TensorPrimitive<D>> =
            tensors.into_iter().map(|a| a.tensor()).collect();
        let tensors_inner_ref: Vec<&B::TensorPrimitive<D>> = tensors_inner.iter().collect();

        let out = TensorOpsCat::cat(tensors_inner_ref, dim);

        let shape = *out.shape();
        let state = crate::graph::node::ForwardNodeState::new(out);

        let ops = ForwardCatOps::<D, B>::new(nodes, dim);
        let ops = Arc::new(ops);

        let node = crate::graph::node::ForwardNode::new(order, state, ops);
        let node = std::sync::Arc::new(node);

        ADTensor { node, shape }
    }
}

#[cfg(test)]
mod tests {
    use crate::tensor::{backend::autodiff::helper::TestADTensor, Data};

    #[test]
    fn should_diff_cat() {
        let data_1 = Data::<_, 2>::from([[2.0, -1.0], [5.0, 2.0]]);
        let data_2 = Data::<_, 2>::from([[5.0, 4.0], [-1.0, 4.0]]);

        let tensor_1 = TestADTensor::from_data(data_1);
        let tensor_2 = TestADTensor::from_data(data_2);

        let tensor_3 = tensor_1.matmul(&tensor_2);
        let grads = tensor_3.backward();

        let grad_1 = tensor_1.grad(&grads).unwrap();
        let grad_2 = tensor_2.grad(&grads).unwrap();

        let mut tensor_1_list = Vec::new();
        let mut tensor_2_list = Vec::new();

        for i in 0..2 {
            tensor_1_list.push(TestADTensor::from_data(
                tensor_1.index([i..i + 1]).to_data(),
            ));
            tensor_2_list.push(TestADTensor::from_data(
                tensor_2.index([i..i + 1]).to_data(),
            ));
        }

        let tensor_1_cat = TestADTensor::cat(tensor_1_list.clone(), 0);
        let tensor_2_cat = TestADTensor::cat(tensor_2_list.clone(), 0);

        let tensor_3_cat = tensor_1_cat.matmul(&tensor_2_cat);
        let grads_cat = tensor_3_cat.backward();

        let grad_1_cat = tensor_1_cat.grad(&grads_cat).unwrap();
        let grad_2_cat = tensor_2_cat.grad(&grads_cat).unwrap();

        let grad_1_list_1 = tensor_1_list.get(0).unwrap().grad(&grads_cat).unwrap();
        let grad_1_list_2 = tensor_1_list.get(1).unwrap().grad(&grads_cat).unwrap();

        let grad_2_list_1 = tensor_2_list.get(0).unwrap().grad(&grads_cat).unwrap();
        let grad_2_list_2 = tensor_2_list.get(1).unwrap().grad(&grads_cat).unwrap();

        grad_1.to_data().assert_approx_eq(&grad_1_cat.to_data(), 3);
        grad_2.to_data().assert_approx_eq(&grad_2_cat.to_data(), 3);

        grad_1
            .index([0..1])
            .to_data()
            .assert_approx_eq(&grad_1_list_1.to_data(), 3);

        grad_1
            .index([1..2])
            .to_data()
            .assert_approx_eq(&grad_1_list_2.to_data(), 3);
        grad_2
            .index([0..1])
            .to_data()
            .assert_approx_eq(&grad_2_list_1.to_data(), 3);

        grad_2
            .index([1..2])
            .to_data()
            .assert_approx_eq(&grad_2_list_2.to_data(), 3);
    }
}
