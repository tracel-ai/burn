// use crate::{
//     graph::{
//         node::{ForwardNode, ForwardNodeState},
//         ops::{ForwardUnaryRecordedOps, UnaryOps, UnaryOpsNodeState},
//     },
//     tensor::{
//         backend::autodiff::{ADKind, ADTensor},
//         ops::TensorOpsBackend,
//         Backend, Element, Tensor, TensorTrait, TensorType,
//     },
// };
// use std::sync::Arc;
//
// #[derive(Debug)]
// struct ADTensorOpsBackend<E, const D: usize> {
//     _kind: ADKind<E>,
// }
//
// impl<P: Default, const D: usize> ADTensorOpsBackend<P, D> {
//     pub fn new() -> Self {
//         Self {
//             _kind: ADKind::new(),
//         }
//     }
// }
//
// impl<T1, T2, E, const D: usize> UnaryOps<T1, T2> for ADTensorOpsBackend<E, D>
// where
//     E: Element,
//     T1: TensorTrait<E, D> + TensorOpsBackend<E, D, Input = Backend<E = E>, Output = T2>,
//     T2: TensorTrait<E, D> + TensorOpsBackend<E, D, T2, Output = T1>,
// {
//     fn partial(&self, state: &UnaryOpsNodeState<T1, T2>) -> T1 {
//         state.output.grad().to_backend()
//     }
// }
//
// impl<T1, T2, E, const D: usize, B: Backend> TensorOpsBackend<E, D, B>
//     for ADTensor<E, D, Tensor<D, B1>>
// where
//     T1: TensorTrait<E, D> + TensorOpsBackend<E, D, B2, Output = T2>,
//     T2: TensorTrait<E, D> + TensorOpsBackend<E, D, B1, Output = T1>,
//     E: Element + tch::kind::Element,
//     B1: Backend<E = E> + TensorType<D, B>,
// {
//     type Output = ADTensor<E, D, Tensor<D, B>>;
//
//     fn to_backend(&self) -> Self::Output {
//         let tensor: Tensor<D, B> = self.tensor().to_backend();
//         let state = ForwardNodeState::new(tensor);
//
//         let ops = ADTensorOpsBackend::<E, D, B>::new(B1::default(), B2::default());
//         let ops = Arc::new(ops);
//         let ops = ForwardUnaryRecordedOps::new(self.node.clone(), ops);
//         let ops = Arc::new(ops);
//
//         let node = ForwardNode::from_unary(&self.node, state, ops);
//         let node = Arc::new(node);
//
//         let shape = self.shape.clone();
//         let kind = self.kind.clone();
//
//         ADTensor { node, shape, kind }
//     }
// }
