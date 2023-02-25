use burn_tensor::backend::Backend;

use crate::{
    grads::Gradients,
    graph::ops::{Backward, MetadataRef, Requirement},
    tensor::{ADTensor, BackwardTensor},
};

/// Binary operation that does not require tensors collected during the foward pass.
pub trait BinaryOpsNoCapture<B: Backend, const D: usize>: Send + Sync + std::fmt::Debug
where
    Self: Sized + 'static,
{
    fn forward(
        &self,
        lhs: B::TensorPrimitive<D>,
        rhs: B::TensorPrimitive<D>,
    ) -> B::TensorPrimitive<D>;
    fn backward(
        self,
        lhs: Option<MetadataRef>,
        rhs: Option<MetadataRef>,
        output: BackwardTensor<B, D>,
        grads: &mut Gradients<B>,
    );
    fn execute(self, lhs: ADTensor<B, D>, rhs: ADTensor<B, D>) -> ADTensor<B, D> {
        let output = ADTensor::from_binary_ops(
            lhs.metadata.clone(),
            rhs.metadata.clone(),
            self.forward(lhs.primitive, rhs.primitive),
            lhs.graph,
            rhs.graph,
        );

        if let Requirement::None = output.metadata.requirement {
            return output;
        }

        let ops = BinaryOpsNoCaptureBackward::new(
            lhs.metadata.clone_if_require_grad(),
            rhs.metadata.clone_if_require_grad(),
            output.to_backward(),
            self,
        );

        output.register_ops(ops)
    }
}

/// Binary operation that requires lhs AND rhs tensors to be collected during the foward pass.
pub trait BinaryOps<B: Backend, const D: usize>: Send + Sync + std::fmt::Debug
where
    Self: Sized + 'static,
{
    fn forward(
        &self,
        lhs: B::TensorPrimitive<D>,
        rhs: B::TensorPrimitive<D>,
    ) -> B::TensorPrimitive<D>;
    fn backward(
        self,
        lhs: Option<BackwardTensor<B, D>>,
        rhs: Option<BackwardTensor<B, D>>,
        output: BackwardTensor<B, D>,
        grads: &mut Gradients<B>,
    );
    fn execute(self, lhs: ADTensor<B, D>, rhs: ADTensor<B, D>) -> ADTensor<B, D> {
        let lhs_backward_tensor = lhs.to_backward_if_required();
        let rhs_backward_tensor = rhs.to_backward_if_required();

        let output = ADTensor::from_binary_ops(
            lhs.metadata.clone(),
            rhs.metadata.clone(),
            self.forward(lhs.primitive, rhs.primitive),
            lhs.graph,
            rhs.graph,
        );

        if let Requirement::None = output.metadata.requirement {
            return output;
        }

        let ops = BinaryOpsBackward::new(
            lhs_backward_tensor,
            rhs_backward_tensor,
            output.to_backward(),
            self,
        );

        output.register_ops(ops)
    }
}

#[derive(new, Debug)]
struct BinaryOpsNoCaptureBackward<B, const D: usize, T>
where
    B: Backend,
    T: BinaryOpsNoCapture<B, D>,
{
    lhs: Option<MetadataRef>,
    rhs: Option<MetadataRef>,
    output: BackwardTensor<B, D>,
    ops: T,
}

impl<B, const D: usize, T> Backward<B> for BinaryOpsNoCaptureBackward<B, D, T>
where
    B: Backend,
    T: BinaryOpsNoCapture<B, D>,
{
    fn backward(self: Box<Self>, grads: &mut Gradients<B>) {
        self.ops.backward(self.lhs, self.rhs, self.output, grads);
    }

    fn metadata(&self) -> MetadataRef {
        self.output.metadata.clone()
    }
}

#[derive(new, Debug)]
struct BinaryOpsBackward<B, const D: usize, T>
where
    B: Backend,
    T: BinaryOps<B, D>,
{
    lhs: Option<BackwardTensor<B, D>>,
    rhs: Option<BackwardTensor<B, D>>,
    output: BackwardTensor<B, D>,
    ops: T,
}

impl<B, const D: usize, T> Backward<B> for BinaryOpsBackward<B, D, T>
where
    B: Backend,
    T: BinaryOps<B, D>,
{
    fn backward(self: Box<Self>, grads: &mut Gradients<B>) {
        self.ops.backward(self.lhs, self.rhs, self.output, grads);
    }

    fn metadata(&self) -> MetadataRef {
        self.output.metadata.clone()
    }
}
