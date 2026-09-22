use core::marker::PhantomData;

use burn_backend::{Backend, TensorMetadata, tensor::FloatTensor};

use crate::{
    Autodiff,
    checkpoint::{base::Checkpointer, strategy::CheckpointStrategy},
    grads::Gradients,
    ops::{Backward, Ops, OpsKind},
};

/// Transfers plain floating-point primitives between two backends for a recorded operation.
///
/// Both directions must preserve shape, dtype, and values on the requested device. The backward
/// direction transfers the incoming gradient, not the original forward input. Implementations
/// choose the transfer mechanism independently in each direction.
pub trait DifferentiableTransfer<Src: Backend, Dst: Backend>:
    Send + core::fmt::Debug + 'static
{
    /// Transfers values to the destination backend.
    fn forward(tensor: FloatTensor<Src>, device: &Dst::Device) -> FloatTensor<Dst>;

    /// Transfers gradients back to the source backend.
    fn backward(tensor: FloatTensor<Dst>, device: &Src::Device) -> FloatTensor<Src>;
}

#[derive(Debug)]
struct Transfer<Src, Dst, Adapter>(PhantomData<(Src, Dst, Adapter)>);

impl<Src, Dst, Adapter> Backward<Dst, 1> for Transfer<Src, Dst, Adapter>
where
    Src: Backend,
    Dst: Backend,
    Adapter: DifferentiableTransfer<Src, Dst>,
{
    type State = Src::Device;

    fn backward(
        self,
        ops: Ops<Self::State, 1>,
        grads: &mut Gradients,
        _checkpointer: &mut Checkpointer,
    ) {
        let grad = grads.consume::<Dst>(&ops.node);
        if let Some(parent) = &ops.parents[0] {
            let grad = Adapter::backward(grad, &ops.state);
            grads.register::<Src>(parent.id, grad);
        }
    }
}

impl<Src: Backend, C: CheckpointStrategy> Autodiff<Src, C> {
    /// Transfers a tensor to another backend through the supplied adapter.
    ///
    /// Tracked inputs remain connected to the graph; their outputs are non-leaf tensors and cannot
    /// retain their own gradients. Detach the output before requiring its gradient to start a new
    /// leaf on the destination, severing the source connection. Untracked inputs remain untracked.
    /// The checkpointing strategy is preserved.
    /// Transfers aren't replayed during gradient checkpointing.
    ///
    /// Distributed backward requires every distributed parameter to use the same backend as the
    /// loss. Incompatible graphs panic before synchronization or gradient computation begins.
    pub fn to_backend<Dst, Adapter>(
        tensor: FloatTensor<Self>,
        device: &Dst::Device,
    ) -> FloatTensor<Autodiff<Dst, C>>
    where
        Dst: Backend,
        Adapter: DifferentiableTransfer<Src, Dst>,
    {
        let source_device = tensor.primitive.device();
        let prep = Transfer::<Src, Dst, Adapter>(PhantomData)
            .prepare::<C>([tensor.node()])
            .compute_bound()
            .stateful();
        let output = Adapter::forward(tensor.primitive, device);

        match prep {
            OpsKind::Tracked(prep) => prep.finish(source_device, output),
            OpsKind::UnTracked(prep) => prep.finish(output),
        }
    }
}
