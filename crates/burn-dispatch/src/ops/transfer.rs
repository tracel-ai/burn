use burn_backend::{Backend, tensor::FloatTensor};

/// Default cross-backend float transfer, also used by the autodiff adapter.
#[allow(
    dead_code,
    reason = "Only used when dispatch generates cross-backend or capture transfer arms"
)]
pub(crate) fn float_transfer<Src: Backend, Dst: Backend>(
    tensor: FloatTensor<Src>,
    device: &Dst::Device,
) -> FloatTensor<Dst> {
    let data = burn_backend::read_sync(Src::float_into_data(tensor))
        .expect("Failed to read tensor during cross-backend transfer");
    Dst::float_from_data(data, device)
}

/// Selected by the dispatch matrix for pairs using host-mediated transfers.
#[cfg(feature = "autodiff")]
#[derive(Debug)]
#[allow(
    dead_code,
    reason = "Only used when dispatch generates transfers between multiple compute backends"
)]
pub(crate) struct HostTransfer;

#[cfg(feature = "autodiff")]
impl<Src: Backend, Dst: Backend> burn_autodiff::ops::DifferentiableTransfer<Src, Dst>
    for HostTransfer
{
    fn forward(tensor: FloatTensor<Src>, device: &Dst::Device) -> FloatTensor<Dst> {
        float_transfer::<Src, Dst>(tensor, device)
    }

    fn backward(tensor: FloatTensor<Dst>, device: &Src::Device) -> FloatTensor<Src> {
        float_transfer::<Dst, Src>(tensor, device)
    }
}

#[cfg(all(test, feature = "autodiff", feature = "flex", feature = "ndarray"))]
mod tests {
    use super::*;
    use burn_autodiff::{
        Autodiff, checkpoint::strategy::BalancedCheckpointing, ops::DifferentiableTransfer,
    };
    use burn_backend::{AutodiffBackend, TensorData, TensorMetadata, ops::FloatTensorOps};
    use core::sync::atomic::{AtomicUsize, Ordering};

    use crate::backends::{Flex, NdArray};

    /// Used when backward must not execute, either for an invalid graph or an untracked constant.
    #[derive(Debug)]
    struct ForwardOnlyTransfer;

    impl<Src: Backend, Dst: Backend> DifferentiableTransfer<Src, Dst> for ForwardOnlyTransfer {
        fn forward(tensor: FloatTensor<Src>, device: &Dst::Device) -> FloatTensor<Dst> {
            float_transfer::<Src, Dst>(tensor, device)
        }

        fn backward(_tensor: FloatTensor<Dst>, _device: &Src::Device) -> FloatTensor<Src> {
            panic!("this test must not execute transfer backward")
        }
    }

    #[test]
    #[should_panic(
        expected = "Distributed backward requires all distributed parameters to use the same backend as the loss"
    )]
    fn distributed_parameter_on_a_different_backend_is_rejected_before_backward() {
        type Src = Autodiff<Flex, BalancedCheckpointing>;
        type Dst = Autodiff<NdArray, BalancedCheckpointing>;
        let x = Src::float_from_data(TensorData::from([2.0f32, 3.0]), &Default::default())
            .grad_distributed(burn_backend::distributed::DistributedParamId::new());
        // Rebuilding the root when enabling gradients must preserve its backend identity.
        let x = Src::float_set_require_grad(x, true);
        // The transferred node itself has no distributed metadata. The check must reach its
        // distributed ancestor, before the adapter's backward (which deliberately panics) runs.
        let derived = Src::float_mul(x.clone(), x);
        let moved = Src::to_backend::<NdArray, ForwardOnlyTransfer>(derived, &Default::default());
        let _ = Dst::backward(Dst::float_sum(moved));
    }

    #[test]
    fn distributed_branch_can_join_a_round_trip() {
        type Src = Autodiff<Flex>;
        let device = Default::default();
        let x = Src::float_set_require_grad(
            Src::float_from_data(TensorData::from([2.0f32, 3.0]), &device),
            true,
        );
        let distributed = Src::float_set_require_grad(
            Src::float_from_data(TensorData::from([4.0f32, 5.0]), &device),
            true,
        )
        .grad_distributed(burn_backend::distributed::DistributedParamId::new());
        let moved = Src::to_backend::<NdArray, HostTransfer>(x.clone(), &Default::default());
        let returned = Autodiff::<NdArray>::to_backend::<Flex, HostTransfer>(moved, &device);
        let grads = Src::backward(Src::float_sum(Src::float_add(
            returned,
            distributed.clone(),
        )));
        for tensor in [&x, &distributed] {
            burn_backend::read_sync(Flex::float_into_data(Src::grad(tensor, &grads).unwrap()))
                .unwrap()
                .assert_eq(&TensorData::from([1.0f32, 1.0]), true);
        }
    }

    #[test]
    fn distributed_graph_accepts_untracked_transferred_constants_and_same_backend_transfers() {
        type Ad = Autodiff<Flex>;
        let device = Default::default();
        let x = Ad::float_set_require_grad(
            Ad::float_from_data(TensorData::from([2.0f32, 3.0]), &device),
            true,
        )
        .grad_distributed(burn_backend::distributed::DistributedParamId::new());
        let constant = Autodiff::<NdArray>::float_from_data(
            TensorData::from([4.0f32, 5.0]),
            &Default::default(),
        );
        let constant =
            Autodiff::<NdArray>::to_backend::<Flex, ForwardOnlyTransfer>(constant, &device);
        let moved = Ad::to_backend::<Flex, HostTransfer>(x.clone(), &device);
        let grads = Ad::backward(Ad::float_sum(Ad::float_mul(moved, constant)));
        burn_backend::read_sync(Flex::float_into_data(Ad::grad(&x, &grads).unwrap()))
            .unwrap()
            .assert_eq(&TensorData::from([4.0f32, 5.0]), true);
    }

    static FORWARD: AtomicUsize = AtomicUsize::new(0);
    static BACKWARD: AtomicUsize = AtomicUsize::new(0);

    #[derive(Debug)]
    struct ObservedTransfer;

    impl DifferentiableTransfer<Flex, NdArray> for ObservedTransfer {
        fn forward(
            tensor: FloatTensor<Flex>,
            device: &<NdArray as burn_backend::BackendTypes>::Device,
        ) -> FloatTensor<NdArray> {
            FORWARD.fetch_add(1, Ordering::Relaxed);
            float_transfer::<Flex, NdArray>(tensor, device)
        }

        fn backward(
            tensor: FloatTensor<NdArray>,
            device: &<Flex as burn_backend::BackendTypes>::Device,
        ) -> FloatTensor<Flex> {
            BACKWARD.fetch_add(1, Ordering::Relaxed);
            assert_eq!(*device, Default::default());
            float_transfer::<NdArray, Flex>(tensor, device)
        }
    }

    #[test]
    fn recorded_transfer_uses_adapter_in_both_directions_without_replay() {
        type Src = Autodiff<Flex, BalancedCheckpointing>;
        type Dst = Autodiff<NdArray, BalancedCheckpointing>;
        let source = Default::default();
        let destination = Default::default();
        let x = Src::float_set_require_grad(
            Src::float_from_data(TensorData::from([2.0f32, 3.0]), &source),
            true,
        );
        let moved = Src::to_backend::<NdArray, ObservedTransfer>(x.clone(), &destination);
        let output = Dst::float_sum(Dst::float_mul(moved.clone(), moved));
        let grads = Dst::backward(output);
        let grad = Src::grad(&x, &grads).unwrap();
        assert_eq!(grad.device(), source);
        burn_backend::read_sync(Flex::float_into_data(grad))
            .unwrap()
            .assert_eq(&TensorData::from([4.0f32, 6.0]), true);
        assert_eq!(FORWARD.load(Ordering::Relaxed), 1);
        assert_eq!(BACKWARD.load(Ordering::Relaxed), 1);
    }
}
