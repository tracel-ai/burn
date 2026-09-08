use burn_backend::{Backend, tensor::FloatTensor};

/// Default cross-backend float transfer, also used by the autodiff adapter.
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
