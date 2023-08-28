#[burn_tensor_testgen::testgen(ad_broadcast)]
mod tests {
    use super::*;
    use burn_tensor::{Data, Distribution, Int, Shape, Tensor};

    #[test]
    fn mul_broadcast() {
        test_ops_broadcast_backward(|x, y| x * y);
    }

    #[test]
    fn div_broadcast() {
        test_ops_broadcast_backward(|x, y| x / y);
    }

    #[test]
    fn sub_broadcast() {
        test_ops_broadcast_backward(|x, y| x - y);
    }

    #[test]
    fn add_broadcast() {
        test_ops_broadcast_backward(|x, y| x + y);
    }

    #[test]
    fn matmul_broadcast() {
        test_ops_broadcast_backward(|x, y| x.matmul(y));
    }

    #[test]
    fn mask_where_broadcast() {
        test_ops_broadcast_backward(|x, y| x.mask_where(y.clone().equal_elem(4), y));
    }

    fn test_ops_broadcast_backward<F>(func: F)
    where
        F: Fn(Tensor<TestADBackend, 3>, Tensor<TestADBackend, 3>) -> Tensor<TestADBackend, 3>,
    {
        let w = TestADTensor::zeros([16, 5, 5]).require_grad();
        let x = TestADTensor::zeros([4, 5, 5]).require_grad();

        // Slice isn't a broadcastable operation, so it will fail when the previous backward pass
        // of an operation that support broadcast doesn't support it during the backward pass.
        let y = func(w.clone().slice([0..1]), x.clone());

        // Will panic if broadcast isn't supported!
        let grads = y.backward();

        let w_grad = w.grad(&grads).unwrap();
        let x_grad = x.grad(&grads).unwrap();

        assert_eq!(w_grad.shape(), w.shape());
        assert_eq!(x_grad.shape(), x.shape());
    }
}
