#[burn_tensor_testgen::testgen(bridge)]
mod tests {
    use super::*;
    use burn_tensor::{backend::Backend, Distribution, Tensor};

    #[test]
    fn test_full_precision() {
        let device = Default::default();
        let x1 = Tensor::<TestAutodiffBackend, 2>::random([32, 32], Distribution::Default, &device)
            .require_grad();
        let x2 = Tensor::<TestAutodiffBackend, 2>::random([32, 32], Distribution::Default, &device)
            .require_grad();

        let x3 = x1.clone().into_full_precision();
        let x4 = x2.clone().into_full_precision();

        let x5 = x3.matmul(x4);
        let x6 = Tensor::<TestAutodiffBackend, 2>::from_full_precision(x5);
        let x7 = x6 * x1.clone() / x2.clone();

        let mut grads = x7.backward();

        let x1_grad = x1.grad(&mut grads);
        let x2_grad = x2.grad(&mut grads);

        assert!(x1_grad.is_some());
        assert!(x2_grad.is_some());
    }
}
