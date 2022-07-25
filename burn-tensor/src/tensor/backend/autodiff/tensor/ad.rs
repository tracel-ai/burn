use super::ADTensor;
use crate::{
    grad::{AsNode, Gradients},
    node::{Ones, Zeros},
};
use std::ops::Add;

impl<T, P, const D: usize> ADTensor<P, D, T>
where
    T: Zeros<T> + Ones<T> + Clone + Add<Output = T>,
    T: std::fmt::Debug + 'static,
{
    pub fn backward(&self) -> Gradients {
        self.node.backward()
    }
}

impl<T, P, const D: usize> ADTensor<P, D, T>
where
    T: Zeros<T> + Clone + Add<Output = T>,
    T: std::fmt::Debug,
{
    pub fn grad(&self) -> T {
        self.node.state.borrow_mut().grad()
    }
}

impl<T, P, const D: usize> AsNode<T> for ADTensor<P, D, T> {
    fn as_node(&self) -> &crate::node::Node<T> {
        &self.node
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        backend::autodiff::helper::ADTchTensor, Data, TensorBase, TensorOpsAdd, TensorOpsMatmul,
        TensorOpsMul, TensorOpsSub,
    };

    #[test]
    fn should_diff_full_complex_1() {
        let data_1: Data<f64, 2> = Data::from([[1.0, 7.0], [13.0, -3.0]]);
        let data_2: Data<f64, 2> = Data::from([[4.0, 7.0], [2.0, 3.0]]);

        let tensor_1 = ADTchTensor::from_data(data_1.clone());
        let tensor_2 = ADTchTensor::from_data(data_2.clone());

        let tensor_3 = tensor_1.matmul(&tensor_2);
        let tensor_4 = tensor_3.matmul(&tensor_1);
        let tensor_5 = tensor_4.mul(&tensor_2);

        tensor_5.backward();

        let grad_1 = tensor_1.grad();
        let grad_2 = tensor_2.grad();

        assert_eq!(
            grad_1.into_data(),
            Data::from([[593., 463.0], [487.0, 539.0]])
        );
        assert_eq!(
            grad_2.into_data(),
            Data::from([[734.0, 294.0], [1414.0, 242.0]])
        );
    }

    #[test]
    fn should_diff_full_complex_2() {
        let data_1: Data<f64, 2> = Data::from([[1.0, 7.0], [13.0, -3.0]]);
        let data_2: Data<f64, 2> = Data::from([[4.0, 7.0], [2.0, 3.0]]);

        let tensor_1 = ADTchTensor::from_data(data_1.clone());
        let tensor_2 = ADTchTensor::from_data(data_2.clone());

        let tensor_3 = tensor_1.matmul(&tensor_2);
        let tensor_4 = tensor_3.matmul(&tensor_1);
        let tensor_5 = tensor_4.add_scalar(&17.0).add(&tensor_2);

        tensor_5.backward();

        let grad_1 = tensor_1.grad();
        let grad_2 = tensor_2.grad();

        assert_eq!(
            grad_1.into_data(),
            Data::from([[166.0, 110.0], [212.0, 156.0]])
        );
        assert_eq!(
            grad_2.into_data(),
            Data::from([[113.0, 141.0], [33.0, 41.0]])
        );
    }

    #[test]
    fn should_diff_full_complex_3() {
        let data_1: Data<f64, 2> = Data::from([[1.0, 7.0], [13.0, -3.0]]);
        let data_2: Data<f64, 2> = Data::from([[4.0, 7.0], [2.0, 3.0]]);

        let tensor_1 = ADTchTensor::from_data(data_1.clone());
        let tensor_2 = ADTchTensor::from_data(data_2.clone());

        let tensor_3 = tensor_1.matmul(&tensor_2);
        let tensor_4 = tensor_3.matmul(&tensor_1);
        let tensor_5 = tensor_4.sub(&tensor_2);
        let tensor_6 = tensor_5.add(&tensor_4);

        let grads = tensor_6.backward();

        let grad_1 = grads.wrt(&tensor_1).unwrap();
        let grad_2 = grads.wrt(&tensor_2).unwrap();

        assert_eq!(
            grad_1.to_data(),
            Data::from([[332.0, 220.0], [424.0, 312.0]])
        );
        assert_eq!(grad_2.to_data(), Data::from([[223.0, 279.0], [63.0, 79.0]]));
    }
}
