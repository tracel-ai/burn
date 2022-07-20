use super::ADTensor;
use crate::{node::Zeros, tape::Tape};
use std::ops::Add;

impl<T, P, const D: usize> ADTensor<P, D, T> {
    pub fn backprob(&self) {
        let mut tape = Tape::new();
        self.node.ops.set_last_ops();
        self.node.record(&mut tape);
        tape.backward();
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

#[cfg(test)]
mod tests {
    use crate::{backend::autodiff::helper::ADTchTensor, Data, TensorBase, TensorOpsMatmul};

    #[test]
    fn should_diff_complex_1() {
        let data_1: Data<f64, 2> = Data::from([[1.0, 7.0], [13.0, -3.0]]);
        let data_2: Data<f64, 2> = Data::from([[4.0, 7.0], [2.0, 3.0]]);
        let data_3: Data<f64, 2> = Data::from([[2.0, 2.0], [2.0, 2.0]]);

        let tensor_1 = ADTchTensor::from_data(data_1.clone());
        let tensor_2 = ADTchTensor::from_data(data_2.clone());
        let tensor_3 = ADTchTensor::from_data(data_3.clone());

        let tensor_4 = tensor_1.matmul(&tensor_2);
        let tensor_5 = tensor_4.matmul(&tensor_3);

        tensor_5.backprob();

        let grad_1 = tensor_1.grad();
        let grad_2 = tensor_2.grad();

        assert_eq!(grad_1.into_data(), Data::from([[44.0, 20.0], [44.0, 20.0]]));
        assert_eq!(grad_2.into_data(), Data::from([[56.0, 56.0], [16.0, 16.0]]));
    }

    #[test]
    fn should_diff_complex_2() {
        let data_1: Data<f64, 2> = Data::from([[1.0, 7.0], [13.0, -3.0]]);
        let data_2: Data<f64, 2> = Data::from([[4.0, 7.0], [2.0, 3.0]]);
        let data_3: Data<f64, 2> = Data::from([[2.0, 2.0], [2.0, 2.0]]);

        let tensor_1 = ADTchTensor::from_data(data_1.clone());
        let tensor_2 = ADTchTensor::from_data(data_2.clone());
        let tensor_3 = ADTchTensor::from_data(data_3.clone());

        let tensor_4 = tensor_1.matmul(&tensor_2);
        let tensor_5 = tensor_4.matmul(&tensor_3);
        let tensor_6 = tensor_1.matmul(&tensor_5);

        tensor_6.backprob();

        let grad_1 = tensor_1.grad();
        let grad_2 = tensor_2.grad();

        assert_eq!(
            grad_1.into_data(),
            Data::from([[800.0, 792.0], [360.0, 592.0]])
        );
        assert_eq!(
            grad_2.into_data(),
            Data::from([[264.0, 264.0], [344.0, 344.0]])
        );
    }
}
