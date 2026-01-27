use crate::include_models;
include_models!(split);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Shape, Tensor};

    use crate::backend::TestBackend;

    #[test]
    fn split() {
        let device = Default::default();
        let model = split::Model::<TestBackend>::new(&device);
        let shape = [5, 2];
        let input = Tensor::ones(shape, &device);

        let (tensor_1, tensor_2, tensor_3) = model.forward(input);

        assert_eq!(tensor_1.shape(), Shape::from([2, 2]));
        assert_eq!(tensor_2.shape(), Shape::from([2, 2]));
        assert_eq!(tensor_3.shape(), Shape::from([1, 2]));
    }
}
