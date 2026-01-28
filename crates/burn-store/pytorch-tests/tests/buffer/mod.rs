use burn::{
    module::{Module, Param},
    tensor::{Tensor, backend::Backend},
};

#[derive(Module, Debug)]
pub struct Net<B: Backend> {
    buffer: Param<Tensor<B, 2>>,
}

impl<B: Backend> Net<B> {
    /// Create a new model with placeholder values.
    pub fn init(device: &B::Device) -> Self {
        Self {
            buffer: Param::from_tensor(Tensor::zeros([3, 3], device)),
        }
    }

    /// Forward pass of the model.
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        self.buffer.val() + x
    }
}

#[cfg(test)]
mod tests {
    use crate::backend::TestBackend;

    use burn::tensor::Tolerance;
    use burn_store::{ModuleSnapshot, PytorchStore};

    use super::*;

    #[test]
    fn buffer() {
        let device = Default::default();
        let mut model = Net::<TestBackend>::init(&device);
        let mut store = PytorchStore::from_file("tests/buffer/buffer.pt");
        model
            .load_from(&mut store)
            .expect("Should decode state successfully");

        let input = Tensor::<TestBackend, 2>::ones([3, 3], &device);

        let output = model.forward(input);

        let expected = Tensor::<TestBackend, 2>::ones([3, 3], &device) * 2.0;

        output
            .to_data()
            .assert_approx_eq::<f32>(&expected.to_data(), Tolerance::default());
    }
}
