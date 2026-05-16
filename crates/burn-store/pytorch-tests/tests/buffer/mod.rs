use burn::{
    module::{Module, Param},
    tensor::{Device, Tensor},
};

#[derive(Module, Debug)]
pub struct Net {
    buffer: Param<Tensor<2>>,
}

impl Net {
    /// Create a new model with placeholder values.
    pub fn init(device: &Device) -> Self {
        Self {
            buffer: Param::from_tensor(Tensor::zeros([3, 3], device)),
        }
    }

    /// Forward pass of the model.
    pub fn forward(&self, x: Tensor<2>) -> Tensor<2> {
        self.buffer.val() + x
    }
}

#[cfg(test)]
mod tests {

    use burn::tensor::Tolerance;
    use burn_store::{ModuleSnapshot, PytorchStore};

    use super::*;

    #[test]
    fn buffer() {
        let device = Default::default();
        let mut model = Net::init(&device);
        let mut store = PytorchStore::from_file("tests/buffer/buffer.pt");
        model
            .load_from(&mut store)
            .expect("Should decode state successfully");

        let input = Tensor::<2>::ones([3, 3], &device);

        let output = model.forward(input);

        let expected = Tensor::<2>::ones([3, 3], &device) * 2.0;

        output
            .to_data()
            .assert_approx_eq::<f32>(&expected.to_data(), Tolerance::default());
    }
}
