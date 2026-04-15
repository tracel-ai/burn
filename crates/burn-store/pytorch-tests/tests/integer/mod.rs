use burn::{
    module::{Module, Param, ParamId},
    tensor::{Int, Tensor, TensorData, Device},
};

#[derive(Module, Debug)]
pub struct Net {
    buffer: Param<Tensor< 1, Int>>,
}

impl Net {
    /// Create a new model with placeholder values.
    pub fn init(device: &Device) -> Self {
        Self {
            buffer: Param::initialized(
                ParamId::new(),
                Tensor::< 1, Int>::from_data(TensorData::from([0, 0, 0]), device),
            ),
        }
    }

    /// Forward pass of the model.
    pub fn forward(&self, _x: Tensor< 2>) -> Tensor< 1, Int> {
        self.buffer.val()
    }
}

#[cfg(test)]
mod tests {
    
    use burn::tensor::TensorData;
    use burn_store::{ModuleSnapshot, PytorchStore};

    use super::*;

    fn integer(model: Net) {
        let device = Default::default();

        let input = Tensor::< 2>::ones([3, 3], &device);

        let output = model.forward(input);

        let expected =
            Tensor::< 1, Int>::from_data(TensorData::from([1, 2, 3]), &device);

        assert_eq!(output.to_data(), expected.to_data());
    }

    #[test]
    fn integer_full_precision() {
        let device = Default::default();
        let mut model = Net::init(&device);
        let mut store = PytorchStore::from_file("tests/integer/integer.pt");
        model
            .load_from(&mut store)
            .expect("Should decode state successfully");

        integer(model);
    }

    #[test]
    fn integer_half_precision() {
        let device = Default::default();
        let mut model = Net::init(&device);
        let mut store = PytorchStore::from_file("tests/integer/integer.pt");
        model
            .load_from(&mut store)
            .expect("Should decode state successfully");

        integer(model);
    }
}
