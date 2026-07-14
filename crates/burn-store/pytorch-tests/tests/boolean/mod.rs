use burn::{
    module::{Module, Param, ParamId},
    tensor::{Bool, Device, Tensor, TensorData},
};

#[derive(Module, Debug)]
pub struct Net {
    buffer: Param<Tensor<1, Bool>>,
}

impl Net {
    /// Create a new model with placeholder values.
    pub fn init(device: &Device) -> Self {
        Self {
            buffer: Param::initialized(
                ParamId::new(),
                Tensor::from_bool(TensorData::from([false, false, false]), device),
            ),
        }
    }

    /// Forward pass of the model.
    pub fn forward(&self, _x: Tensor<2>) -> Tensor<1, Bool> {
        self.buffer.val()
    }
}

#[cfg(test)]
mod tests {

    use burn::tensor::TensorData;
    use burn_store::{ModuleSnapshot, PytorchStore};

    use super::*;

    #[test]
    fn boolean() {
        let device = Default::default();
        let mut model = Net::init(&device);
        let mut store = PytorchStore::from_file("tests/boolean/boolean.pt");
        model
            .load_from(&mut store)
            .expect("Should decode state successfully");

        let input = Tensor::<2>::ones([3, 3], &device);

        let output = model.forward(input);

        let expected = Tensor::<1, Bool>::from_bool(TensorData::from([true, false, true]), &device);

        assert_eq!(output.to_data(), expected.to_data());
    }
}
