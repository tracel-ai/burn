// Import the shared macro
use crate::include_models;
include_models!(not);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Bool, Tensor, TensorData};

    use crate::backend::Backend;

    #[test]
    fn not() {
        let device = Default::default();
        let model: not::Model<Backend> = not::Model::new(&device);

        let input = Tensor::<Backend, 4, Bool>::from_bool(
            TensorData::from([[[[true, false, true, false]]]]),
            &device,
        );

        let output = model.forward(input).to_data();
        let expected = TensorData::from([[[[false, true, false, true]]]]);

        output.assert_eq(&expected, true);
    }
}
