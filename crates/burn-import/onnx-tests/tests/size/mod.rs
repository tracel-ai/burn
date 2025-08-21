// Import the shared macro
use crate::include_models;
include_models!(size);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Tensor, TensorData};

use crate::backend::TestBackend;

    #[test]
    fn size() {
        let model: size::Model<TestBackend> = size::Model::default();
        let device = Default::default();

        let input =
            Tensor::<TestBackend, 1>::arange(0..(1 * 2 * 3 * 4 * 5), &device).reshape([1, 2, 3, 4, 5]);
        let output = model.forward(input);
        let expected = TensorData::from([120]);

        output.to_data().assert_eq(&expected, true);
    }
}
