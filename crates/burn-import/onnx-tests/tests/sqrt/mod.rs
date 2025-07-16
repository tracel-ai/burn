// Import the shared macro
use crate::include_models;
include_models!(sqrt);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Tensor, TensorData};

    use crate::backend::Backend;

    #[test]
    fn sqrt() {
        let device = Default::default();
        let model: sqrt::Model<Backend> = sqrt::Model::new(&device);

        let input1 = Tensor::<Backend, 4>::from_floats([[[[1.0, 4.0, 9.0, 25.0]]]], &device);
        let input2 = 36f64;

        let (output1, output2) = model.forward(input1, input2);
        let expected1 = TensorData::from([[[[1.0f32, 2.0, 3.0, 5.0]]]]);
        let expected2 = 6.0;

        output1.to_data().assert_eq(&expected1, true);
        assert_eq!(output2, expected2);
    }
}
