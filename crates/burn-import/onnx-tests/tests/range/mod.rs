use crate::include_models;
include_models!(range);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::TensorData;

    use crate::backend::TestBackend;

    #[test]
    fn range() {
        let device = Default::default();
        let model: range::Model<TestBackend> = range::Model::new(&device);

        // Run the model
        let start = 0i64;
        let limit = 10i64;
        let delta = 2i64;
        let output = model.forward(start, limit, delta);

        let expected = TensorData::from([0i64, 2, 4, 6, 8]);
        output.to_data().assert_eq(&expected, true);
    }
}
