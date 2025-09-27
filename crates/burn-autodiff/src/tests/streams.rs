#[burn_tensor_testgen::testgen(streams)]
mod tests {
    use super::*;
    use burn_tensor::StreamId;
    use burn_tensor::TensorData;
    use burn_tensor::{Tolerance, ops::FloatElem};
    type FT = FloatElem<TestBackend>;

    #[test]
    fn detach_should_reset_stream_id() {
        let data_1 = TensorData::from([[0.0, 1.0], [3.0, 4.0]]);
        let data_2 = TensorData::from([[6.0, 7.0], [9.0, 10.0]]);

        let stream_id = StreamId::current();
        let device = Default::default();
        let tensor_1 = TestAutodiffTensor::<2>::from_data(data_1, &device).require_grad();

        let tensor_1_primitive = tensor_1.clone().into_primitive().tensor();
        assert_eq!(tensor_1_primitive.node.stream, stream_id);

        std::thread::spawn(move || {
            let tensor_1 = tensor_1.detach();
            let tensor_1_primitive = tensor_1.into_primitive().tensor();
            assert_ne!(tensor_1_primitive.node.stream, stream_id);
        })
        .join()
        .unwrap();
    }
}
