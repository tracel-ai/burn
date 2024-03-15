#[burn_tensor_testgen::testgen(dyn_data_roundtrip)]
mod tests {
    use super::*;
    use burn_tensor::{Data, Tensor};

    #[test]
    fn should_support_dyn_data_roundtrip_float() {
        let dyn_data = DynData::Float(Data::from([[0.0, -1.0, 2.0], [3.0, 4.0, -5.0]]).into());

        let roundtrip_data = TestBackend::dyn_into_data(TestBackend::dyn_from_data(dyn_data.clone(), &Default::default()));

        assert_eq!(dyn_data, roundtrip_data);
    }

    #[test]
    fn should_support_dyn_data_roundtrip_int() {
        let dyn_data = DynData::Int(Data::from([[0, -1, 2], [3, 4, -5]]).into());

        let roundtrip_data = TestBackend::dyn_into_data(TestBackend::dyn_from_data(dyn_data.clone(), &Default::default()));

        assert_eq!(dyn_data, roundtrip_data);
    }

    #[test]
    fn should_support_dyn_data_roundtrip_bool() {
        let dyn_data = DynData::Bool(Data::from([[false, false, false], [true, true, false]]).into());

        let roundtrip_data = TestBackend::dyn_into_data(TestBackend::dyn_from_data(dyn_data.clone(), &Default::default()));

        assert_eq!(dyn_data, roundtrip_data);
    }
}
