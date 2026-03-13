#[cfg(feature = "std")]
mod tests {
    use burn::{
        module::{Module, Param},
        record::{
            BinFileRecorder, DefaultFileRecorder, FileRecorder, FullPrecisionSettings,
            PrettyJsonFileRecorder, RecorderError,
        },
    };
    use burn_core as burn;
    use burn_ndarray::NdArrayDevice;
    use burn_tensor::{Tensor, backend::Backend};
    use std::path::PathBuf;

    type TestBackend = burn_ndarray::NdArray<f32>;

    /// Simple linear module.
    #[derive(Module, Debug)]
    pub struct Linear<B: Backend> {
        pub weight: Param<Tensor<B, 2>>,
        pub bias: Option<Param<Tensor<B, 1>>>,
    }

    impl<B: Backend> Linear<B> {
        pub fn new(in_features: usize, out_features: usize, device: &B::Device) -> Self {
            let weight = Tensor::random(
                [out_features, in_features],
                burn_tensor::Distribution::Default,
                device,
            );
            let bias = Tensor::random([out_features], burn_tensor::Distribution::Default, device);

            Self {
                weight: Param::from_tensor(weight),
                bias: Some(Param::from_tensor(bias)),
            }
        }
    }

    #[derive(Module, Debug)]
    pub struct Model<B: Backend> {
        single_const: f32,
        linear1: Linear<B>,
        array_const: [usize; 2],
        linear2: Linear<B>,
        array_lin: [Linear<B>; 2],
    }

    #[derive(Module, Debug)]
    pub struct ModelNewOptionalField<B: Backend> {
        single_const: f32,
        linear1: Linear<B>,
        array_const: [usize; 2],
        linear2: Linear<B>,
        array_lin: [Linear<B>; 2],
        new_field: Option<usize>,
    }

    #[derive(Module, Debug)]
    pub struct ModelNewConstantField<B: Backend> {
        single_const: f32,
        linear1: Linear<B>,
        array_const: [usize; 2],
        linear2: Linear<B>,
        array_lin: [Linear<B>; 2],
        new_field: usize,
    }

    #[derive(Module, Debug)]
    #[allow(unused)]
    pub struct ModelNewFieldOrders<B: Backend> {
        array_const: [usize; 2],
        linear2: Linear<B>,
        single_const: f32,
        array_lin: [Linear<B>; 2],
        linear1: Linear<B>,
    }

    #[test]
    fn deserialize_with_new_optional_field_works_with_default_file_recorder() {
        deserialize_with_new_optional_field(
            "default",
            DefaultFileRecorder::<FullPrecisionSettings>::new(),
        )
        .unwrap();
    }

    #[test]
    fn deserialize_with_removed_optional_field_works_with_default_file_recorder() {
        deserialize_with_removed_optional_field(
            "default",
            DefaultFileRecorder::<FullPrecisionSettings>::new(),
        )
        .unwrap();
    }

    #[test]
    fn deserialize_with_new_constant_field_works_with_default_file_recorder() {
        deserialize_with_new_constant_field(
            "default",
            DefaultFileRecorder::<FullPrecisionSettings>::new(),
        )
        .unwrap();
    }

    #[test]
    fn deserialize_with_removed_constant_field_works_with_default_file_recorder() {
        deserialize_with_removed_constant_field(
            "default",
            DefaultFileRecorder::<FullPrecisionSettings>::new(),
        )
        .unwrap();
    }

    #[test]
    fn deserialize_with_new_field_order_works_with_default_file_recorder() {
        deserialize_with_new_field_order(
            "default",
            DefaultFileRecorder::<FullPrecisionSettings>::new(),
        )
        .unwrap();
    }
    #[test]
    fn deserialize_with_new_optional_field_works_with_pretty_json() {
        deserialize_with_new_optional_field(
            "pretty-json",
            PrettyJsonFileRecorder::<FullPrecisionSettings>::new(),
        )
        .unwrap();
    }

    #[test]
    fn deserialize_with_removed_optional_field_works_with_pretty_json() {
        deserialize_with_removed_optional_field(
            "pretty-json",
            PrettyJsonFileRecorder::<FullPrecisionSettings>::new(),
        )
        .unwrap();
    }

    #[test]
    fn deserialize_with_new_constant_field_works_with_pretty_json() {
        deserialize_with_new_constant_field(
            "pretty-json",
            PrettyJsonFileRecorder::<FullPrecisionSettings>::new(),
        )
        .unwrap();
    }

    #[test]
    fn deserialize_with_removed_constant_field_works_with_pretty_json() {
        deserialize_with_removed_constant_field(
            "pretty-json",
            PrettyJsonFileRecorder::<FullPrecisionSettings>::new(),
        )
        .unwrap();
    }

    #[test]
    fn deserialize_with_new_field_order_works_with_pretty_json() {
        deserialize_with_new_field_order(
            "pretty-json",
            PrettyJsonFileRecorder::<FullPrecisionSettings>::new(),
        )
        .unwrap();
    }

    #[test]
    fn deserialize_with_new_optional_field_works_with_bin_file_recorder() {
        deserialize_with_new_optional_field("bin", BinFileRecorder::<FullPrecisionSettings>::new())
            .unwrap();
    }

    #[test]
    fn deserialize_with_removed_optional_field_works_with_bin_file_recorder() {
        deserialize_with_removed_optional_field(
            "bin",
            BinFileRecorder::<FullPrecisionSettings>::new(),
        )
        .unwrap();
    }

    #[test]
    fn deserialize_with_new_constant_field_works_with_bin_file_recorder() {
        deserialize_with_new_constant_field("bin", BinFileRecorder::<FullPrecisionSettings>::new())
            .unwrap();
    }

    #[test]
    fn deserialize_with_removed_constant_field_works_with_bin_file_recorder() {
        deserialize_with_removed_constant_field(
            "bin",
            BinFileRecorder::<FullPrecisionSettings>::new(),
        )
        .unwrap();
    }

    #[test]
    fn deserialize_with_new_field_order_works_with_bin_file_recorder() {
        deserialize_with_new_field_order("bin", BinFileRecorder::<FullPrecisionSettings>::new())
            .unwrap();
    }

    #[inline(always)]
    fn file_path(filename: String) -> PathBuf {
        std::env::temp_dir().join(filename)
    }

    #[test]
    fn test_tensor_serde() {
        let tensor: burn_tensor::Tensor<TestBackend, 1> =
            burn_tensor::Tensor::ones([1], &NdArrayDevice::default());
        let encoded = serde_json::to_string(&tensor).unwrap();
        let decoded: burn_tensor::Tensor<TestBackend, 1> = serde_json::from_str(&encoded).unwrap();
        assert_eq!(tensor.into_data(), decoded.into_data());
    }

    fn deserialize_with_new_optional_field<R>(name: &str, recorder: R) -> Result<(), RecorderError>
    where
        R: FileRecorder<TestBackend>,
    {
        let device = Default::default();
        let file_path: PathBuf = file_path(format!("deserialize_with_new_optional_field-{name}"));
        let model = Model {
            single_const: 32.0,
            linear1: Linear::<TestBackend>::new(20, 20, &device),
            array_const: [2, 2],
            linear2: Linear::<TestBackend>::new(20, 20, &device),
            array_lin: [
                Linear::<TestBackend>::new(20, 20, &device),
                Linear::<TestBackend>::new(20, 20, &device),
            ],
        };

        recorder
            .record(model.into_record(), file_path.clone())
            .unwrap();
        let result =
            recorder.load::<ModelNewOptionalFieldRecord<TestBackend>>(file_path.clone(), &device);
        std::fs::remove_file(file_path).ok();

        result?;
        Ok(())
    }

    fn deserialize_with_removed_optional_field<R>(
        name: &str,
        recorder: R,
    ) -> Result<(), RecorderError>
    where
        R: FileRecorder<TestBackend>,
    {
        let device = Default::default();
        let file_path: PathBuf =
            file_path(format!("deserialize_with_removed_optional_field-{name}"));
        let model = ModelNewOptionalField {
            single_const: 32.0,
            linear1: Linear::<TestBackend>::new(20, 20, &device),
            array_const: [2, 2],
            linear2: Linear::<TestBackend>::new(20, 20, &device),
            array_lin: [
                Linear::<TestBackend>::new(20, 20, &device),
                Linear::<TestBackend>::new(20, 20, &device),
            ],
            new_field: None,
        };

        recorder
            .record(model.into_record(), file_path.clone())
            .unwrap();
        let result = recorder.load::<ModelRecord<TestBackend>>(file_path.clone(), &device);
        std::fs::remove_file(file_path).ok();

        result?;
        Ok(())
    }

    fn deserialize_with_new_constant_field<R>(name: &str, recorder: R) -> Result<(), RecorderError>
    where
        R: FileRecorder<TestBackend>,
    {
        let device = Default::default();
        let file_path: PathBuf = file_path(format!("deserialize_with_new_constant_field-{name}"));
        let model = Model {
            single_const: 32.0,
            array_const: [2, 2],
            linear1: Linear::<TestBackend>::new(20, 20, &device),
            linear2: Linear::<TestBackend>::new(20, 20, &device),
            array_lin: [
                Linear::<TestBackend>::new(20, 20, &device),
                Linear::<TestBackend>::new(20, 20, &device),
            ],
        };

        recorder
            .record(model.into_record(), file_path.clone())
            .unwrap();
        let result =
            recorder.load::<ModelNewConstantFieldRecord<TestBackend>>(file_path.clone(), &device);
        std::fs::remove_file(file_path).ok();

        result?;
        Ok(())
    }

    fn deserialize_with_removed_constant_field<R>(
        name: &str,
        recorder: R,
    ) -> Result<(), RecorderError>
    where
        R: FileRecorder<TestBackend>,
    {
        let device = Default::default();
        let file_path: PathBuf =
            file_path(format!("deserialize_with_removed_constant_field-{name}"));
        let model = ModelNewConstantField {
            single_const: 32.0,
            array_const: [2, 2],
            linear1: Linear::<TestBackend>::new(20, 20, &device),
            linear2: Linear::<TestBackend>::new(20, 20, &device),
            array_lin: [
                Linear::<TestBackend>::new(20, 20, &device),
                Linear::<TestBackend>::new(20, 20, &device),
            ],
            new_field: 0,
        };

        recorder
            .record(model.into_record(), file_path.clone())
            .unwrap();
        let result = recorder.load::<ModelRecord<TestBackend>>(file_path.clone(), &device);
        std::fs::remove_file(file_path).ok();

        result?;
        Ok(())
    }

    fn deserialize_with_new_field_order<R>(name: &str, recorder: R) -> Result<(), RecorderError>
    where
        R: FileRecorder<TestBackend>,
    {
        let device = Default::default();
        let file_path: PathBuf = file_path(format!("deserialize_with_new_field_order-{name}"));
        let model = Model {
            array_const: [2, 2],
            single_const: 32.0,
            linear1: Linear::<TestBackend>::new(20, 20, &device),
            linear2: Linear::<TestBackend>::new(20, 20, &device),
            array_lin: [
                Linear::<TestBackend>::new(20, 20, &device),
                Linear::<TestBackend>::new(20, 20, &device),
            ],
        };

        recorder
            .record(model.into_record(), file_path.clone())
            .unwrap();

        let result =
            recorder.load::<ModelNewFieldOrdersRecord<TestBackend>>(file_path.clone(), &device);
        std::fs::remove_file(file_path).ok();

        result?;
        Ok(())
    }
}
