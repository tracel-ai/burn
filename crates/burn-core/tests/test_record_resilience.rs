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
    use burn_tensor::{Device, Tensor};
    use std::path::PathBuf;

    /// Simple linear module.
    #[derive(Module, Debug)]
    pub struct Linear {
        pub weight: Param<Tensor<2>>,
        pub bias: Option<Param<Tensor<1>>>,
    }

    impl Linear {
        pub fn new(in_features: usize, out_features: usize, device: &Device) -> Self {
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
    pub struct Model {
        single_const: f32,
        linear1: Linear,
        array_const: [usize; 2],
        linear2: Linear,
        array_lin: [Linear; 2],
    }

    #[derive(Module, Debug)]
    pub struct ModelNewOptionalField {
        single_const: f32,
        linear1: Linear,
        array_const: [usize; 2],
        linear2: Linear,
        array_lin: [Linear; 2],
        new_field: Option<usize>,
    }

    #[derive(Module, Debug)]
    pub struct ModelNewConstantField {
        single_const: f32,
        linear1: Linear,
        array_const: [usize; 2],
        linear2: Linear,
        array_lin: [Linear; 2],
        new_field: usize,
    }

    #[derive(Module, Debug)]
    #[allow(unused)]
    pub struct ModelNewFieldOrders {
        array_const: [usize; 2],
        linear2: Linear,
        single_const: f32,
        array_lin: [Linear; 2],
        linear1: Linear,
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
        let tensor = burn_tensor::Tensor::<1>::ones([1], &Default::default());
        let encoded = serde_json::to_string(&tensor).unwrap();
        let decoded: burn_tensor::Tensor<1> = serde_json::from_str(&encoded).unwrap();
        assert_eq!(tensor.into_data(), decoded.into_data());
    }

    fn deserialize_with_new_optional_field<R>(name: &str, recorder: R) -> Result<(), RecorderError>
    where
        R: FileRecorder,
    {
        let device = Default::default();
        let file_path: PathBuf = file_path(format!("deserialize_with_new_optional_field-{name}"));
        let model = Model {
            single_const: 32.0,
            linear1: Linear::new(20, 20, &device),
            array_const: [2, 2],
            linear2: Linear::new(20, 20, &device),
            array_lin: [Linear::new(20, 20, &device), Linear::new(20, 20, &device)],
        };

        recorder
            .record(model.into_record(), file_path.clone())
            .unwrap();
        let result = recorder.load::<ModelNewOptionalFieldRecord>(file_path.clone(), &device);
        std::fs::remove_file(file_path).ok();

        result?;
        Ok(())
    }

    fn deserialize_with_removed_optional_field<R>(
        name: &str,
        recorder: R,
    ) -> Result<(), RecorderError>
    where
        R: FileRecorder,
    {
        let device = Default::default();
        let file_path: PathBuf =
            file_path(format!("deserialize_with_removed_optional_field-{name}"));
        let model = ModelNewOptionalField {
            single_const: 32.0,
            linear1: Linear::new(20, 20, &device),
            array_const: [2, 2],
            linear2: Linear::new(20, 20, &device),
            array_lin: [Linear::new(20, 20, &device), Linear::new(20, 20, &device)],
            new_field: None,
        };

        recorder
            .record(model.into_record(), file_path.clone())
            .unwrap();
        let result = recorder.load::<ModelRecord>(file_path.clone(), &device);
        std::fs::remove_file(file_path).ok();

        result?;
        Ok(())
    }

    fn deserialize_with_new_constant_field<R>(name: &str, recorder: R) -> Result<(), RecorderError>
    where
        R: FileRecorder,
    {
        let device = Default::default();
        let file_path: PathBuf = file_path(format!("deserialize_with_new_constant_field-{name}"));
        let model = Model {
            single_const: 32.0,
            array_const: [2, 2],
            linear1: Linear::new(20, 20, &device),
            linear2: Linear::new(20, 20, &device),
            array_lin: [Linear::new(20, 20, &device), Linear::new(20, 20, &device)],
        };

        recorder
            .record(model.into_record(), file_path.clone())
            .unwrap();
        let result = recorder.load::<ModelNewConstantFieldRecord>(file_path.clone(), &device);
        std::fs::remove_file(file_path).ok();

        result?;
        Ok(())
    }

    fn deserialize_with_removed_constant_field<R>(
        name: &str,
        recorder: R,
    ) -> Result<(), RecorderError>
    where
        R: FileRecorder,
    {
        let device = Default::default();
        let file_path: PathBuf =
            file_path(format!("deserialize_with_removed_constant_field-{name}"));
        let model = ModelNewConstantField {
            single_const: 32.0,
            array_const: [2, 2],
            linear1: Linear::new(20, 20, &device),
            linear2: Linear::new(20, 20, &device),
            array_lin: [Linear::new(20, 20, &device), Linear::new(20, 20, &device)],
            new_field: 0,
        };

        recorder
            .record(model.into_record(), file_path.clone())
            .unwrap();
        let result = recorder.load::<ModelRecord>(file_path.clone(), &device);
        std::fs::remove_file(file_path).ok();

        result?;
        Ok(())
    }

    fn deserialize_with_new_field_order<R>(name: &str, recorder: R) -> Result<(), RecorderError>
    where
        R: FileRecorder,
    {
        let device = Default::default();
        let file_path: PathBuf = file_path(format!("deserialize_with_new_field_order-{name}"));
        let model = Model {
            array_const: [2, 2],
            single_const: 32.0,
            linear1: Linear::new(20, 20, &device),
            linear2: Linear::new(20, 20, &device),
            array_lin: [Linear::new(20, 20, &device), Linear::new(20, 20, &device)],
        };

        recorder
            .record(model.into_record(), file_path.clone())
            .unwrap();

        let result = recorder.load::<ModelNewFieldOrdersRecord>(file_path.clone(), &device);
        std::fs::remove_file(file_path).ok();

        result?;
        Ok(())
    }
}
