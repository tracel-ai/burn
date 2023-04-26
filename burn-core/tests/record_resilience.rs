#[cfg(feature = "std")]
mod tests {
    use burn_core as burn;

    use burn::{
        module::Module,
        nn,
        record::{
            CompactRecordSettings, DefaultRecordSettings, FileRecorder, Record, RecordSettings,
            RecorderError, SentitiveCompactRecordSettings,
        },
    };
    use burn_tensor::backend::Backend;
    use std::path::PathBuf;

    type TestBackend = burn_ndarray::NdArrayBackend<f32>;

    #[derive(Module, Debug)]
    pub struct Model<B: Backend> {
        linear1: nn::Linear<B>,
        linear2: nn::Linear<B>,
    }

    #[derive(Module, Debug)]
    pub struct ModelNewOptionalField<B: Backend> {
        linear1: nn::Linear<B>,
        linear2: nn::Linear<B>,
        new_field: Option<usize>,
    }

    #[derive(Module, Debug)]
    pub struct ModelNewFieldOrders<B: Backend> {
        linear2: nn::Linear<B>,
        linear1: nn::Linear<B>,
    }

    #[test]
    fn deserialize_with_new_optional_field_works_with_default_settings() {
        deserialize_with_new_optional_field::<DefaultRecordSettings>("default").unwrap();
    }

    #[test]
    fn deserialize_with_new_field_order_works_with_default_settings() {
        deserialize_with_new_field_order::<DefaultRecordSettings>("default").unwrap();
    }

    #[test]
    fn deserialize_with_new_optional_field_works_with_compact_settings() {
        deserialize_with_new_optional_field::<CompactRecordSettings>("compact").unwrap();
    }

    #[test]
    fn deserialize_with_new_field_order_works_with_compact_settings() {
        deserialize_with_new_field_order::<CompactRecordSettings>("compact").unwrap();
    }

    #[test]
    #[should_panic]
    fn deserialize_with_new_optional_field_fails_with_sentitive_compact_settings() {
        deserialize_with_new_optional_field::<SentitiveCompactRecordSettings>("sensitive").unwrap();
    }

    #[test]
    fn deserialize_with_new_field_order_works_with_sensitive_compact_settings() {
        deserialize_with_new_field_order::<SentitiveCompactRecordSettings>("sensitive").unwrap();
    }

    fn deserialize_with_new_optional_field<S>(name: &str) -> Result<(), RecorderError>
    where
        S: RecordSettings,
        S::Recorder: FileRecorder,
    {
        let file_path: PathBuf = format!("/tmp/deserialize_with_new_optional_field-{name}").into();
        let model = Model {
            linear1: nn::LinearConfig::new(20, 20).init::<TestBackend>(),
            linear2: nn::LinearConfig::new(20, 20).init::<TestBackend>(),
        };

        let record = model.into_record();
        record.record::<S>(file_path.clone()).unwrap();
        let result = ModelNewOptionalFieldRecord::<TestBackend>::load::<S>(file_path.clone());
        std::fs::remove_file(file_path).ok();

        result?;
        Ok(())
    }

    fn deserialize_with_new_field_order<S>(name: &str) -> Result<(), RecorderError>
    where
        S: RecordSettings,
        S::Recorder: FileRecorder,
    {
        let file_path: PathBuf = format!("/tmp/deserialize_with_new_field_order-{name}").into();
        let model = Model {
            linear1: nn::LinearConfig::new(20, 20).init::<TestBackend>(),
            linear2: nn::LinearConfig::new(20, 20).init::<TestBackend>(),
        };

        let record = model.into_record();
        record.record::<S>(file_path.clone()).unwrap();
        let result = ModelNewFieldOrdersRecord::<TestBackend>::load::<S>(file_path.clone());
        std::fs::remove_file(file_path).ok();

        result?;
        Ok(())
    }
}
