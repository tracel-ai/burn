#[cfg(feature = "std")]
mod tests {
    use burn_core as burn;

    use burn::{module::Module, nn};
    use burn_tensor::backend::Backend;
    

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

    //     #[test]
    //     fn deserialize_with_new_optional_field_works_with_json() {
    //         deserialize_with_new_optional_field("default").unwrap();
    //     }
    //
    //     fn deserialize_with_new_optional_field<R1, R2>(
    //         name: &str,
    //         recorder1: R1,
    //         recorder2: R2,
    //     ) -> Result<(), RecorderError>
    //     where
    //         R1: FileRecorder<ModelRecord<TestBackend>>,
    //         R2: FileRecorder<ModelNewOptionalFieldRecord<TestBackend>>,
    //     {
    //         let file_path: PathBuf = format!("/tmp/deserialize_with_new_optional_field-{name}").into();
    //         let model = Model {
    //             linear1: nn::LinearConfig::new(20, 20).init::<TestBackend>(),
    //             linear2: nn::LinearConfig::new(20, 20).init::<TestBackend>(),
    //         };
    //
    //         let record = model.into_record();
    //         record
    //             .record::<DefaultRecordSettings>(file_path.clone())
    //             .unwrap();
    //         let result = ModelNewOptionalFieldRecord::<TestBackend>::load::<DefaultRecordSettings>(
    //             file_path.clone(),
    //         );
    //         std::fs::remove_file(file_path).ok();
    //
    //         result?;
    //         Ok(())
    //     }
    //
    //     fn deserialize_with_new_field_order<R1, R2>(
    //         name: &str,
    //         recorder1: R1,
    //         recorder2: R2,
    //     ) -> Result<(), RecorderError>
    //     where
    //         R1: FileRecorder<ModelRecord<TestBackend>>,
    //         R2: FileRecorder<ModelNewOptionalFieldRecord<TestBackend>>,
    //     {
    //         let file_path: PathBuf = format!("/tmp/deserialize_with_new_field_order-{name}").into();
    //         let model = Model {
    //             linear1: nn::LinearConfig::new(20, 20).init::<TestBackend>(),
    //             linear2: nn::LinearConfig::new(20, 20).init::<TestBackend>(),
    //         };
    //
    //         let record = model.into_record();
    //         record.record::<S>(file_path.clone()).unwrap();
    //         let result = ModelNewFieldOrdersRecord::<TestBackend>::load::<S>(file_path.clone());
    //         std::fs::remove_file(file_path).ok();
    //
    //         result?;
    //         Ok(())
    //     }
}
