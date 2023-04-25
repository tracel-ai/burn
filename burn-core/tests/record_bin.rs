use burn_core as burn;

use burn::{
    module::Module,
    nn,
    record::{CompactRecordSettings, DebugRecordSettings, DefaultRecordSettings, Record},
};
use burn_tensor::backend::Backend;
use std::path::PathBuf;

type TestBackend = burn_ndarray::NdArrayBackend<f32>;

#[derive(Module, Debug)]
pub struct Model1<B: Backend> {
    linear1: nn::Linear<B>,
    linear2: nn::Linear<B>,
}

#[derive(Module, Debug)]
pub struct ModelNewField<B: Backend> {
    linear1: nn::Linear<B>,
    linear2: nn::Linear<B>,
    new_field: Option<usize>,
}

#[derive(Module, Debug)]
pub struct ModelFieldOrders<B: Backend> {
    linear2: nn::Linear<B>,
    linear1: nn::Linear<B>,
}

#[test]
fn should_be_able_to_deserialize_with_new_option_field() {
    let file_path: PathBuf = "/tmp/allo".into();
    let model1 = Model1 {
        linear1: nn::LinearConfig::new(20, 20).init::<TestBackend>(),
        linear2: nn::LinearConfig::new(20, 20).init::<TestBackend>(),
    };

    let record = model1.into_record();
    record
        .record::<DefaultRecordSettings>(file_path.clone())
        .unwrap();

    let _record =
        ModelNewFieldRecord::<TestBackend>::load::<DefaultRecordSettings>(file_path.clone())
            .unwrap();
}

#[test]
fn should_be_able_to_deserialize_with_new_field_order() {
    let file_path: PathBuf = "/tmp/allo".into();
    let model1 = Model1 {
        linear1: nn::LinearConfig::new(20, 20).init::<TestBackend>(),
        linear2: nn::LinearConfig::new(20, 20).init::<TestBackend>(),
    };

    let record = model1.into_record();
    record
        .record::<DefaultRecordSettings>(file_path.clone())
        .unwrap();

    let _record =
        ModelFieldOrdersRecord::<TestBackend>::load::<DefaultRecordSettings>(file_path).unwrap();
}
