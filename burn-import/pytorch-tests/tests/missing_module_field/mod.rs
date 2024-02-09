use burn::{module::Module, nn::conv::Conv2d, tensor::backend::Backend};

#[derive(Module, Debug)]
pub struct Net<B: Backend> {
    do_not_exist_in_pt: Conv2d<B>,
}

#[cfg(test)]
mod tests {
    type Backend = burn_ndarray::NdArray<f32>;

    use burn::record::{FullPrecisionSettings, Recorder};
    use burn_import::pytorch::PyTorchFileRecorder;

    use super::*;

    #[test]
    #[should_panic(
        expected = "Cannot deserialize Conv2dRecordItem struct with default values orinated from 'do_not_exist_in_pt' field"
    )]
    fn should_fail_if_struct_field_is_missing() {
        let device = Default::default();
        let _record: NetRecord<Backend> = PyTorchFileRecorder::<FullPrecisionSettings>::default()
            .load(
                "tests/missing_module_field/missing_module_field.pt".into(),
                &device,
            )
            .expect("Should decode state successfully");
    }
}
