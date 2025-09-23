use burn::{module::Module, nn::conv::Conv2d, tensor::backend::Backend};

#[derive(Module, Debug)]
#[allow(unused)]
pub struct Net<B: Backend> {
    do_not_exist_in_pt: Conv2d<B>,
}

#[cfg(test)]
mod tests {
    use crate::backend::TestBackend;

    use burn::record::{FullPrecisionSettings, Recorder};
    use burn_import::pytorch::PyTorchFileRecorder;

    use super::*;

    #[test]
    #[should_panic(
        expected = "Missing source values for the 'do_not_exist_in_pt' field of type 'Conv2dRecordItem'. Please verify the source data and ensure the field name is correct"
    )]
    fn should_fail_if_struct_field_is_missing() {
        let device = Default::default();
        let _record: NetRecord<TestBackend> =
            PyTorchFileRecorder::<FullPrecisionSettings>::default()
                .load(
                    "tests/missing_module_field/missing_module_field.pt".into(),
                    &device,
                )
                .expect("Should decode state successfully");
    }
}
