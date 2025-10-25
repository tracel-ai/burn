use burn::{module::Module, nn::conv::Conv2d, tensor::backend::Backend};

#[derive(Module, Debug)]
#[allow(unused)]
pub struct Net<B: Backend> {
    conv1: Conv2d<B>,
}

#[cfg(test)]
mod tests {
    use crate::backend::TestBackend;

    use burn::record::{FullPrecisionSettings, Recorder};
    use burn_import::pytorch::{LoadArgs, PyTorchFileRecorder};

    use super::*;

    #[test]
    #[should_panic]
    fn should_fail_if_not_found() {
        let device = Default::default();
        let _record: NetRecord<TestBackend> =
            PyTorchFileRecorder::<FullPrecisionSettings>::default()
                .load("tests/top_level_key/top_level_key.pt".into(), &device)
                .expect("Should decode state successfully");
    }

    #[test]
    fn should_load() {
        let device = Default::default();
        let load_args = LoadArgs::new("tests/top_level_key/top_level_key.pt".into())
            .with_top_level_key("my_state_dict");

        let _record: NetRecord<TestBackend> =
            PyTorchFileRecorder::<FullPrecisionSettings>::default()
                .load(load_args, &device)
                .expect("Should decode state successfully");
    }
}
