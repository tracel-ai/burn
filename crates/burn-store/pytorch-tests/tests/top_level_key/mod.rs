use burn::{module::Module, nn::conv::Conv2d, tensor::backend::Backend};

#[derive(Module, Debug)]
#[allow(unused)]
pub struct Net<B: Backend> {
    conv1: Conv2d<B>,
}

#[cfg(test)]
mod tests {
    use crate::backend::TestBackend;

    use burn::nn::conv::Conv2dConfig;
    use burn_store::{ModuleSnapshot, PytorchStore};

    use super::*;

    impl<B: Backend> Net<B> {
        pub fn init(device: &B::Device) -> Self {
            Self {
                conv1: Conv2dConfig::new([2, 2], [2, 2]).init(device),
            }
        }
    }

    #[test]
    #[should_panic]
    fn should_fail_if_not_found() {
        let device = Default::default();
        let mut model = Net::<TestBackend>::init(&device);
        let mut store = PytorchStore::from_file("tests/top_level_key/top_level_key.pt");
        model
            .load_from(&mut store)
            .expect("Should decode state successfully");
    }

    #[test]
    fn should_load() {
        let device = Default::default();
        let mut model = Net::<TestBackend>::init(&device);
        let mut store = PytorchStore::from_file("tests/top_level_key/top_level_key.pt")
            .with_top_level_key("my_state_dict");

        model
            .load_from(&mut store)
            .expect("Should decode state successfully");
    }
}
