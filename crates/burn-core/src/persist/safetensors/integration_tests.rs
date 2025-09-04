//! Integration tests demonstrating the SafeTensors persister API

#[cfg(test)]
mod tests {
    use crate as burn;
    use crate::TestBackend;
    use crate::module::{Module, Param};
    use crate::persist::safetensors::SafetensorsPersister;
    use crate::persist::safetensors::SafetensorsPersisterConfig;
    use crate::persist::{ModulePersist, PathFilter};
    use crate::tensor::backend::Backend;
    use burn_tensor::Tensor;

    #[derive(Module, Debug)]
    struct TestModel<B: Backend> {
        encoder: EncoderModule<B>,
        decoder: DecoderModule<B>,
        head: HeadModule<B>,
    }

    #[derive(Module, Debug)]
    struct EncoderModule<B: Backend> {
        layer1: LinearLayer<B>,
        layer2: LinearLayer<B>,
        norm: NormLayer<B>,
    }

    #[derive(Module, Debug)]
    struct DecoderModule<B: Backend> {
        layer1: LinearLayer<B>,
        layer2: LinearLayer<B>,
        norm: NormLayer<B>,
    }

    #[derive(Module, Debug)]
    struct HeadModule<B: Backend> {
        weight: Param<Tensor<B, 2>>,
        bias: Param<Tensor<B, 1>>,
    }

    #[derive(Module, Debug)]
    struct LinearLayer<B: Backend> {
        weight: Param<Tensor<B, 2>>,
        bias: Param<Tensor<B, 1>>,
    }

    #[derive(Module, Debug)]
    struct NormLayer<B: Backend> {
        scale: Param<Tensor<B, 1>>,
        shift: Param<Tensor<B, 1>>,
    }

    impl<B: Backend> TestModel<B> {
        fn new(device: &B::Device) -> Self {
            Self {
                encoder: EncoderModule::new(device),
                decoder: DecoderModule::new(device),
                head: HeadModule::new(device),
            }
        }
    }

    impl<B: Backend> EncoderModule<B> {
        fn new(device: &B::Device) -> Self {
            Self {
                layer1: LinearLayer::new(device, 1),
                layer2: LinearLayer::new(device, 2),
                norm: NormLayer::new(device),
            }
        }
    }

    impl<B: Backend> DecoderModule<B> {
        fn new(device: &B::Device) -> Self {
            Self {
                layer1: LinearLayer::new(device, 3),
                layer2: LinearLayer::new(device, 4),
                norm: NormLayer::new(device),
            }
        }
    }

    impl<B: Backend> HeadModule<B> {
        fn new(device: &B::Device) -> Self {
            Self {
                weight: Param::from_data([[5.0, 6.0], [7.0, 8.0]], device),
                bias: Param::from_data([9.0, 10.0], device),
            }
        }
    }

    impl<B: Backend> LinearLayer<B> {
        fn new(device: &B::Device, seed: i32) -> Self {
            let weight_data = [
                [seed as f32, (seed + 1) as f32],
                [(seed + 2) as f32, (seed + 3) as f32],
            ];
            let bias_data = [(seed + 4) as f32, (seed + 5) as f32];

            Self {
                weight: Param::from_data(weight_data, device),
                bias: Param::from_data(bias_data, device),
            }
        }
    }

    impl<B: Backend> NormLayer<B> {
        fn new(device: &B::Device) -> Self {
            Self {
                scale: Param::from_data([1.0, 2.0], device),
                shift: Param::from_data([0.1, 0.2], device),
            }
        }
    }

    #[test]
    fn test_new_api_basic_usage() {
        let device = Default::default();
        let model = TestModel::<TestBackend>::new(&device);

        // Save using new API
        let mut save_persister = SafetensorsPersisterConfig::new()
            .with_metadata("framework", "burn")
            .with_metadata("version", "0.19.0")
            .build_memory();

        // Use collect_to method
        model.collect_to(&mut save_persister).unwrap();

        // Load using new API
        let mut load_persister = SafetensorsPersisterConfig::new().build_memory();
        if let SafetensorsPersister::Memory(ref mut p) = load_persister {
            if let SafetensorsPersister::Memory(ref p_save) = save_persister {
                p.set_data(p_save.data().unwrap().as_ref().clone());
            }
        }

        let mut target_model = TestModel::<TestBackend>::new(&device);
        let result = target_model.apply_from(&mut load_persister).unwrap();

        assert!(result.is_success());
        assert_eq!(result.applied.len(), 14); // All tensors should be applied
        assert_eq!(result.errors.len(), 0);
        assert_eq!(result.unused.len(), 0);
    }

    #[test]
    #[cfg(target_has_atomic = "ptr")]
    fn test_new_api_with_filtering() {
        let device = Default::default();
        let model = TestModel::<TestBackend>::new(&device);

        // Save only encoder tensors
        let mut save_persister = SafetensorsPersisterConfig::new()
            .with_filter(PathFilter::new().with_regex(r"^encoder\..*"))
            .with_metadata("subset", "encoder_only")
            .build_memory();

        model.collect_to(&mut save_persister).unwrap();

        // Load into new model - need to allow partial loading since we only saved encoder tensors
        let mut load_persister = SafetensorsPersisterConfig::new()
            .allow_partial(true)
            .build_memory();
        if let SafetensorsPersister::Memory(ref mut p) = load_persister {
            if let SafetensorsPersister::Memory(ref p_save) = save_persister {
                p.set_data(p_save.data().unwrap().as_ref().clone());
            }
        }

        let mut target_model = TestModel::<TestBackend>::new(&device);
        let result = target_model.apply_from(&mut load_persister).unwrap();

        // Only encoder tensors should be applied
        assert_eq!(result.applied.len(), 6); // encoder has 6 tensors (2 layers × 2 + norm × 2)

        // Check that only encoder tensors were applied
        for tensor_name in &result.applied {
            assert!(tensor_name.starts_with("encoder."));
        }
    }
}
