//! Integration tests demonstrating the new ModulePersister API

#[cfg(test)]
mod tests {
    use crate as burn;
    use crate::TestBackend;
    use crate::module::{Module, Param};
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
            .build_in_memory();

        // Use collect_to method
        model.collect_to(&mut save_persister).unwrap();

        // Load using new API
        let mut load_persister = SafetensorsPersisterConfig::new().build_in_memory();
        load_persister.set_data(save_persister.data().unwrap().to_vec());

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
            .build_in_memory();

        model.collect_to(&mut save_persister).unwrap();

        // Load into new model - need to allow partial loading since we only saved encoder tensors
        let mut load_persister = SafetensorsPersisterConfig::new()
            .allow_partial(true)
            .build_in_memory();
        load_persister.set_data(save_persister.data().unwrap().to_vec());

        let mut target_model = TestModel::<TestBackend>::new(&device);
        let result = target_model.apply_from(&mut load_persister).unwrap();

        // Only encoder tensors should be applied
        assert_eq!(result.applied.len(), 6); // encoder has 6 tensors (2 layers × 2 + norm × 2)

        // Check that only encoder tensors were applied
        for tensor_name in &result.applied {
            assert!(tensor_name.starts_with("encoder."));
        }
    }

    #[test]
    #[cfg(target_has_atomic = "ptr")]
    fn test_new_api_with_remapping() {
        let device = Default::default();
        let model = TestModel::<TestBackend>::new(&device);

        // Save with remapping - rename encoder to enc, decoder to dec
        let mut save_persister = SafetensorsPersisterConfig::new()
            .with_remapping(&[("encoder", "enc"), ("decoder", "dec")])
            .build_in_memory();

        model.collect_to(&mut save_persister).unwrap();

        // Load without remapping to see the effect
        let mut load_persister = SafetensorsPersisterConfig::new()
            .allow_partial(true)
            .build_in_memory();
        load_persister.set_data(save_persister.data().unwrap().to_vec());

        let mut target_model = TestModel::<TestBackend>::new(&device);
        let result = target_model.apply_from(&mut load_persister).unwrap();

        // With remapping during save, the tensors are renamed in storage
        // When loading without remapping, only tensors that were NOT renamed can be applied
        // In this case, only head tensors (which weren't remapped) should apply
        assert_eq!(result.applied.len(), 2); // Only head.weight and head.bias
        assert!(result.applied.contains(&"head.weight".to_string()));
        assert!(result.applied.contains(&"head.bias".to_string()));

        // The remapped tensors (enc.* and dec.*) won't match the model's encoder.* and decoder.* names
        assert_eq!(result.missing.len(), 12); // encoder (6) + decoder (6) tensors are missing

        // The unused tensors in storage are the renamed ones (enc.* and dec.*)
        assert_eq!(result.unused.len(), 12); // 12 tensors were renamed and thus unused

        // Check that unused tensors have remapped names
        let unused_names: Vec<&String> = result.unused.iter().collect();
        assert!(unused_names.iter().any(|name| name.starts_with("enc.")));
        assert!(unused_names.iter().any(|name| name.starts_with("dec.")));
    }

    #[test]
    fn test_new_api_with_predicate_filter() {
        let device = Default::default();
        let model = TestModel::<TestBackend>::new(&device);

        // Save only weight tensors using predicate
        fn weight_filter(path: &str, _container: &str) -> bool {
            path.ends_with(".weight")
        }

        let mut save_persister = SafetensorsPersisterConfig::new()
            .filter_by_predicate(weight_filter)
            .build_in_memory();

        model.collect_to(&mut save_persister).unwrap();

        // Load into new model
        let mut load_persister = SafetensorsPersisterConfig::new()
            .allow_partial(true)
            .build_in_memory();
        load_persister.set_data(save_persister.data().unwrap().to_vec());

        let mut target_model = TestModel::<TestBackend>::new(&device);
        let result = target_model.apply_from(&mut load_persister).unwrap();

        // Should only apply weight tensors
        assert_eq!(result.applied.len(), 5); // 3 linear weights + 1 head weight + norm doesn't have weight

        // All applied tensors should be weights
        for tensor_name in &result.applied {
            assert!(tensor_name.ends_with(".weight"));
        }
    }

    #[test]
    fn test_new_api_with_specific_tensor_names() {
        let device = Default::default();
        let model = TestModel::<TestBackend>::new(&device);

        // Save only specific tensors
        let mut save_persister = SafetensorsPersisterConfig::new()
            .filter_by_names(vec![
                "encoder.layer1.weight",
                "decoder.layer2.bias",
                "head.weight",
            ])
            .build_in_memory();

        model.collect_to(&mut save_persister).unwrap();

        // Load into new model
        let mut load_persister = SafetensorsPersisterConfig::new()
            .allow_partial(true)
            .build_in_memory();
        load_persister.set_data(save_persister.data().unwrap().to_vec());

        let mut target_model = TestModel::<TestBackend>::new(&device);
        let result = target_model.apply_from(&mut load_persister).unwrap();

        // Should only apply the specific tensors
        assert_eq!(result.applied.len(), 3);
        assert!(
            result
                .applied
                .contains(&"encoder.layer1.weight".to_string())
        );
        assert!(result.applied.contains(&"decoder.layer2.bias".to_string()));
        assert!(result.applied.contains(&"head.weight".to_string()));
    }

    #[test]
    #[cfg(target_has_atomic = "ptr")]
    fn test_new_api_complex_configuration() {
        let device = Default::default();
        let model = TestModel::<TestBackend>::new(&device);

        // Complex configuration: filter decoder tensors, remap layer to lyr, add metadata
        let mut save_persister = SafetensorsPersisterConfig::new()
            .with_filter(PathFilter::new().with_regex(r"^decoder\..*"))
            .with_remapping(&[("layer", "lyr")])
            .with_metadata("subset", "decoder")
            .with_metadata("transformation", "layer_to_lyr")
            .with_validation(true)
            .with_partial_loading(true)
            .build_in_memory();

        model.collect_to(&mut save_persister).unwrap();

        // Load into new model
        let mut load_persister = SafetensorsPersisterConfig::new()
            .allow_partial(true)
            .build_in_memory();
        load_persister.set_data(save_persister.data().unwrap().to_vec());

        let mut target_model = TestModel::<TestBackend>::new(&device);
        let result = target_model.apply_from(&mut load_persister).unwrap();

        // Only decoder.norm tensors should apply (they weren't affected by layer->lyr remapping)
        // Note that encoder and head tensors were not saved due to filter
        assert_eq!(result.applied.len(), 2); // decoder.norm.scale and decoder.norm.shift
        assert_eq!(result.unused.len(), 4); // 4 decoder layer tensors with remapped names

        // Applied tensors should be norm tensors
        for tensor_name in &result.applied {
            assert!(tensor_name.starts_with("decoder.norm"));
        }

        // Unused tensors should have remapped names (layer -> lyr)
        for tensor_name in &result.unused {
            assert!(tensor_name.starts_with("decoder.lyr"));
        }
    }

    #[test]
    fn test_api_comparison_old_vs_new() {
        let device = Default::default();
        let model = TestModel::<TestBackend>::new(&device);

        // Old API way
        let tensor_views = model.collect();
        let old_count = tensor_views.len();

        // New API way with same result
        let mut new_persister = SafetensorsPersisterConfig::new().build_in_memory();
        model.collect_to(&mut new_persister).unwrap();

        let mut load_persister = SafetensorsPersisterConfig::new().build_in_memory();
        load_persister.set_data(new_persister.data().unwrap().to_vec());

        let mut target_model = TestModel::<TestBackend>::new(&device);
        let result = target_model.apply_from(&mut load_persister).unwrap();

        // Should collect and apply the same number of tensors
        assert_eq!(old_count, result.applied.len());
        assert!(result.is_success());
    }
}
