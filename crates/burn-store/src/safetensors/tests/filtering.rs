use crate::{ModuleSnapshot, SafetensorsStore};

use super::round_trip::ComplexModule;

type TestBackend = burn_ndarray::NdArray;

#[test]
#[cfg(target_has_atomic = "ptr")]
fn filtered_export_import() {
    let device = Default::default();
    let module1 = ComplexModule::<TestBackend>::new(&device);
    let mut module2 = ComplexModule::<TestBackend>::new_zeros(&device);

    // Export only encoder tensors using the builder pattern
    let mut save_store = SafetensorsStore::from_bytes(None).with_regex(r"^encoder\..*");
    module1.collect_to(&mut save_store).unwrap();

    // Import filtered tensors - need to allow partial since we only saved encoder tensors
    let mut load_store = SafetensorsStore::from_bytes(None).allow_partial(true);
    if let SafetensorsStore::Memory(ref mut p) = load_store
        && let SafetensorsStore::Memory(ref p_save) = save_store
    {
        // Get Arc and extract data
        let data_arc = p_save.data().unwrap();
        p.set_data(data_arc.as_ref().clone());
    }
    let result = module2.apply_from(&mut load_store).unwrap();

    assert!(result.is_success());
    assert_eq!(result.applied.len(), 3); // encoder.weight, encoder.bias, encoder.norm
    assert!(!result.missing.is_empty()); // decoder and layers tensors are missing
}

#[test]
#[cfg(target_has_atomic = "ptr")]
fn builder_pattern_filtering() {
    let device = Default::default();
    let module = ComplexModule::<TestBackend>::new(&device);

    // Test with_regex - multiple patterns (OR logic)
    let mut store = SafetensorsStore::from_bytes(None)
        .with_regex(r"^encoder\..*") // Match encoder tensors
        .with_regex(r".*\.bias$"); // OR match any bias tensors

    let views = module.collect();
    let filtered_count = views
        .iter()
        .filter(|v| {
            let path = v.full_path();
            path.starts_with("encoder.") || path.ends_with(".bias")
        })
        .count();

    module.collect_to(&mut store).unwrap();

    // Verify we saved the expected number of tensors
    if let SafetensorsStore::Memory(ref p) = store {
        let data = p.data().unwrap();
        let tensors = safetensors::SafeTensors::deserialize(&data).unwrap();
        assert_eq!(tensors.len(), filtered_count);
    }
}

#[test]
fn builder_pattern_exact_paths() {
    let device = Default::default();
    let module = ComplexModule::<TestBackend>::new(&device);

    // Test with_full_path and with_full_paths
    let paths = vec!["encoder.weight", "decoder.scale"];
    let mut store = SafetensorsStore::from_bytes(None)
        .with_full_path("encoder.norm")
        .with_full_paths(paths.clone());

    module.collect_to(&mut store).unwrap();

    // Verify only specified tensors were saved
    if let SafetensorsStore::Memory(ref p) = store {
        let data = p.data().unwrap();
        let tensors = safetensors::SafeTensors::deserialize(&data).unwrap();
        assert_eq!(tensors.len(), 3); // encoder.norm + encoder.weight + decoder.scale

        for (name, _) in tensors.tensors() {
            assert!(name == "encoder.norm" || name == "encoder.weight" || name == "decoder.scale");
        }
    }
}

#[test]
fn builder_pattern_with_predicate() {
    let device = Default::default();
    let module = ComplexModule::<TestBackend>::new(&device);

    // Test with_predicate - custom logic
    let mut store = SafetensorsStore::from_bytes(None).with_predicate(|path, _| {
        // Only save tensors with "layer" in the path and ending with "weight"
        path.contains("layer") && path.ends_with("weight")
    });

    module.collect_to(&mut store).unwrap();

    // Verify only layer weights were saved
    if let SafetensorsStore::Memory(ref p) = store {
        let data = p.data().unwrap();
        let tensors = safetensors::SafeTensors::deserialize(&data).unwrap();

        for (name, _) in tensors.tensors() {
            assert!(name.contains("layer"));
            assert!(name.ends_with("weight"));
        }
    }
}

#[test]
fn builder_pattern_combined() {
    let device = Default::default();
    let module = ComplexModule::<TestBackend>::new(&device);

    // Combine multiple filter methods
    #[cfg(target_has_atomic = "ptr")]
    {
        let mut store = SafetensorsStore::from_bytes(None)
            .with_regex(r"^encoder\..*") // All encoder tensors
            .with_full_path("decoder.scale") // Plus specific decoder.scale
            .with_predicate(|path, _| {
                // Plus any projection tensors
                path.contains("projection")
            });

        module.collect_to(&mut store).unwrap();

        if let SafetensorsStore::Memory(ref p) = store {
            let data = p.data().unwrap();
            let tensors = safetensors::SafeTensors::deserialize(&data).unwrap();

            // Should have encoder.*, decoder.scale, and projection tensors
            let mut names = Vec::new();
            for (name, _) in tensors.tensors() {
                names.push(name);
            }
            assert!(names.iter().any(|n| n == "encoder.weight"));
            assert!(names.iter().any(|n| n == "encoder.bias"));
            assert!(names.iter().any(|n| n == "encoder.norm"));
            assert!(names.iter().any(|n| n == "decoder.scale"));
            // decoder.projection.* should also be included due to predicate
            assert!(names.iter().any(|n| n.contains("projection")));
        }
    }
}

#[test]
fn builder_pattern_match_all() {
    let device = Default::default();
    let module = ComplexModule::<TestBackend>::new(&device);

    let all_views = module.collect();
    let total_count = all_views.len();

    // Test match_all - should save everything
    let mut store = SafetensorsStore::from_bytes(None).match_all();

    module.collect_to(&mut store).unwrap();

    if let SafetensorsStore::Memory(ref p) = store {
        let data = p.data().unwrap();
        let tensors = safetensors::SafeTensors::deserialize(&data).unwrap();
        assert_eq!(tensors.len(), total_count);
    }
}
