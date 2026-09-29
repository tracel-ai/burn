use crate::{ModuleSnapshot, SafetensorsStore};
use burn_nn::LinearConfig;

#[test]
fn shape_mismatch_errors() {
    let device = Default::default();

    // Create a module
    let module = LinearConfig::new(2, 2).with_bias(true).init(&device);

    // Save module
    let mut save_store = SafetensorsStore::from_bytes(None);
    module.save_into(&mut save_store).unwrap();

    // Try to load into incompatible module (different dimensions)
    let mut incompatible_module = LinearConfig::new(3, 3).with_bias(true).init(&device);

    // Load without validation - should return errors in the result
    let mut load_store = SafetensorsStore::from_bytes(None).validate(false); // Disable validation to get errors in result
    if let SafetensorsStore::Memory(ref mut p) = load_store
        && let SafetensorsStore::Memory(ref p_save) = save_store
    {
        // Get Arc and extract data
        let data_arc = p_save.data().unwrap();
        p.set_data(data_arc.as_ref().clone());
    }

    let result = incompatible_module.load_from(&mut load_store).unwrap();

    // Should have errors due to shape mismatch
    assert!(!result.errors.is_empty());

    // Try again with validation enabled - should return Err
    let mut load_store_with_validation = SafetensorsStore::from_bytes(None).validate(true);
    if let SafetensorsStore::Memory(ref mut p) = load_store_with_validation
        && let SafetensorsStore::Memory(ref p_save) = save_store
    {
        // Get Arc and extract data
        let data_arc = p_save.data().unwrap();
        p.set_data(data_arc.as_ref().clone());
    }

    let validation_result = incompatible_module.load_from(&mut load_store_with_validation);
    assert!(validation_result.is_err());
}

/// `safetensors::View::data` has no error channel, so a tensor that fails to materialize can
/// only be reported by panicking out of it. `collect_from` declares a `Result`, so that unwind
/// has to be caught at the boundary and handed back as an error, rather than escaping the
/// store and leaving a half-written file behind.
#[test]
fn a_failing_tensor_does_not_unwind_out_of_collect_from() {
    use crate::{ModuleAdapter, ModuleContext, bridge};
    use alloc::boxed::Box;
    use burn_pack::Tensor as PackTensor;

    /// Stands in for a backend that panics reading a parameter back from its device.
    #[derive(Clone)]
    struct PanickingAdapter;

    impl ModuleAdapter for PanickingAdapter {
        fn adapt(&self, tensor: PackTensor, _ctx: ModuleContext<'_>) -> PackTensor {
            bridge::deferred(
                tensor.name.clone(),
                tensor.dtype,
                tensor.shape.clone(),
                None,
                || panic!("device readback panicked"),
            )
        }

        fn clone_box(&self) -> Box<dyn ModuleAdapter> {
            Box::new(self.clone())
        }
    }

    let device = Default::default();
    let module = LinearConfig::new(2, 2).init(&device);

    let mut store = SafetensorsStore::from_bytes(None).with_to_adapter(PanickingAdapter);
    let err = module
        .save_into(&mut store)
        .expect_err("a panicking provider must be returned, not unwound");

    // Which parameter serializes first is not fixed, so assert on the shape of the report:
    // it names the tensor it failed on and carries the original panic's message.
    let message = alloc::format!("{err}");
    assert!(
        message.contains("tensor '") && message.contains("device readback panicked"),
        "the error should name the tensor and carry the cause, got: {message}"
    );
}
