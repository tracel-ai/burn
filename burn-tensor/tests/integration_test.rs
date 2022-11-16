pub type TestBackend = burn_tensor::backend::NdArrayBackend<f32>;

#[cfg(feature = "export_tests")]
burn_tensor::test_all!();
