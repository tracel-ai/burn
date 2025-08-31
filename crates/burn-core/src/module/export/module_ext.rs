use hashbrown::HashMap;

use super::{TensorView, TensorViewCollector};
use crate::module::Module;
use crate::tensor::backend::Backend;

/// Extension trait for modules that provides tensor export functionality
pub trait ModuleExport<B: Backend>: Module<B> {
    /// Export tensor views for inspection without copying data
    fn export_tensor_views(&self) -> HashMap<String, TensorView> {
        let mut collector = TensorViewCollector::new();
        self.visit(&mut collector);
        collector.tensors
    }

    /// Export filtered tensor views matching a regex pattern
    fn export_tensor_views_filtered(
        &self,
        pattern: &str,
    ) -> Result<HashMap<String, TensorView>, regex::Error> {
        let mut collector = TensorViewCollector::with_filter(pattern)?;
        self.visit(&mut collector);
        Ok(collector.tensors)
    }
}

// Blanket implementation for all modules recursively
impl<B: Backend, M: Module<B>> ModuleExport<B> for M {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate as burn;
    use crate::{
        TestBackend,
        module::{Module, Param},
        nn::LinearConfig,
    };
    use burn_tensor::Tensor;

    #[derive(Module, Debug)]
    struct TestModule<B: Backend> {
        encoder: TestSubModule<B>,
        decoder: TestSubModule<B>,
    }

    #[derive(Module, Debug)]
    struct TestSubModule<B: Backend> {
        weight: Param<Tensor<B, 2>>,
        bias: Param<Tensor<B, 1>>,
    }

    impl<B: Backend> TestModule<B> {
        fn new(device: &B::Device) -> Self {
            Self {
                encoder: TestSubModule::new(device),
                decoder: TestSubModule::new(device),
            }
        }
    }

    impl<B: Backend> TestSubModule<B> {
        fn new(device: &B::Device) -> Self {
            Self {
                weight: Param::from_data([[1.0, 2.0], [3.0, 4.0]], device),
                bias: Param::from_data([5.0, 6.0], device),
            }
        }
    }

    #[test]
    fn test_export_tensor_views() {
        let device = Default::default();
        let module = TestModule::<TestBackend>::new(&device);

        let views = module.export_tensor_views();

        assert_eq!(views.len(), 4);
        assert!(views.contains_key("encoder.weight"));
        assert!(views.contains_key("encoder.bias"));
        assert!(views.contains_key("decoder.weight"));
        assert!(views.contains_key("decoder.bias"));
    }

    #[test]
    fn test_export_tensor_views_filtered() {
        let device = Default::default();
        let module = TestModule::<TestBackend>::new(&device);

        let views = module
            .export_tensor_views_filtered(r"^encoder\..*")
            .unwrap();

        assert_eq!(views.len(), 2);
        assert!(views.contains_key("encoder.weight"));
        assert!(views.contains_key("encoder.bias"));
        assert!(!views.contains_key("decoder.weight"));
        assert!(!views.contains_key("decoder.bias"));
    }

    #[test]
    fn test_with_linear_module() {
        let device = Default::default();
        let linear = LinearConfig::new(10, 20)
            .with_bias(true)
            .init::<TestBackend>(&device);

        let views = linear.export_tensor_views();
        assert!(views.contains_key("weight"));
        assert!(views.contains_key("bias"));

        // Verify the views contain correct data
        let weight_view = views.get("weight").unwrap();
        let weight_data = weight_view.to_data();
        assert_eq!(weight_data.shape, vec![10, 20]);

        let bias_view = views.get("bias").unwrap();
        let bias_data = bias_view.to_data();
        assert_eq!(bias_data.shape, vec![20]);
    }
}
