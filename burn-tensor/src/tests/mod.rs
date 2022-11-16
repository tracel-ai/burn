mod activation;
mod grad;
mod module;
mod ops;
mod stats;

#[macro_export]
macro_rules! test_all {
    () => {
        use burn_tensor::activation::*;
        use burn_tensor::backend::Backend;
        use burn_tensor::module::*;
        use burn_tensor::*;

        type TestADTensor<const D: usize> = burn_tensor::Tensor<TestADBackend, D>;
        type TestADBackend = burn_tensor::backend::ADBackendDecorator<TestBackend>;

        // test activation
        burn_tensor::test_gelu!();
        burn_tensor::test_relu!();
        burn_tensor::test_softmax!();

        // test ad
        burn_tensor::test_ad_add!();
        burn_tensor::test_ad_aggregation!();
        burn_tensor::test_ad_cat!();
        burn_tensor::test_ad_cross_entropy_loss!();
        burn_tensor::test_ad_div!();
        burn_tensor::test_ad_erf!();
        burn_tensor::test_ad_exp!();
        burn_tensor::test_ad_index!();
        burn_tensor::test_ad_log!();
        burn_tensor::test_ad_mask!();
        burn_tensor::test_ad_matmul!();
        burn_tensor::test_ad_mul!();
        burn_tensor::test_ad_neg!();
        burn_tensor::test_ad_powf!();
        burn_tensor::test_ad_relu!();
        burn_tensor::test_ad_reshape!();
        burn_tensor::test_ad_softmax!();
        burn_tensor::test_ad_sub!();
        burn_tensor::test_ad_transpose!();

        // test module
        burn_tensor::test_module_backward!();
        burn_tensor::test_module_forward!();

        // test ops
        burn_tensor::test_add!();
        burn_tensor::test_aggregation!();
        burn_tensor::test_arg!();
        burn_tensor::test_div!();
        burn_tensor::test_erf!();
        burn_tensor::test_exp!();
        burn_tensor::test_index!();
        burn_tensor::test_map_comparison!();
        burn_tensor::test_mask!();
        burn_tensor::test_matmul!();
        burn_tensor::test_mul!();
        burn_tensor::test_neg!();
        burn_tensor::test_powf!();
        burn_tensor::test_repeat!();
        burn_tensor::test_reshape!();
        burn_tensor::test_sub!();
        burn_tensor::test_transpose!();

        // test stats
        burn_tensor::test_stats!();
    };
}
