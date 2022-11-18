mod activation;
mod grad;
mod module;
mod ops;
mod stats;

#[macro_export]
macro_rules! testgen_all {
    () => {
        // test activation
        burn_tensor::testgen_gelu!();
        burn_tensor::testgen_relu!();
        burn_tensor::testgen_softmax!();

        // test ad
        burn_tensor::testgen_ad_complex!();
        burn_tensor::testgen_ad_multithread!();
        burn_tensor::testgen_ad_add!();
        burn_tensor::testgen_ad_aggregation!();
        burn_tensor::testgen_ad_cat!();
        burn_tensor::testgen_ad_cross_entropy_loss!();
        burn_tensor::testgen_ad_div!();
        burn_tensor::testgen_ad_erf!();
        burn_tensor::testgen_ad_exp!();
        burn_tensor::testgen_ad_index!();
        burn_tensor::testgen_ad_log!();
        burn_tensor::testgen_ad_mask!();
        burn_tensor::testgen_ad_matmul!();
        burn_tensor::testgen_ad_mul!();
        burn_tensor::testgen_ad_neg!();
        burn_tensor::testgen_ad_powf!();
        burn_tensor::testgen_ad_relu!();
        burn_tensor::testgen_ad_reshape!();
        burn_tensor::testgen_ad_softmax!();
        burn_tensor::testgen_ad_sub!();
        burn_tensor::testgen_ad_transpose!();

        // test module
        burn_tensor::testgen_module_backward!();
        burn_tensor::testgen_module_forward!();

        // test ops
        burn_tensor::testgen_add!();
        burn_tensor::testgen_aggregation!();
        burn_tensor::testgen_arg!();
        burn_tensor::testgen_div!();
        burn_tensor::testgen_erf!();
        burn_tensor::testgen_exp!();
        burn_tensor::testgen_index!();
        burn_tensor::testgen_map_comparison!();
        burn_tensor::testgen_mask!();
        burn_tensor::testgen_matmul!();
        burn_tensor::testgen_mul!();
        burn_tensor::testgen_neg!();
        burn_tensor::testgen_powf!();
        burn_tensor::testgen_repeat!();
        burn_tensor::testgen_reshape!();
        burn_tensor::testgen_sub!();
        burn_tensor::testgen_transpose!();

        // test stats
        burn_tensor::testgen_stats!();
    };
}
