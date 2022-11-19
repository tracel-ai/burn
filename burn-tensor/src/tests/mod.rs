mod activation;
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

        // test module
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
