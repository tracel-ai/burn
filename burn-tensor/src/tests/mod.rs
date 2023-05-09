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
        burn_tensor::testgen_sigmoid!();

        // test module
        burn_tensor::testgen_module_forward!();
        burn_tensor::testgen_module_conv1d!();
        burn_tensor::testgen_module_conv2d!();
        burn_tensor::testgen_module_conv_transpose1d!();
        burn_tensor::testgen_module_conv_transpose2d!();
        burn_tensor::testgen_module_max_pool2d!();
        burn_tensor::testgen_module_avg_pool2d!();

        // test ops
        burn_tensor::testgen_add!();
        burn_tensor::testgen_aggregation!();
        burn_tensor::testgen_arg!();
        burn_tensor::testgen_maxmin!();
        burn_tensor::testgen_cos!();
        burn_tensor::testgen_div!();
        burn_tensor::testgen_erf!();
        burn_tensor::testgen_exp!();
        burn_tensor::testgen_log!();
        burn_tensor::testgen_log1p!();
        burn_tensor::testgen_index!();
        burn_tensor::testgen_index_select!();
        burn_tensor::testgen_index_select_dim!();
        burn_tensor::testgen_map_comparison!();
        burn_tensor::testgen_mask!();
        burn_tensor::testgen_matmul!();
        burn_tensor::testgen_mul!();
        burn_tensor::testgen_neg!();
        burn_tensor::testgen_powf!();
        burn_tensor::testgen_repeat!();
        burn_tensor::testgen_reshape!();
        burn_tensor::testgen_flatten!();
        burn_tensor::testgen_sin!();
        burn_tensor::testgen_tanh!();
        burn_tensor::testgen_sub!();
        burn_tensor::testgen_transpose!();

        // test stats
        burn_tensor::testgen_stats!();
    };
}
