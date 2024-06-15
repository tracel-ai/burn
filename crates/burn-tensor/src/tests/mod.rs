mod activation;
mod clone_invariance;
mod module;
mod ops;
mod stats;

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_all {
    () => {
        // test activation
        burn_tensor::testgen_gelu!();
        burn_tensor::testgen_mish!();
        burn_tensor::testgen_relu!();
        burn_tensor::testgen_leaky_relu!();
        burn_tensor::testgen_softmax!();
        burn_tensor::testgen_softplus!();
        burn_tensor::testgen_sigmoid!();
        burn_tensor::testgen_log_sigmoid!();
        burn_tensor::testgen_silu!();
        burn_tensor::testgen_tanh_activation!();

        // test module
        burn_tensor::testgen_module_forward!();
        burn_tensor::testgen_module_conv1d!();
        burn_tensor::testgen_module_conv2d!();
        burn_tensor::testgen_module_conv_transpose1d!();
        burn_tensor::testgen_module_conv_transpose2d!();
        burn_tensor::testgen_module_unfold4d!();
        burn_tensor::testgen_module_max_pool1d!();
        burn_tensor::testgen_module_max_pool2d!();
        burn_tensor::testgen_module_avg_pool1d!();
        burn_tensor::testgen_module_avg_pool2d!();
        burn_tensor::testgen_module_adaptive_avg_pool1d!();
        burn_tensor::testgen_module_adaptive_avg_pool2d!();
        burn_tensor::testgen_module_nearest_interpolate!();
        burn_tensor::testgen_module_bilinear_interpolate!();
        burn_tensor::testgen_module_bicubic_interpolate!();

        // test ops
        burn_tensor::testgen_add!();
        burn_tensor::testgen_aggregation!();
        burn_tensor::testgen_arange!();
        burn_tensor::testgen_arange_step!();
        burn_tensor::testgen_arg!();
        burn_tensor::testgen_cast!();
        burn_tensor::testgen_cat!();
        burn_tensor::testgen_chunk!();
        burn_tensor::testgen_clamp!();
        burn_tensor::testgen_close!();
        burn_tensor::testgen_cos!();
        burn_tensor::testgen_create_like!();
        burn_tensor::testgen_div!();
        burn_tensor::testgen_erf!();
        burn_tensor::testgen_exp!();
        burn_tensor::testgen_flatten!();
        burn_tensor::testgen_full!();
        burn_tensor::testgen_gather_scatter!();
        burn_tensor::testgen_init!();
        burn_tensor::testgen_iter_dim!();
        burn_tensor::testgen_log!();
        burn_tensor::testgen_log1p!();
        burn_tensor::testgen_map_comparison!();
        burn_tensor::testgen_mask!();
        burn_tensor::testgen_matmul!();
        burn_tensor::testgen_maxmin!();
        burn_tensor::testgen_mul!();
        burn_tensor::testgen_narrow!();
        burn_tensor::testgen_neg!();
        burn_tensor::testgen_one_hot!();
        burn_tensor::testgen_powf_scalar!();
        burn_tensor::testgen_random!();
        burn_tensor::testgen_recip!();
        burn_tensor::testgen_repeat!();
        burn_tensor::testgen_reshape!();
        burn_tensor::testgen_select!();
        burn_tensor::testgen_sin!();
        burn_tensor::testgen_slice!();
        burn_tensor::testgen_stack!();
        burn_tensor::testgen_sqrt!();
        burn_tensor::testgen_abs!();
        burn_tensor::testgen_squeeze!();
        burn_tensor::testgen_sub!();
        burn_tensor::testgen_tanh!();
        burn_tensor::testgen_transpose!();
        burn_tensor::testgen_tri!();
        burn_tensor::testgen_powf!();
        burn_tensor::testgen_any!();
        burn_tensor::testgen_all_op!();
        burn_tensor::testgen_permute!();
        burn_tensor::testgen_movedim!();
        burn_tensor::testgen_flip!();
        burn_tensor::testgen_bool!();
        burn_tensor::testgen_argwhere_nonzero!();
        burn_tensor::testgen_sign!();
        burn_tensor::testgen_expand!();
        burn_tensor::testgen_tri_mask!();
        burn_tensor::testgen_sort_argsort!();
        burn_tensor::testgen_topk!();
        burn_tensor::testgen_remainder!();
        burn_tensor::testgen_cartesian_grid!();

        // test stats
        burn_tensor::testgen_var!();
        burn_tensor::testgen_cov!();
        burn_tensor::testgen_eye!();
        burn_tensor::testgen_display!();

        // test clone invariance
        burn_tensor::testgen_clone_invariance!();

        // test padding
        burn_tensor::testgen_padding!();
    };
}
