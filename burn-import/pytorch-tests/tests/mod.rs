cfg_if::cfg_if! {
    if #[cfg(not(target_os = "windows"))] {
        // The crate is not supported on Windows because of Candle's pt bug on Windows
        // (see https://github.com/huggingface/candle/issues/1454).
        mod batch_norm;
        mod boolean;
        mod buffer;
        mod complex_nested;
        mod conv1d;
        mod conv2d;
        mod conv_transpose1d;
        mod conv_transpose2d;
        mod embedding;
        mod group_norm;
        mod integer;
        mod key_remap;
        mod layer_norm;
        mod linear;
    }
}
