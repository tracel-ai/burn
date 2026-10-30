fn main() {
    layer_parallelism_text::finetune::finetune(
        layer_parallelism_text::devices::trainable(),
        &layer_parallelism_text::artifact_dir(),
    );
}
