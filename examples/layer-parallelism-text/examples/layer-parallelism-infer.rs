fn main() {
    layer_parallelism_text::inference::infer(
        layer_parallelism_text::devices::available(),
        &layer_parallelism_text::artifact_dir(),
    );
}
