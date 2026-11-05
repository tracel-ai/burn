fn main() {
    tensor_parallelism_text::inference::infer(
        tensor_parallelism_text::devices(),
        &tensor_parallelism_text::artifact_dir(),
    );
}
