use crate::include_models;
include_models!(graph_multiple_output_tracking);

#[cfg(test)]
mod tests {
    use super::*;

    use crate::backend::TestBackend;

    #[test]
    fn graph_multiple_output_tracking() {
        // Initialize the model with weights (loaded from the exported file)
        let _model: graph_multiple_output_tracking::Model<TestBackend> =
            graph_multiple_output_tracking::Model::default();

        // We don't actually care about the output here, the compiler will tell us if we passed
    }
}
