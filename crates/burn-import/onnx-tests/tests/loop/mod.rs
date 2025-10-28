// Tests for ONNX Loop operator

use crate::include_models;
include_models!(
    // TODO: Enable when Loop support is added
    // loop_simple,
    // loop_dynamic_cond,
    // loop_multi_deps
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::TestBackend;
    use burn::tensor::{Tensor, TensorData};

    // TODO: Add loop tests when Loop operator is implemented
}
