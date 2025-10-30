// Tests for ONNX Scan operator

use crate::include_models;
include_models!(
    // TODO: Enable when Scan support is added
    // scan_cumsum,
    // scan_reverse,
    // scan_multi_state
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::TestBackend;
    use burn::tensor::{Tensor, TensorData};

    // TODO: Add scan tests when Scan operator is implemented
}
