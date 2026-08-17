use std::{fs, path::Path};

use crate::ExportError;

/// A serialized ONNX model produced by [`crate::OnnxExporter`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OnnxModel {
    bytes: Vec<u8>,
}

impl OnnxModel {
    pub(crate) fn new(bytes: Vec<u8>) -> Self {
        Self { bytes }
    }

    /// Return the serialized ONNX protobuf.
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Consume this model and return its serialized ONNX protobuf.
    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }

    /// Save the serialized ONNX model to a file.
    ///
    /// An existing file at `path` is replaced.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), ExportError> {
        let path = path.as_ref();
        fs::write(path, &self.bytes).map_err(|error| ExportError::FileWrite {
            path: path.display().to_string(),
            reason: error.to_string(),
        })
    }
}

impl AsRef<[u8]> for OnnxModel {
    fn as_ref(&self) -> &[u8] {
        self.as_bytes()
    }
}
