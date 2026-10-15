//! The tensor burn-store carries from a store to a module.

use alloc::string::String;

use burn_core::tensor::{DType, TensorData};
use burn_pack::{Error as PackError, Tensor as PackTensor};

use crate::bridge;

/// A tensor on its way from a store to a module, kept as its source handed it over.
///
/// Keeping the source means what it can do cheaply survives until an adapter can use it. A
/// PyTorch checkpoint, for one, holds a Linear weight saved as a `.t()` view in the transposed
/// order already, so [`PyTorchToBurnAdapter`](crate::PyTorchToBurnAdapter) can read it as a
/// straight copy instead of gathering it into the view's order and transposing it back.
///
/// Every source reads out as a [`burn_pack::Tensor`] in the end, via
/// [`into_pack`](Self::into_pack), and converts into one with [`From`]. Only the name changes
/// in place; a transform of the dtype, shape or data goes through
/// [`bridge::map_data`], which builds a new tensor and so drops whatever the old source offered.
#[derive(Clone, Debug)]
pub struct StoreTensor {
    source: TensorSource,
}

/// Where a [`StoreTensor`] came from.
#[derive(Clone, Debug)]
enum TensorSource {
    Pack(PackTensor),
    #[cfg(feature = "pytorch")]
    PyTorch(pytorch_reader::Tensor),
}

impl StoreTensor {
    /// Fully-qualified tensor name (e.g. `"encoder.layer1.weight"`).
    #[must_use]
    pub fn name(&self) -> &str {
        match &self.source {
            TensorSource::Pack(tensor) => &tensor.name,
            #[cfg(feature = "pytorch")]
            TensorSource::PyTorch(tensor) => &tensor.name,
        }
    }

    /// Rename the tensor, leaving its data untouched.
    pub fn set_name(&mut self, name: impl Into<String>) {
        let name = name.into();
        match &mut self.source {
            TensorSource::Pack(tensor) => tensor.name = name,
            #[cfg(feature = "pytorch")]
            TensorSource::PyTorch(tensor) => tensor.name = name,
        }
    }

    /// Data type of the tensor.
    #[must_use]
    pub fn dtype(&self) -> DType {
        match &self.source {
            TensorSource::Pack(tensor) => tensor.dtype,
            #[cfg(feature = "pytorch")]
            TensorSource::PyTorch(tensor) => bridge::pytorch_dtype(tensor.dtype()),
        }
    }

    /// Tensor shape.
    #[must_use]
    pub fn shape(&self) -> &[usize] {
        match &self.source {
            TensorSource::Pack(tensor) => &tensor.shape,
            #[cfg(feature = "pytorch")]
            TensorSource::PyTorch(tensor) => tensor.shape(),
        }
    }

    /// Parameter id, used to preserve identities for stateful training. PyTorch carries none.
    #[must_use]
    pub fn param_id(&self) -> Option<u64> {
        match &self.source {
            TensorSource::Pack(tensor) => tensor.param_id,
            #[cfg(feature = "pytorch")]
            TensorSource::PyTorch(_) => None,
        }
    }

    /// Number of raw bytes the tensor's data occupies, known without reading it.
    #[must_use]
    pub fn byte_len(&self) -> usize {
        match &self.source {
            TensorSource::Pack(tensor) => tensor.byte_len(),
            #[cfg(feature = "pytorch")]
            TensorSource::PyTorch(tensor) => tensor.byte_len(),
        }
    }

    /// The tensor as a [`burn_pack::Tensor`], with its bytes still unread.
    #[must_use]
    pub fn into_pack(self) -> PackTensor {
        match self.source {
            TensorSource::Pack(tensor) => tensor,
            #[cfg(feature = "pytorch")]
            TensorSource::PyTorch(tensor) => bridge::from_pytorch(tensor),
        }
    }

    /// Read the tensor's data, leaving it intact. See [`bridge::to_data`] for the cost.
    ///
    /// # Errors
    ///
    /// If the source fails to produce the bytes, or produces ones that disagree with the
    /// tensor's dtype and shape.
    pub fn to_data(&self) -> Result<TensorData, PackError> {
        match &self.source {
            TensorSource::Pack(tensor) => bridge::to_data(tensor),
            // The reader's handle is reference-counted, so the clone copies no data.
            #[cfg(feature = "pytorch")]
            TensorSource::PyTorch(tensor) => {
                bridge::into_data(bridge::from_pytorch(tensor.clone()))
            }
        }
    }

    /// Take the tensor's data, reading it from the source.
    ///
    /// # Errors
    ///
    /// As [`to_data`](Self::to_data).
    pub fn into_data(self) -> Result<TensorData, PackError> {
        bridge::into_data(self.into_pack())
    }

    /// The transpose of a 2-D tensor whose source holds it in transposed order, which reads
    /// as a straight copy. `None` when the transpose would have to move the data.
    pub(crate) fn stored_transpose(&self) -> Option<Self> {
        match &self.source {
            TensorSource::Pack(_) => None,
            #[cfg(feature = "pytorch")]
            TensorSource::PyTorch(tensor) => tensor.transposed().map(Self::from),
        }
    }
}

impl From<PackTensor> for StoreTensor {
    fn from(tensor: PackTensor) -> Self {
        Self {
            source: TensorSource::Pack(tensor),
        }
    }
}

#[cfg(feature = "pytorch")]
impl From<pytorch_reader::Tensor> for StoreTensor {
    fn from(tensor: pytorch_reader::Tensor) -> Self {
        Self {
            source: TensorSource::PyTorch(tensor),
        }
    }
}

#[cfg(all(test, feature = "pytorch"))]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    /// A weight saved as a `.t()` view offers its transpose, which reads the stored bytes as
    /// they are and agrees with transposing what the tensor itself reads.
    #[test]
    fn a_transposed_view_offers_its_stored_transpose() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/pytorch-tests/tests/linear/linear_transposed.pt"
        );
        let mut tensors = pytorch_reader::PytorchReader::new(path)
            .unwrap()
            .into_tensors();

        let weight = StoreTensor::from(tensors.remove("fc2.weight").unwrap());
        assert_eq!(weight.shape(), [4, 3]);
        let transposed = weight
            .stored_transpose()
            .expect("a `.t()` view has a transpose");
        assert_eq!(transposed.shape(), [3, 4]);
        assert_eq!(transposed.name(), weight.name());

        let values = weight.into_data().unwrap().try_into_vec::<f32>().unwrap();
        let expected: Vec<f32> = (0..3)
            .flat_map(|j| values.iter().skip(j).step_by(3).copied())
            .collect();
        let transposed = transposed
            .into_data()
            .unwrap()
            .try_into_vec::<f32>()
            .unwrap();
        assert_eq!(transposed, expected);

        let bias = StoreTensor::from(tensors.remove("fc1.bias").unwrap());
        assert!(bias.stored_transpose().is_none());
        assert!(
            StoreTensor::from(bias.into_pack())
                .stored_transpose()
                .is_none()
        );
    }
}
