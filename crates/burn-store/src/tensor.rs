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
/// [`into_pack`](Self::into_pack). Only the name changes in place; a transform of the dtype,
/// shape or data goes through [`bridge::map_data`], which builds a new tensor and so drops
/// whatever the old source offered.
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
    use alloc::string::ToString;
    use alloc::vec;
    use alloc::vec::Vec;
    use burn_core::tensor::shape;

    use crate::{ModuleAdapter, ModuleContext, ModuleStore, PyTorchToBurnAdapter, PytorchStore};

    const FIXTURE: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/pytorch-tests/tests/linear/linear_transposed.pt"
    );

    /// The fixture's `[4, 3]` `fc2.weight`, saved as a `.t()` view.
    fn transposed_view_weight() -> StoreTensor {
        let mut tensors = pytorch_reader::PytorchReader::new(FIXTURE)
            .unwrap()
            .into_tensors();
        tensors.remove("fc2.weight").unwrap().into()
    }

    fn values(tensor: StoreTensor) -> Vec<f32> {
        tensor.into_data().unwrap().try_into_vec::<f32>().unwrap()
    }

    /// The row-major transpose of a row-major `[rows, cols]` matrix.
    fn transpose(values: &[f32], cols: usize) -> Vec<f32> {
        (0..cols)
            .flat_map(|j| values.iter().skip(j).step_by(cols).copied())
            .collect()
    }

    fn adapt_as_linear(tensor: StoreTensor) -> StoreTensor {
        let containers = vec!["Struct:Linear".to_string()];
        PyTorchToBurnAdapter.adapt(tensor, ModuleContext::new(&containers))
    }

    /// A weight saved as a `.t()` view offers its transpose, which reads the stored bytes as
    /// they are and agrees with transposing what the tensor itself reads. The same weight as a
    /// [`burn_pack::Tensor`] offers none.
    #[test]
    fn a_transposed_view_offers_its_stored_transpose() {
        let weight = transposed_view_weight();
        assert_eq!(weight.shape(), [4, 3]);
        let transposed = weight
            .stored_transpose()
            .expect("a `.t()` view has a transpose");
        assert_eq!(transposed.shape(), [3, 4]);
        assert_eq!(transposed.name(), weight.name());
        assert!(
            StoreTensor::from(weight.clone().into_pack())
                .stored_transpose()
                .is_none()
        );

        assert_eq!(values(transposed), transpose(&values(weight), 3));
    }

    /// The Linear adapter takes the stored transpose, leaving the reader's tensor in place
    /// rather than composing a transpose onto it.
    #[test]
    fn linear_adapter_takes_the_stored_transpose() {
        let weight = transposed_view_weight();
        let expected = transpose(&values(weight.clone()), 3);

        let adapted = adapt_as_linear(weight);
        assert!(matches!(adapted.source, TensorSource::PyTorch(_)));
        assert_eq!(adapted.shape(), [3, 4]);
        assert_eq!(values(adapted), expected);
    }

    /// A transform ahead of the Linear adapter is transposed with it, not bypassed by reading
    /// the stored transpose of the original data.
    #[test]
    fn linear_adapter_transposes_a_transformed_view() {
        let weight = transposed_view_weight();
        let expected: Vec<f32> = transpose(&values(weight.clone()), 3)
            .into_iter()
            .map(|value| value + 10.0)
            .collect();

        let name = weight.name().to_string();
        let shifted = bridge::map_data(weight, name, DType::F32, shape![4, 3], |data| {
            let shifted: Vec<f32> = data.iter::<f32>().map(|value| value + 10.0).collect();
            TensorData::new(shifted, [4, 3])
        });

        assert_eq!(values(adapt_as_linear(shifted)), expected);
    }

    /// A tensor keeps the reader's source, and so its stored transpose, through the store's
    /// remapping.
    #[test]
    fn the_stored_transpose_survives_the_store_and_a_remap() {
        let mut store = PytorchStore::from_file(FIXTURE).with_key_remapping(r"^fc2\.", "out.");
        let weight = store.get_tensor("out.weight").unwrap().unwrap();

        assert!(matches!(weight.source, TensorSource::PyTorch(_)));
        assert!(weight.stored_transpose().is_some());
    }
}
