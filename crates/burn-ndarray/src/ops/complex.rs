use burn_backend::TypedDevice;

use crate::NdArray;

impl TypedDevice<Self> for NdArray {
    fn complex_device(
        _tensor: &burn_backend::ComplexTensor<Self>,
    ) -> <Self as burn_backend::BackendTypes>::Device {
        panic!("NdArray backend does not yet support interleaved complex tensors")
    }
}
