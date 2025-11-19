/// The element type ID to be used for dynamic element type while expanding a fused kernel.
pub(crate) const DYN_ELEM_ID: u8 = u8::MAX;
/// The element type ID to be used for the quantization store element type while expanding a fused kernel.
pub(crate) const Q_STORE_DYN_ELEM_ID: u8 = u8::MAX - 1;
/// The element type ID to be used for the quantization param element type while expanding a fused kernel.
pub(crate) const Q_PARAM_DYN_ELEM_ID: u8 = u8::MAX - 2;
