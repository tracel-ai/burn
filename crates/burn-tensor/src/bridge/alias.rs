use crate::BackendTypes;

/// Device type used by the backend.
pub type Device<B> = <B as BackendTypes>::Device;
