use serde::{Deserialize, Serialize};

#[derive(Default, Clone, Serialize, Deserialize)]
pub struct Scalars {
    pub(crate) num_f32: usize,
    pub(crate) num_f16: usize,
    pub(crate) num_bf16: usize,
    pub(crate) num_int: usize,
    pub(crate) num_uint: usize,
    pub(crate) num_bool: usize,
}
