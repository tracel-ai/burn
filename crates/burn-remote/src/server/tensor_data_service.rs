use std::{collections::HashMap, sync::Mutex};

use burn_ir::TensorId;

pub struct TensorExposeState {
    pub bytes: bytes::Bytes,
    pub total_upload_count: u32,
    pub cur_upload_count: u32,
}

pub struct TensorDataService {
    pub exposed_tensors: Mutex<HashMap<TensorId, TensorExposeState>>,
}
