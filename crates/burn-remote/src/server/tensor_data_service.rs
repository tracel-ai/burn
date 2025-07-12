use std::collections::HashMap;

use burn_ir::TensorId;
use tokio::sync::Mutex;

pub struct TensorExposeState {
    pub bytes: bytes::Bytes,
    pub max_downloads: u32,
    pub cur_download_count: u32,
}

pub struct TensorDataService {
    pub exposed_tensors: Mutex<HashMap<TensorId, TensorExposeState>>,
}
