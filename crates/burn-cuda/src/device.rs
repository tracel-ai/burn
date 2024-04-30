use burn_tensor::backend::{DeviceId, DeviceOps};

#[derive(new, Clone, Debug, PartialEq, Eq, Default, Hash)]
pub struct CudaDevice {
    pub index: usize,
}

impl DeviceOps for CudaDevice {
    fn id(&self) -> DeviceId {
        DeviceId::new(0, self.index as u32)
    }
}
