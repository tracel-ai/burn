#[derive(new, Clone, Debug, PartialEq, Eq, Default, Hash)]
pub struct CudaDevice {
    pub index: usize,
}

#[cfg(feature = "fusion")]
mod fusion {
    use super::*;
    use burn_fusion::{DeviceId, FusionDevice};

    impl FusionDevice for CudaDevice {
        fn id(&self) -> DeviceId {
            DeviceId::new(0, self.index as u32)
        }
    }
}
