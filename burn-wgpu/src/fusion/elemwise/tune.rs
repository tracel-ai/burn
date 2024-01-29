use crate::{compute::WgpuAutotuneKey, fusion::kernel::AutotuneFusionKernel};
use burn_compute::tune::{AutotuneOperation, AutotuneOperationSet};

#[derive(new)]
pub struct ElementWiseAutotuneOperationSet {
    kernel_1: AutotuneFusionKernel,
    kernel_2: AutotuneFusionKernel,
    kernel_default: AutotuneFusionKernel,
}

impl AutotuneOperationSet<WgpuAutotuneKey> for ElementWiseAutotuneOperationSet {
    fn key(&self) -> WgpuAutotuneKey {
        WgpuAutotuneKey::ElemWise(())
    }

    fn autotunables(&self) -> Vec<Box<dyn burn_compute::tune::AutotuneOperation>> {
        let kernel_1: Box<dyn AutotuneOperation> = self.kernel_1.clone();
        let kernel_2: Box<dyn AutotuneOperation> = self.kernel_2.clone();

        vec![kernel_1, kernel_2]
    }

    fn fastest(self: Box<Self>, _: usize) -> Box<dyn AutotuneOperation> {
        Box::new(self.kernel_default)
    }
}
