use derive_new::new;

use crate::ComputeServer;

#[derive(new)]
pub struct DummyServer {}
pub struct DummyKernelDescription {}
pub struct DummyResourceDescription {}

impl ComputeServer for DummyServer {
    type KernelDescription = DummyKernelDescription;

    type ResourceDescription = DummyResourceDescription;

    type Resource = Vec<u8>;

    fn read(&self, resource_description: Self::ResourceDescription) -> Self::Resource {
        todo!()
    }

    fn create(&self, resource: Self::Resource) -> Self::ResourceDescription {
        todo!()
    }

    fn empty(&self, size: usize) -> Self::ResourceDescription {
        todo!()
    }

    fn execute(
        &self,
        kernel_description: Self::KernelDescription,
        resource_descriptions: Vec<Self::ResourceDescription>,
    ) {
        todo!()
    }

    fn sync(&self) {
        todo!()
    }
}
