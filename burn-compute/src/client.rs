use alloc::vec::Vec;
use derive_new::new;

use crate::channel::ComputeChannel;

#[derive(new)]
pub struct ComputeClient<KernelDescription, ResourceDescription> {
    channel: ComputeChannel<KernelDescription, ResourceDescription>,
}

impl<KernelDescription, ResourceDescription> ComputeClient<KernelDescription, ResourceDescription> {
    pub fn read(&self, resource_description: &ResourceDescription) -> Vec<u8> {
        self.channel.read(resource_description)
    }

    pub fn create(&self, resource: Vec<u8>) -> ResourceDescription {
        self.channel.create(resource)
    }

    pub fn empty(&self, size: usize) -> ResourceDescription {
        self.channel.empty(size)
    }

    pub fn execute(
        &self,
        kernel_description: KernelDescription,
        resource_descriptions: Vec<&ResourceDescription>,
    ) {
        self.channel
            .execute(kernel_description, resource_descriptions)
    }

    pub fn sync(&self) {
        self.channel.sync()
    }
}
