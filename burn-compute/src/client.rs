use derive_new::new;

use crate::channel::ComputeChannel;

#[derive(new)]
pub struct ComputeClient<KernelDescription, ResourceDescription, Resource> {
    channel: ComputeChannel<KernelDescription, ResourceDescription, Resource>,
}

impl<KernelDescription, ResourceDescription, Resource>
    ComputeClient<KernelDescription, ResourceDescription, Resource>
{
    pub fn read(&self, resource_description: ResourceDescription) -> Resource {
        self.channel.read(resource_description)
    }

    pub fn create(&self, resource: Resource) -> ResourceDescription {
        self.channel.create(resource)
    }

    pub fn empty(&self, size: usize) -> ResourceDescription {
        self.channel.empty(size)
    }

    pub fn execute(
        &self,
        kernel_description: KernelDescription,
        resource_descriptions: Vec<ResourceDescription>,
    ) {
        self.channel
            .execute(kernel_description, resource_descriptions)
    }

    pub fn sync(&self) {
        self.channel.sync()
    }
}
