use std::marker::PhantomData;

use derive_new::new;

use crate::{ComputeClient, ComputeServer};

#[derive(new)]
pub struct ComputeChannel<KernelDescription, ResourceDescription, Resource> {
    server: Box<
        dyn ComputeServer<
            KernelDescription = KernelDescription,
            ResourceDescription = ResourceDescription,
            Resource = Resource,
        >,
    >,
    _kd: PhantomData<KernelDescription>,
    _rd: PhantomData<ResourceDescription>,
    _r: PhantomData<Resource>,
}

impl<KernelDescription, ResourceDescription, Resource>
    ComputeChannel<KernelDescription, ResourceDescription, Resource>
{
    pub fn init(
        server: Box<
            dyn ComputeServer<
                KernelDescription = KernelDescription,
                ResourceDescription = ResourceDescription,
                Resource = Resource,
            >,
        >,
    ) -> ComputeClient<KernelDescription, ResourceDescription, Resource> {
        let channel = ComputeChannel {
            server: server,
            _kd: PhantomData,
            _rd: PhantomData,
            _r: PhantomData,
        };

        ComputeClient::new(channel)
    }

    pub fn read(&self, resource_description: ResourceDescription) -> Resource {
        self.server.read(resource_description)
    }

    pub fn create(&self, resource: Resource) -> ResourceDescription {
        self.server.create(resource)
    }

    pub fn empty(&self, size: usize) -> ResourceDescription {
        self.server.empty(size)
    }

    pub fn execute(
        &self,
        kernel_description: KernelDescription,
        resource_descriptions: Vec<ResourceDescription>,
    ) {
        self.server
            .execute(kernel_description, resource_descriptions)
    }

    pub fn sync(&self) {
        self.server.sync()
    }
}
