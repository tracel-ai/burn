use core::marker::PhantomData;

use alloc::{boxed::Box, vec::Vec};
use derive_new::new;
use spin::Mutex;

use crate::{ComputeClient, ComputeServer};

#[derive(new)]
pub struct ComputeChannel<KernelDescription, ResourceDescription> {
    server: Mutex<
        Box<
            dyn ComputeServer<
                KernelDescription = KernelDescription,
                ResourceDescription = ResourceDescription,
            >,
        >,
    >,
    _kd: PhantomData<KernelDescription>,
    _rd: PhantomData<ResourceDescription>,
}

impl<'a, KernelDescription, ResourceDescription>
    ComputeChannel<KernelDescription, ResourceDescription>
{
    pub fn init(
        server: Mutex<
            Box<
                dyn ComputeServer<
                    KernelDescription = KernelDescription,
                    ResourceDescription = ResourceDescription,
                >,
            >,
        >,
    ) -> ComputeClient<KernelDescription, ResourceDescription> {
        let channel = ComputeChannel {
            server: server,
            _kd: PhantomData,
            _rd: PhantomData,
        };

        ComputeClient::new(channel)
    }

    pub fn read(&self, resource_description: &ResourceDescription) -> Vec<u8> {
        self.server.lock().read(resource_description)
    }

    pub fn create(&self, resource: Vec<u8>) -> ResourceDescription {
        self.server.lock().create(resource)
    }

    pub fn empty(&self, size: usize) -> ResourceDescription {
        self.server.lock().empty(size)
    }

    pub fn execute(
        &self,
        kernel_description: KernelDescription,
        resource_descriptions: Vec<&ResourceDescription>,
    ) {
        self.server
            .lock()
            .execute(kernel_description, resource_descriptions)
    }

    pub fn sync(&self) {
        self.server.lock().sync()
    }
}
