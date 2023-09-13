// #![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;
mod channel;
mod client;
mod memory_management;
mod server;
pub use channel::*;

pub use client::*;
use dummy::{DummyKernelDescription, DummyResourceDescription};
pub use memory_management::*;
pub use server::*;
mod dummy;
type Server = dyn ComputeServer<
    KernelDescription = DummyKernelDescription,
    ResourceDescription = DummyResourceDescription,
>;

#[cfg(test)]
mod tests {

    use alloc::{boxed::Box, vec::Vec};
    use spin::Mutex;

    use crate::dummy::{
        DummyAllocator, DummyElementwiseAddition, DummyMemoryManagement, DummyServer,
    };

    use super::*;

    #[test]
    fn created_resource_is_the_same_when_read() {
        let client = make_client();
        let resource = Vec::from([0, 1, 2]);
        let resource_description = client.create(resource.clone());

        let obtained_resource = client.read(&resource_description);

        assert_eq!(resource, obtained_resource)
    }

    #[test]
    fn empty_allocates_memory() {
        let client = make_client();
        let size = 4;
        let resource_description = client.empty(size);
        let empty_resource = client.read(&resource_description);

        assert_eq!(empty_resource.len(), 4);
    }

    #[test]
    fn execute_elementwise_addition() {
        let client = make_client();
        let lhs = client.create([0, 1, 2].into());
        let rhs = client.create([4, 4, 4].into());
        let out = client.empty(3);

        let kernel_description =
            DummyKernelDescription::new(Box::new(DummyElementwiseAddition::new()));

        client.execute(kernel_description, [&lhs, &rhs, &out].into());

        let obtained_resource = client.read(&out);

        assert_eq!(obtained_resource, Vec::from([4, 5, 6]))
    }

    fn make_client() -> ComputeClient<DummyKernelDescription, DummyResourceDescription> {
        let memory_management = DummyMemoryManagement::new(Vec::new(), DummyAllocator::new());
        let server = DummyServer::new(Box::new(memory_management));
        let server_for_channel: Mutex<Box<Server>> = Mutex::new(Box::new(server));
        ComputeChannel::init(server_for_channel)
    }
}
