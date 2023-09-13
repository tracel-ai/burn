use alloc::vec::Vec;

pub trait ComputeServer {
    type KernelDescription;
    type ResourceDescription;

    fn read(&mut self, resource_description: &Self::ResourceDescription) -> Vec<u8>;

    fn create(&mut self, resource: Vec<u8>) -> Self::ResourceDescription;

    fn empty(&mut self, size: usize) -> Self::ResourceDescription;

    fn execute(
        &mut self,
        kernel_description: Self::KernelDescription,
        resource_descriptions: Vec<&Self::ResourceDescription>,
    );

    fn sync(&self);
}
