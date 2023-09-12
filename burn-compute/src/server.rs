pub trait ComputeServer {
    type KernelDescription;
    type ResourceDescription;
    type Resource: Sized; // in WGPU, this would be Vec<u8>

    fn read(&self, resource_description: Self::ResourceDescription) -> Self::Resource;

    fn create(&self, resource: Self::Resource) -> Self::ResourceDescription;

    fn empty(&self, size: usize) -> Self::ResourceDescription;

    fn execute(
        &self,
        kernel_description: Self::KernelDescription,
        resource_descriptions: Vec<Self::ResourceDescription>,
    );

    fn sync(&self);
}
