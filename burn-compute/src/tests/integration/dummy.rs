mod tests {
    use alloc::{boxed::Box, vec::Vec};

    use crate::tests::integration::dummy_server::{get, DummyDevice, DummyElementwiseAddition};

    #[test]
    fn created_resource_is_the_same_when_read() {
        let client = get(&DummyDevice);
        let resource = Vec::from([0, 1, 2]);
        let resource_description = client.create(resource.clone());

        let obtained_resource = client.read(&resource_description);

        assert_eq!(resource, obtained_resource)
    }

    #[test]
    fn empty_allocates_memory() {
        let client = get(&DummyDevice);
        let size = 4;
        let resource_description = client.empty(size);
        let empty_resource = client.read(&resource_description);

        assert_eq!(empty_resource.len(), 4);
    }

    #[test]
    fn execute_elementwise_addition() {
        let client = get(&DummyDevice);
        let lhs = client.create([0, 1, 2].into());
        let rhs = client.create([4, 4, 4].into());
        let out = client.empty(3);

        let kernel_description = Box::new(DummyElementwiseAddition);

        client.execute(kernel_description, &[&lhs, &rhs, &out]);

        let obtained_resource = client.read(&out);

        assert_eq!(obtained_resource, Vec::from([4, 5, 6]))
    }
}
