#[burn_tensor_testgen::testgen(container)]
mod test {
    use super::*;
    use burn_tensor::{
        container::{TensorContainer, TensorContainerError},
        TensorPrimitive,
    };
    #[test]
    fn test_registered_id_should_return_ok() {
        let tensor: TensorPrimitive<TestBackend> = TestTensor::<1>::from([3.0,4.0, -5.0]).into_primitive();
        let mut container = TensorContainer::new();
        let registered_id = 1;
        container.register(registered_id,tensor);
        let extraction_result:Result<TensorPrimitive<TestBackend>,TensorContainerError>  = container.get(&registered_id);
        assert!(extraction_result.is_ok())
    }

    #[test]
    fn test_empty_container_should_return_not_found() {
        let mut container = TensorContainer::new();
        let unregistered_id = 1;
        let extraction_result: Result<TensorPrimitive<TestBackend>,TensorContainerError> = container.get(&unregistered_id);
        match extraction_result {
            Ok(_) => assert!(
                false,
                "Found an registered tensor on {} on even though the container should be empty",
                unregistered_id,
            ),
            Err(err) => {
                assert_eq!(err, TensorContainerError::NotFound);
            }
        }
    }

    #[test]
    fn test_unregistered_id_should_return_not_found() {
        let tensor = TestTensor::<1>::from([3.0, 4.0, -5.0]).into_primitive();
        let mut container = TensorContainer::new();
        let registered_id = 1;
        let unregistered_id = 2;
        container.register(registered_id, tensor);
        let extraction_result: Result<TensorPrimitive<TestBackend>,TensorContainerError> = container.get(&unregistered_id);

        match extraction_result {
            Ok(_) => assert!(
                false,
                "Found an entry on {} even though there only should be one on {}",
                unregistered_id,
                registered_id,
            ),
            Err(err) => {
                assert_eq!(err, TensorContainerError::NotFound);
            }
        }
    }
}