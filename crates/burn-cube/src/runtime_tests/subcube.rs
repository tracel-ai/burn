use crate::Runtime;

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_subcube {
    () => {
        use super::*;

        #[test]
        fn test_subcube_sum() {
            let client = TestRuntime::client(&Default::default());
            burn_cube::runtime_tests::subcube::test_subcube_sum::<TestRuntime>(client);
        }
    };
}

use crate as burn_cube;
use crate::prelude::*;

pub fn test_subcube_sum<TestRuntime: Runtime>(
    client: ComputeClient<TestRuntime::Server, TestRuntime::Channel>,
) {
    #[cube(launch)]
    fn kernel<F: Float>(mut output: Tensor<F>) {
        let val = output[ABSOLUTE_INDEX];
        let val2 = subcube_sum::<F>(val);

        if ABSOLUTE_INDEX == UInt::new(0) {
            output[0] = val2;
        }
    }

    let handle = client.create(f32::as_bytes(&[4.0, 5.0, 7.0]));
    let (shape, strides) = ([3], [1]);

    kernel_launch::<F32, TestRuntime>(
        client.clone(),
        WorkGroup::new(1, 1, 1),
        CompilationSettings::default(),
        TensorHandle::new(&handle, &strides, &shape),
    );

    let actual = client.read(handle.binding()).read_sync().unwrap();
    let actual = f32::from_bytes(&actual);

    assert_eq!(actual, [16.0, 5.0, 7.0]);
}
