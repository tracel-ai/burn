use crate as burn_cube;
use burn_cube::prelude::*;

#[cube(launch)]
pub fn kernel_cmma(output: &mut Array<F32>, input: &Array<F32>) {
    if UNIT_POS == UInt::new(0) {
        let matrix = cmma::Matrix::<F32>::new(cmma::MatrixIdent::A, 16, 16, 16, None);
        cmma::fill::<F32>(matrix, F32::new(0.0));

        output[0] = input[0];
    }
}

pub fn test_kernel_with_generics<R: Runtime>(client: ComputeClient<R::Server, R::Channel>) {
    let output = client.create(f32::as_bytes(&[0.0, 1.0]));
    let input = client.create(f32::as_bytes(&[0.0, 1.0]));

    kernel_cmma_launch::<R>(
        client.clone(),
        CubeCount::new(1, 1, 1),
        KernelSettings::default(),
        ArrayHandle::new(&input, 2),
        ArrayHandle::new(&output, 2),
    );

    let actual = client.read(handle.binding()).read_sync().unwrap();
    let actual = f32::from_bytes(&actual);

    assert_eq!(actual[0], 5.0);
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_cmma {
    () => {
        use super::*;

        #[test]
        fn test_aa() {
            let client = TestRuntime::client(&Default::default());
            burn_cube::runtime_tests::cmma::test_kernel_with_generics::<TestRuntime>(client);
        }
    };
}
