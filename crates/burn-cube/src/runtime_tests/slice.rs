use crate as burn_cube;
use burn_cube::prelude::*;

#[cube(launch)]
pub fn slice_select<F: Float>(input: &Array<F>, output: &mut Array<F>) {
    if UNIT_POS == UInt::new(0) {
        let slice = input.slice(2, 3);
        output[0] = slice[0u32];
    }
}

#[cube(launch)]
pub fn slice_assign<F: Float>(input: &Array<F>, output: &mut Array<F>) {
    if UNIT_POS == UInt::new(0) {
        let mut slice_1 = output.slice_mut(2, 3);
        slice_1[0] = input[0u32];
    }
}

#[cube(launch)]
pub fn slice_len<F: Float>(input: &Array<F>, output: &mut Array<UInt>) {
    if UNIT_POS == UInt::new(0) {
        let slice = input.slice(2, 4);
        let _tmp = slice[0]; // It must be used at least once, otherwise wgpu isn't happy.
        output[0] = slice.len();
    }
}

pub fn test_slice_select<R: Runtime>(client: ComputeClient<R::Server, R::Channel>) {
    let input = client.create(f32::as_bytes(&[0.0, 1.0, 2.0, 3.0, 4.0]));
    let output = client.empty(core::mem::size_of::<f32>());

    slice_select_launch::<F32, R>(
        client.clone(),
        CubeCount::Static(1, 1, 1),
        CubeDim::new(1, 1, 1),
        ArrayArg::new(&input, 5),
        ArrayArg::new(&output, 1),
    );

    let actual = client.read(output.binding());
    let actual = f32::from_bytes(&actual);

    assert_eq!(actual[0], 2.0);
}

pub fn test_slice_len<R: Runtime>(client: ComputeClient<R::Server, R::Channel>) {
    let input = client.create(f32::as_bytes(&[0.0, 1.0, 2.0, 3.0, 4.0]));
    let output = client.empty(core::mem::size_of::<u32>());

    slice_len_launch::<F32, R>(
        client.clone(),
        CubeCount::Static(1, 1, 1),
        CubeDim::new(1, 1, 1),
        ArrayArg::new(&input, 5),
        ArrayArg::new(&output, 1),
    );

    let actual = client.read(output.binding());
    let actual = u32::from_bytes(&actual);

    assert_eq!(actual, &[2]);
}

pub fn test_slice_assign<R: Runtime>(client: ComputeClient<R::Server, R::Channel>) {
    let input = client.create(f32::as_bytes(&[15.0]));
    let output = client.create(f32::as_bytes(&[0.0, 1.0, 2.0, 3.0, 4.0]));

    slice_assign_launch::<F32, R>(
        client.clone(),
        CubeCount::Static(1, 1, 1),
        CubeDim::new(1, 1, 1),
        ArrayArg::new(&input, 5),
        ArrayArg::new(&output, 1),
    );

    let actual = client.read(output.binding());
    let actual = f32::from_bytes(&actual);

    assert_eq!(actual, &[0.0, 1.0, 15.0, 3.0, 4.0]);
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_slice {
    () => {
        use super::*;

        #[test]
        fn test_slice_select() {
            let client = TestRuntime::client(&Default::default());
            burn_cube::runtime_tests::slice::test_slice_select::<TestRuntime>(client);
        }

        #[test]
        fn test_slice_assign() {
            let client = TestRuntime::client(&Default::default());
            burn_cube::runtime_tests::slice::test_slice_assign::<TestRuntime>(client);
        }

        #[test]
        fn test_slice_len() {
            let client = TestRuntime::client(&Default::default());
            burn_cube::runtime_tests::slice::test_slice_len::<TestRuntime>(client);
        }
    };
}
