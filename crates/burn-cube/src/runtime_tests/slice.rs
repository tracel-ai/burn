use crate as burn_cube;
use burn_cube::prelude::*;

#[cube(launch)]
pub fn slice_select<F: Float>(input: &Array<F>, output: &mut Array<F>) {
    if UNIT_POS == UInt::new(0) {
        let slice = Slice::from_array(input, 2, 3);
        output[0] = slice[0u32];
    }
}

#[cube(launch)]
pub fn slice_assign<F: Float>(input: &Array<F>, output: &mut Array<F>) {
    if UNIT_POS == UInt::new(0) {
        let mut slice = SliceMut::from_array(output, 2, 3);
        slice[0] = input[0u32];
    }
}

#[cube(launch)]
pub fn slice_len<F: Float>(input: &Array<F>, output: &mut Array<F>) {
    if UNIT_POS == UInt::new(0) {
        let slice = Slice::from_array(input, 2, 3);
        output[0] = slice[0u32];
    }
}

pub fn test_slice_select<R: Runtime>(client: ComputeClient<R::Server, R::Channel>) {
    let input = client.create(f32::as_bytes(&[0.0, 1.0, 2.0, 3.0, 4.0]));
    let output = client.empty(1 * core::mem::size_of::<f32>());

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
        fn test_slice_aa() {
            let client = TestRuntime::client(&Default::default());
            burn_cube::runtime_tests::slice::test_slice_select::<TestRuntime>(client);
        }

        #[test]
        fn test_slice_bb() {
            let client = TestRuntime::client(&Default::default());
            burn_cube::runtime_tests::slice::test_slice_assign::<TestRuntime>(client);
        }
    };
}
