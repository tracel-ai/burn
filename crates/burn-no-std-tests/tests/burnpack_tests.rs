extern crate alloc;

#[test]
fn test_burnpack_no_std() {
    use burn_ndarray::NdArray;
    use burn_no_std_tests::burnpack;
    type Backend = NdArray<f32>;
    let device = Default::default();

    // Run all Burnpack tests
    burnpack::run_all_tests::<Backend>(&device);
}
