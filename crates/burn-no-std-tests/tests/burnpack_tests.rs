extern crate alloc;

#[test]
fn test_burnpack_no_std() {
    use burn_no_std_tests::burnpack;
    let device = Default::default();

    // Run all Burnpack tests
    burnpack::run_all_tests(&device);
}
