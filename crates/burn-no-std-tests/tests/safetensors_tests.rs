extern crate alloc;

#[test]
fn test_safetensors_no_std() {
    use burn_no_std_tests::safetensors;
    let device = Default::default();

    // Run all SafeTensors tests
    safetensors::run_all_tests(&device);
}
