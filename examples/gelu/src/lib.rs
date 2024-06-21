use burn_cube::prelude::*;

#[cube(launch)]
fn gelu<F: Float>(input: Array<F>, mut output: Array<F>) {
    if ABSOLUTE_POS < input.len() {
        output[ABSOLUTE_POS] = gelu_scalar::<F>(input[ABSOLUTE_POS]);
    }
}

#[cube]
fn gelu_scalar<F: Float>(x: F) -> F {
    x * (F::new(1.0) + F::erf(x / F::sqrt(F::new(2.0)))) / F::new(2.0)
}

pub fn launch<R: Runtime>(device: &R::Device) {
    let client = R::client(device);
    println!("Executing gelu with runtime {:?}", R::name());

    let input = &[-1., 0., 1., 5.];
    let input_handle = client.create(f32::as_bytes(input));
    let output_handle = client.empty(input.len() * core::mem::size_of::<f32>());

    gelu_launch::<F32, R>(
        client.clone(),
        CubeCount::new(1, 1, 1),
        KernelSettings::default(),
        ArrayHandle::new(&input_handle, input.len()),
        ArrayHandle::new(&output_handle, input.len()),
    );

    let output = client.read(output_handle.binding()).read_sync().unwrap();
    let output = f32::from_bytes(&output);

    // Should be [-0.1587,  0.0000,  0.8413,  5.0000]
    println!("{output:?}");
}
