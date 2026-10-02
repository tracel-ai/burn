#![cfg(all(feature = "remote-tests", feature = "flex"))]
use burn_core::{
    backend::Flex,
    tensor::{Device, Tensor, TensorData, Tolerance, remote::RemoteHost},
};
use burn_remote::server::{BackendServer, WebSocketTransport};
use burn_signal::{irfft, rfft};
#[test]
pub fn test_fft_over_websocket() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_io()
        .build()
        .unwrap();

    let mut registry = burn_router::CustomOpRegistry::<Flex>::new();
    burn_signal::register_fft_ops(&mut registry);
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let host = RemoteHost::websocket(&format!("ws://{}", listener.local_addr().unwrap()));
    rt.spawn(
        BackendServer::<Flex>::new([Default::default()])
            .with_custom_ops(registry)
            .serve_async(WebSocketTransport::from_listener(listener)),
    );

    let device = Device::remote_options(&host).init().unwrap();
    let signal = Tensor::<1>::from_floats([1.0, 1.0, 1.0, 1.0], &device);
    let (spectrum_re, spectrum_im) = rfft(signal, 0, None);
    let reconstructed = irfft(spectrum_re.clone(), spectrum_im.clone(), 0, None);

    spectrum_re.into_data().assert_approx_eq::<f32>(
        &TensorData::from([4.0, 0.0, 0.0]),
        Tolerance::absolute(1e-4),
    );
    spectrum_im.into_data().assert_approx_eq::<f32>(
        &TensorData::from([0.0, 0.0, 0.0]),
        Tolerance::absolute(1e-4),
    );
    reconstructed.into_data().assert_approx_eq::<f32>(
        &TensorData::from([1.0, 1.0, 1.0, 1.0]),
        Tolerance::absolute(1e-4),
    );

    let input = Tensor::<1>::from_floats([0.2, -0.3, 0.7, 0.1], &device.autodiff()).require_grad();
    let (real, imag) = rfft(input.clone(), 0, None);
    let gradients = irfft(real, imag, 0, None).sum().backward();
    input
        .grad(&gradients)
        .unwrap()
        .into_data()
        .assert_approx_eq::<f32>(&TensorData::from([1.0f32; 4]), Tolerance::absolute(1e-4));

    rt.shutdown_background();
}
