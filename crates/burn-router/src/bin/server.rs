#[cfg(not(feature = "http"))]
fn main() {
    println!("You need to activate the feature `http-server` to run a server.");
}

#[cfg(feature = "http")]
fn main() {
    #[cfg(feature = "wgpu")]
    {
        type Backend = burn_wgpu::Wgpu;
        burn_router::http::server::start::<Backend>(Default::default(), "0.0.0.0:3000")
    }
}
