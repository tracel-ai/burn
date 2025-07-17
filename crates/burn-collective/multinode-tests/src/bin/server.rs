use std::env;

use burn_collective::server::start_ws;

#[tokio::main]
/// Start the server on the port given as first arg
pub async fn main() {
    let args: Vec<String> = env::args().collect();

    let port = args[1].parse::<u16>().expect("invalid port");
    start_ws(port).await;
}
