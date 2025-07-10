use std::env;

use burn::backend::collective::global::server::base::start;
use burn_network::websocket::server::WsServer;

#[tokio::main]
/// Start the server on the given port and [device](Device).
pub async fn main() {
    let args: Vec<String> = env::args().collect();

    let port = args[1].parse::<u16>().expect("invalid port");
    start::<WsServer>(port).await;
}
