use p2p_remote_training::{run_client, run_server};

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("server") => {
            let topic = args.get(2).map(String::as_str).unwrap_or("burn-default");
            run_server(topic).await;
        }
        Some("client") => {
            let topic = args.get(2).map(String::as_str).unwrap_or("burn-default");
            run_client(topic).await;
        }
        _ => {
            eprintln!("usage:");
            eprintln!("  server [topic]   start compute server");
            eprintln!("  client [topic]   connect and train");
            eprintln!();
            eprintln!("both sides must use the same topic string");
            eprintln!("example: cargo run --example p2p-remote-training -- server my-cluster");
            std::process::exit(1);
        }
    }
}
