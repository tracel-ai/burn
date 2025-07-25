//! Global orchestrator
//!
//! Launches the orchestrator that responds to global collective operations for nodes for the
//! integration test
//!
//! This is necessary for any node who needs global collective operations

use std::env;

#[tokio::main]
/// Start the global orchestrator on the port given as first arg
pub async fn main() {
    let args: Vec<String> = env::args().collect();
    let port = args[1].parse::<u16>().expect("invalid port");

    // Launch the global orchestrator, which will listen and respond to global collective op
    // requests from nodes
    burn_collective::start_global_orchestrator(port).await;
}
