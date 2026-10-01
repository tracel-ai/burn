use std::{net::ToSocketAddrs, str::FromStr};

use burn::{
    remote::{EndpointId, IrohPeer, IrohPeerBuilder, IrohRelays},
    tensor::Device,
};
use clap::{
    Args, CommandFactory, Parser, ValueEnum, builder::NonEmptyStringValueParser, error::ErrorKind,
};

/// Train the MNIST model on another machine's GPU, or classify test images with it there.
#[derive(Parser)]
struct Cli {
    mode: Mode,
    /// The server's Iroh id, printed when it starts, or its `ws://` URL.
    server: Server,
    #[command(flatten)]
    iroh: IrohArgs,
}

#[derive(Clone, Copy, ValueEnum)]
enum Mode {
    /// Train the model on the server's GPU and save it here.
    Train,
    /// Classify test images on the server's GPU with the saved model.
    Infer,
}

#[derive(Clone)]
enum Server {
    WebSocket { url: String },
    Iroh { id: EndpointId },
}

impl FromStr for Server {
    type Err = String;

    fn from_str(server: &str) -> Result<Self, Self::Err> {
        if server.starts_with("ws://") {
            return Ok(Self::WebSocket {
                url: server.to_string(),
            });
        }
        server
            .parse()
            .map(|id| Self::Iroh { id })
            .map_err(|_| format!("{server} is neither a ws:// URL nor an Iroh id"))
    }
}

/// Only for an Iroh server.
#[derive(Args)]
struct IrohArgs {
    /// The server's `REMOTE_BACKEND_TOKEN`.
    #[arg(long, env = "REMOTE_BACKEND_TOKEN", hide_env_values = true, value_parser = NonEmptyStringValueParser::new())]
    token: Option<String>,
    /// The server's relays: `public`, `disabled`, or the URL of a relay you run.
    #[arg(long, default_value = "public")]
    relays: IrohRelays,
    /// The server's `host:port`, required when relays are disabled.
    #[arg(long)]
    address: Option<String>,
}

impl IrohArgs {
    fn peer(self, id: EndpointId) -> Result<IrohPeer, clap::Error> {
        let token = self
            .token
            .ok_or_else(|| usage_error("an Iroh server needs --token or REMOTE_BACKEND_TOKEN"))?;
        let mut peer = IrohPeerBuilder::new(id)
            .with_relays(self.relays.clone())
            .with_credential(token);
        match (self.address, self.relays) {
            (Some(address), _) => {
                let addresses = address
                    .to_socket_addrs()
                    .map_err(|err| usage_error(&format!("--address {address}: {err}")))?;
                for address in addresses {
                    peer = peer.with_address(address);
                }
            }
            (None, IrohRelays::Disabled) => {
                return Err(usage_error(
                    "--address is required when relays are disabled",
                ));
            }
            (None, _) => {}
        }
        Ok(peer.build())
    }
}

fn usage_error(message: &str) -> clap::Error {
    Cli::command().error(ErrorKind::ArgumentConflict, message)
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();
    let device = match cli.server {
        Server::WebSocket { url } => {
            tokio::task::spawn_blocking(move || Device::remote_websocket(&url, 0))
                .await
                .expect("The WebSocket handshake completes")
        }
        Server::Iroh { id } => {
            let peer = cli.iroh.peer(id).unwrap_or_else(|err| err.exit());
            Device::remote_iroh_peer(&peer, 0)
                .await
                .expect("The server can be dialed")
        }
    };

    // Training and inference block for as long as they run, so they stay off the async workers.
    tokio::task::spawn_blocking(move || match cli.mode {
        Mode::Train => mnist::training::run(device),
        Mode::Infer => remote_mnist::inference::infer(&device),
    })
    .await
    .expect("The run completes");
}
