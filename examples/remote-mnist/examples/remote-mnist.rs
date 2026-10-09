use std::{net::ToSocketAddrs, str::FromStr};

use burn::{
    remote::{EndpointId, IrohHost, IrohRelays, RemoteHost},
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
    /// The server's `REMOTE_BACKEND_TOKEN`, required by an Iroh server.
    #[arg(long, env = "REMOTE_BACKEND_TOKEN", hide_env_values = true, value_parser = NonEmptyStringValueParser::new())]
    token: Option<String>,
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
    /// The server's relays: `public`, `disabled`, or the URL of a relay you run.
    #[arg(long, default_value = "public")]
    relays: IrohRelays,
    /// The server's `host:port`, required when relays are disabled.
    #[arg(long)]
    address: Option<String>,
}

impl IrohArgs {
    fn host(self, id: EndpointId) -> Result<IrohHost, clap::Error> {
        let mut host = IrohHost::new(id).with_relays(self.relays.clone());
        match (self.address, self.relays) {
            (Some(address), _) => {
                let addresses = address
                    .to_socket_addrs()
                    .map_err(|err| usage_error(&format!("--address {address}: {err}")))?;
                for address in addresses {
                    host = host.with_address(address);
                }
            }
            (None, IrohRelays::Disabled) => {
                return Err(usage_error(
                    "--address is required when relays are disabled",
                ));
            }
            (None, _) => {}
        }
        Ok(host)
    }
}

impl Cli {
    fn host(self) -> Result<RemoteHost, clap::Error> {
        let host = match self.server {
            Server::WebSocket { url } => RemoteHost::websocket(&url),
            Server::Iroh { id } if self.token.is_some() => RemoteHost::iroh(self.iroh.host(id)?),
            Server::Iroh { .. } => {
                return Err(usage_error(
                    "an Iroh server needs --token or REMOTE_BACKEND_TOKEN",
                ));
            }
        };
        Ok(match self.token {
            Some(token) => host.with_credential(token),
            None => host,
        })
    }
}

fn usage_error(message: &str) -> clap::Error {
    Cli::command().error(ErrorKind::ArgumentConflict, message)
}

fn main() {
    let cli = Cli::parse();
    let mode = cli.mode;
    let host = cli.host().unwrap_or_else(|err| err.exit());
    let device = Device::remote_options(&host)
        .init()
        .expect("The server can be dialed");

    match mode {
        Mode::Train => mnist::training::run(device),
        Mode::Infer => remote_mnist::inference::infer(&device),
    }
}
