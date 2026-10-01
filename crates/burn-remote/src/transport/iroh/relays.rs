use core::{fmt, str::FromStr};

#[cfg(all(
    any(feature = "client", feature = "server"),
    not(target_family = "wasm")
))]
use burn_std::config::config;
use iroh::RelayUrl;
#[cfg(all(
    any(feature = "client", feature = "server"),
    not(target_family = "wasm")
))]
use iroh::endpoint::QuicTransportConfig;
#[cfg(any(
    feature = "client",
    all(feature = "server", not(target_family = "wasm"))
))]
use iroh::{
    Endpoint, RelayMode,
    endpoint::{Builder, presets},
};

/// How an Iroh endpoint reaches peers it cannot dial directly. A server and its clients must agree.
///
/// Written and parsed as `public`, `disabled`, or a private relay's URL.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum IrohRelays {
    /// n0's public relays, with n0's address lookup, so a server is found by its id alone.
    #[default]
    Public,
    /// A relay you run; nothing goes through n0.
    Private {
        /// Where the relay listens.
        url: RelayUrl,
    },
    /// Direct connections only; clients must be given the server's address.
    Disabled,
}

impl FromStr for IrohRelays {
    type Err = String;

    fn from_str(relays: &str) -> Result<Self, Self::Err> {
        match relays {
            "public" => Ok(Self::Public),
            "disabled" => Ok(Self::Disabled),
            url => url.parse().map(|url| Self::Private { url }).map_err(|err| {
                format!("relays are `public`, `disabled` or a relay URL, got `{url}`: {err}")
            }),
        }
    }
}

impl fmt::Display for IrohRelays {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Public => f.write_str("public"),
            Self::Private { url } => url.fmt(f),
            Self::Disabled => f.write_str("disabled"),
        }
    }
}

impl IrohRelays {
    /// An endpoint builder with these relays.
    #[cfg(any(
        feature = "client",
        all(feature = "server", not(target_family = "wasm"))
    ))]
    pub(crate) fn endpoint_builder(&self) -> Builder {
        let builder = match self {
            Self::Public => Endpoint::builder(presets::N0),
            Self::Private { url } => {
                Endpoint::builder(presets::Minimal).relay_mode(RelayMode::custom([url.clone()]))
            }
            Self::Disabled => Endpoint::builder(presets::Minimal).relay_mode(RelayMode::Disabled),
        };
        // Drop with iroh#4555.
        #[cfg(not(target_family = "wasm"))]
        let builder = {
            let segmentation_offload = config().remote().iroh_segmentation_offload;
            builder.transport_config(
                QuicTransportConfig::builder()
                    .enable_segmentation_offload(segmentation_offload)
                    .build(),
            )
        };
        builder
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relays_parse_from_their_names_or_a_url() {
        assert_eq!("public".parse(), Ok(IrohRelays::Public));
        assert_eq!("disabled".parse(), Ok(IrohRelays::Disabled));
        let url: RelayUrl = "https://relay.example.com".parse().unwrap();
        assert_eq!(
            "https://relay.example.com".parse(),
            Ok(IrohRelays::Private { url })
        );
        assert!("off".parse::<IrohRelays>().is_err());
    }

    #[test]
    fn relays_parse_back_from_how_they_are_written() {
        let private = IrohRelays::Private {
            url: "https://relay.example.com".parse().unwrap(),
        };
        for relays in [IrohRelays::Public, private, IrohRelays::Disabled] {
            assert_eq!(relays.to_string().parse(), Ok(relays));
        }
    }
}
