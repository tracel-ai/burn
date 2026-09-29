use core::str::FromStr;

use iroh::{
    Endpoint, RelayMode, RelayUrl,
    endpoint::{Builder, QuicTransportConfig, presets},
};

/// How an Iroh endpoint reaches peers it cannot dial directly. A server and its clients must agree.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum IrohRelays {
    /// n0's public relays, with n0's address lookup, so a server is found by its id alone.
    #[default]
    Public,
    /// A relay you run; nothing goes through n0.
    Private(RelayUrl),
    /// Direct connections only; clients must be given the server's address.
    Disabled,
}

impl FromStr for IrohRelays {
    type Err = String;

    /// `public`, `disabled`, or a private relay's URL.
    fn from_str(relays: &str) -> Result<Self, Self::Err> {
        match relays {
            "public" => Ok(Self::Public),
            "disabled" => Ok(Self::Disabled),
            url => url.parse().map(Self::Private).map_err(|err| {
                format!("relays are `public`, `disabled` or a relay URL, got `{url}`: {err}")
            }),
        }
    }
}

impl IrohRelays {
    /// An endpoint builder with these relays, sending segmentation-offloaded batches only if asked.
    pub(crate) fn endpoint_builder(&self, segmentation_offload: bool) -> Builder {
        let builder = match self {
            Self::Public => Endpoint::builder(presets::N0),
            Self::Private(url) => {
                Endpoint::builder(presets::Minimal).relay_mode(RelayMode::custom([url.clone()]))
            }
            Self::Disabled => Endpoint::builder(presets::Minimal).relay_mode(RelayMode::Disabled),
        };
        builder.transport_config(
            QuicTransportConfig::builder()
                .enable_segmentation_offload(segmentation_offload)
                .build(),
        )
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
            Ok(IrohRelays::Private(url))
        );
        assert!("off".parse::<IrohRelays>().is_err());
    }
}
