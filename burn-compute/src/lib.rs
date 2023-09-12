mod channel;
mod client;
mod memory_management;
mod server;
pub use channel::*;
pub use client::*;
pub use memory_management::*;
pub use server::*;
mod dummy;

#[cfg(test)]
mod tests {
    use crate::dummy::DummyServer;

    use super::*;

    #[test]
    fn dgds() {
        let server = DummyServer::new();
        let client = ComputeChannel::init(Box::new(server));
    }
}
