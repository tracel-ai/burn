use crate::server::Server;
use crate::{ops::ActivationOps, server::ServerBackend};

impl<B: ServerBackend> ActivationOps<Self> for Server<B> {}
