use crate::server::Server;
use crate::{ops::IntTensorOps, server::ServerBackend};

impl<B: ServerBackend> IntTensorOps<Self> for Server<B> {}
