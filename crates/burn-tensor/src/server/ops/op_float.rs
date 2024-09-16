use crate::server::Server;
use crate::{ops::FloatTensorOps, server::ServerBackend};

impl<B: ServerBackend> FloatTensorOps<Self> for Server<B> {}
