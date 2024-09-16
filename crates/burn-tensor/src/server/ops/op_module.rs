use crate::server::Server;
use crate::{
    ops::{ConvOptions, ConvTransposeOptions, FloatTensor, ModuleOps},
    server::ServerBackend,
};

impl<B: ServerBackend> ModuleOps<Self> for Server<B> {}
