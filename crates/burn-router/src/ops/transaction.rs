use burn_backend::ops::TransactionOps;

use crate::{BackendRouter, RouterChannel};

impl<R: RouterChannel> TransactionOps<Self> for BackendRouter<R> {}
