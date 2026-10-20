use burn_backend::ops::CollectiveOps;

use crate::{BackendRouter, RouterChannel};

impl<R: RouterChannel> CollectiveOps<Self> for BackendRouter<R> {}
