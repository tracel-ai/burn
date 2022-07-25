#[derive(Clone, Debug)]
pub struct ADKind<P> {
    _p: P,
}

impl<P: Default> ADKind<P> {
    pub fn new() -> Self {
        Self { _p: P::default() }
    }
}
