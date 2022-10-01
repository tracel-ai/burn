pub trait Fit<D, O> {
    fn fit(self, data: D) -> O;
}
