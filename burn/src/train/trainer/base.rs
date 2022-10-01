pub trait Train<M, D> {
    fn train(self, model: M, data: D) -> M;
}
