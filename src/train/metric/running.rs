pub trait RunningMetric<T>: Send + Sync {
    fn update(&mut self, item: &T) -> RunningMetricResult;
    fn clear(&mut self);
}

#[derive(new)]
pub struct RunningMetricResult {
    pub name: String,
    pub formatted: String,
    pub raw_running: String,
    pub raw_current: String,
}
