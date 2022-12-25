pub trait Metric: Send + Sync {
    type Input;

    fn update(&mut self, item: &Self::Input) -> MetricEntry;
    fn clear(&mut self);
}

pub trait Adaptor<T> {
    fn adapt(&self) -> T;
}

pub trait Numeric {
    fn value(&self) -> f64;
}

#[derive(new)]
pub struct MetricEntry {
    pub name: String,
    pub formatted: String,
    pub serialize: String,
}

#[derive(new)]
pub struct RunningMetricResult {
    pub name: String,
    pub formatted: String,
    pub raw_running: String,
    pub raw_current: String,
}

impl Into<MetricEntry> for RunningMetricResult {
    fn into(self) -> MetricEntry {
        MetricEntry {
            name: self.name,
            formatted: self.formatted,
            serialize: self.raw_current,
        }
    }
}
