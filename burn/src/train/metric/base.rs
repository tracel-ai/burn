pub trait Metric<T>: Send + Sync {
    fn update(&mut self, item: &T) -> MetricStateDyn;
    fn clear(&mut self);
}

pub trait MetricState {
    fn name(&self) -> String;
    fn pretty(&self) -> String;
    fn serialize(&self) -> String;
}

pub trait Numeric {
    fn value(&self) -> f64;
}

pub type MetricStateDyn = Box<dyn MetricState>;

#[derive(new)]
pub struct RunningMetricResult {
    pub name: String,
    pub formatted: String,
    pub raw_running: String,
    pub raw_current: String,
}

impl MetricState for RunningMetricResult {
    fn name(&self) -> String {
        self.name.clone()
    }

    fn pretty(&self) -> String {
        self.formatted.clone()
    }

    fn serialize(&self) -> String {
        self.raw_current.clone()
    }
}
