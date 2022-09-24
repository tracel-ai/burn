use textplots::{Chart, Plot, Shape};

pub struct TextPlot {
    train: Vec<f32>,
    valid: Vec<f32>,
}

impl TextPlot {
    pub fn new() -> Self {
        Self {
            train: Vec::new(),
            valid: Vec::new(),
        }
    }

    pub fn merge(self, other: Self) -> Self {
        let mut other = other;
        let mut train = self.train;
        let mut valid = self.valid;

        train.append(&mut other.train);
        valid.append(&mut other.valid);

        Self { train, valid }
    }

    pub fn update_train(&mut self, item: f32) {
        self.train.push(item);
    }

    pub fn update_valid(&mut self, item: f32) {
        self.valid.push(item);
    }

    pub fn render(&self) -> String {
        Chart::new(256, 32, 0.0, self.train.len() as f32)
            .lineplot(&Shape::Lines(&smooth_values(&self.train, 256)))
            .lineplot(&Shape::Lines(&smooth_values(&self.valid, 256)))
            .to_string()
            + "\n\n"
    }
}

fn smooth_values(values: &Vec<f32>, size_appox: usize) -> Vec<(f32, f32)> {
    let batch_size = values.len() / size_appox;
    if batch_size == 0 {
        return values
            .iter()
            .enumerate()
            .map(|(i, v)| (i as f32, *v as f32))
            .collect();
    }

    let mut output = Vec::with_capacity(size_appox);
    let mut current_sum = 0.0;
    let mut current_count = 0;

    for value in values.iter() {
        current_sum += value;
        current_count += 1;

        if current_count >= batch_size {
            output.push(current_sum / current_count as f32);
        }
    }

    if current_count > 0 {
        output.push(current_sum / current_count as f32);
    }

    output
        .iter()
        .enumerate()
        .map(|(i, v)| (i as f32, *v as f32))
        .collect()
}
