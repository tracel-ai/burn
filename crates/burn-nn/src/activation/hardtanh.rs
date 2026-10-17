use burn::config::Config;
use burn::module::Module;
use burn::module::{Content, DisplaySettings, ModuleDisplay};
use burn::tensor::Tensor;
use burn_core as burn;

use burn::tensor::activation::hardtanh;

/// HardTanh layer, clamping each element to the range `[min_val, max_val]`.
///
/// Should be created with [HardtanhConfig](HardtanhConfig).
#[derive(Module, Debug)]
#[module(custom_display)]
pub struct Hardtanh {
    /// The minimum value of the linear region range.
    pub min_val: f64,
    /// The maximum value of the linear region range.
    pub max_val: f64,
}

/// Configuration to create a [Hardtanh](Hardtanh) layer using the [init function](HardtanhConfig::init).
#[derive(Config, Debug)]
pub struct HardtanhConfig {
    /// The finite minimum value of the linear region range. Default is -1.0.
    /// Must be less than or equal to `max_val`.
    #[config(default = "-1.0")]
    pub min_val: f64,
    /// The finite maximum value of the linear region range. Default is 1.0.
    /// Must be greater than or equal to `min_val`.
    #[config(default = "1.0")]
    pub max_val: f64,
}

impl HardtanhConfig {
    /// Initialize a new [Hardtanh](Hardtanh) layer.
    pub fn init(&self) -> Hardtanh {
        if !self.min_val.is_finite() || !self.max_val.is_finite() {
            panic!(
                "Hardtanh bounds must be finite, but got min_val={} and max_val={}",
                self.min_val, self.max_val
            );
        }
        if self.min_val > self.max_val {
            panic!(
                "Hardtanh min_val must be less than or equal to max_val, but got min_val={} and max_val={}",
                self.min_val, self.max_val
            );
        }

        Hardtanh {
            min_val: self.min_val,
            max_val: self.max_val,
        }
    }
}

impl ModuleDisplay for Hardtanh {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new()
            .with_new_line_after_attribute(false)
            .optional()
    }

    fn custom_content(&self, content: Content) -> Option<Content> {
        content
            .add("min_val", &self.min_val)
            .add("max_val", &self.max_val)
            .optional()
    }
}

impl Hardtanh {
    /// Forward pass for the HardTanh layer.
    ///
    /// See [hardtanh](burn::tensor::activation::hardtanh) for more information.
    ///
    /// # Shapes
    /// - input: `[..., any]`
    /// - output: `[..., any]`
    pub fn forward<const D: usize>(&self, input: Tensor<D>) -> Tensor<D> {
        hardtanh(input, self.min_val, self.max_val)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display() {
        let config = HardtanhConfig::new().init();
        assert_eq!(
            alloc::format!("{config}"),
            "Hardtanh {min_val: -1, max_val: 1}"
        );
    }

    #[test]
    #[should_panic(expected = "Hardtanh min_val must be less than or equal to max_val")]
    fn inverted_bounds_should_panic() {
        HardtanhConfig::new()
            .with_min_val(1.0)
            .with_max_val(-1.0)
            .init();
    }

    #[test]
    #[should_panic(expected = "Hardtanh bounds must be finite")]
    fn nan_bound_should_panic() {
        HardtanhConfig::new().with_min_val(f64::NAN).init();
    }

    #[test]
    #[should_panic(expected = "Hardtanh bounds must be finite")]
    fn infinite_bound_should_panic() {
        HardtanhConfig::new().with_max_val(f64::INFINITY).init();
    }

    #[test]
    fn equal_bounds_are_valid() {
        let layer = HardtanhConfig::new()
            .with_min_val(0.5)
            .with_max_val(0.5)
            .init();

        assert_eq!(layer.min_val, layer.max_val);
    }
}
