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
    /// The minimum value of the linear region range. Default is -1.0
    #[config(default = "-1.0")]
    pub min_val: f64,
    /// The maximum value of the linear region range. Default is 1.0
    #[config(default = "1.0")]
    pub max_val: f64,
}

impl HardtanhConfig {
    /// Initialize a new [Hardtanh](Hardtanh) layer.
    pub fn init(&self) -> Hardtanh {
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
}
