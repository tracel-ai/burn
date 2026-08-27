use alloc::format;
use core::fmt::{Display, Formatter, Result as FmtResult};

use burn_tensor::{Device, Shape};

use crate::module::{
    AutodiffModule, Content, Devices, Module, ModuleDisplay, ModuleDisplayDefault, ModuleMapper,
    ModuleVisitor, Param, ParamId, Parameter,
};

/// A boolean parameter value used for module-owned control state.
///
/// [`Param<Flag>`] gives the value an identity in the module tree, allowing parameter groups to
/// select it by id or path just like other module-owned values. The field holding the flag gives
/// it its meaning; for example, `Dropout::training` uses an enabled flag to permit stochastic
/// behavior on an autodiff device.
///
/// # Records
///
/// Flags are runtime configuration, not persisted model state. Module records omit their value,
/// trainability and [`ParamId`]; loading a record preserves the destination module's flag exactly
/// as configured. This mirrors tensor parameters, whose values and ids are recorded but whose
/// `require_grad` configuration comes from the destination module.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Flag(bool);

impl Flag {
    /// Create a flag with the given value.
    pub const fn new(value: bool) -> Self {
        Self(value)
    }

    /// Whether the flag is enabled.
    pub const fn is_enabled(&self) -> bool {
        self.0
    }

    /// Return the flag with the given value.
    pub const fn with(self, value: bool) -> Self {
        Self(value)
    }
}

impl Display for Flag {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self.0 {
            true => f.write_str("enabled"),
            false => f.write_str("disabled"),
        }
    }
}

impl super::sealed::Sealed for Flag {}

// `Parameter` is currently tensor-oriented. These implementations let `Flag` use the common
// `Param<T>` identity and traversal machinery until tensor-only capabilities move to a dedicated
// trait. The `require_grad` sidecar temporarily stores whether the flag should be re-enabled after
// an inference round trip; the device and shape implementations remain neutral placeholders.
impl Parameter for Flag {
    fn device(&self) -> Device {
        Device::default()
    }

    fn is_require_grad(&self) -> bool {
        self.is_enabled()
    }

    fn set_require_grad(self, require_grad: bool) -> Self {
        self.with(require_grad)
    }

    fn shape(&self) -> Shape {
        Shape::new([])
    }

    fn load_to_device(self, _device: &Device) -> Self {
        self
    }
}

impl Param<Flag> {
    /// Create an identified flag with a fresh parameter id.
    pub fn from_bool(value: bool) -> Self {
        Self::initialized(ParamId::new(), Flag::new(value))
    }

    /// Return this parameter with its flag set to the given value.
    pub fn with_value(self, value: bool) -> Self {
        self.map(|flag| flag.with(value))
    }
}

impl Module for Param<Flag> {
    fn no_grad(self) -> Self {
        self.with_value(false)
    }

    fn map<M: ModuleMapper>(self, mapper: &mut M) -> Self {
        mapper.map_flag(self)
    }

    fn visit<V: ModuleVisitor>(&self, visitor: &mut V) {
        visitor.visit_flag(self);
    }

    fn to_device(self, _device: &Device) -> Self {
        self
    }

    fn fork(self, _device: &Device) -> Self {
        self
    }

    fn collect_devices(&self, devices: Devices) -> Devices {
        devices
    }
}

impl AutodiffModule for Param<Flag> {
    fn valid(&self) -> Self {
        let enabled = self.require_grad;
        let mut flag = Self::initialized(self.id, Flag::new(false));
        flag.require_grad = enabled;
        flag
    }

    fn from_inner(module: Self) -> Self {
        let enabled = module.require_grad;
        Self::initialized(module.id, Flag::new(enabled))
    }
}

impl ModuleDisplayDefault for Param<Flag> {
    fn content(&self, content: Content) -> Option<Content> {
        content.add_formatted(&format!("{}", self.val())).optional()
    }
}

impl ModuleDisplay for Param<Flag> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inference_temporarily_disables_an_enabled_flag() {
        let flag = Param::<Flag>::from_bool(true);
        let valid = flag.valid();

        assert!(!valid.is_enabled());
        assert!(Param::<Flag>::from_inner(valid).is_enabled());
    }

    #[test]
    fn a_disabled_flag_stays_disabled_across_an_inference_round_trip() {
        let flag = Param::<Flag>::from_bool(true).with_value(false);
        let valid = flag.valid();

        assert!(!valid.is_enabled());
        assert!(!Param::<Flag>::from_inner(valid).is_enabled());
    }

    #[test]
    fn flag_transitions_preserve_identity() {
        let flag = Param::<Flag>::from_bool(true);
        let id = flag.id;

        assert_eq!(flag.clone().with_value(false).id, id);
        assert_eq!(flag.valid().id, id);
        assert_eq!(Param::<Flag>::from_inner(flag).id, id);
    }
}
