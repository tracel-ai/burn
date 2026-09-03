use alloc::format;
use core::fmt::{Display, Formatter, Result as FmtResult};

use burn_tensor::Device;

use crate::module::{
    AutodiffModule, Content, Devices, Module, ModuleDisplay, ModuleDisplayDefault, ModuleMapper,
    ModuleVisitor, Param, ParamId, ParameterValue,
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
/// lifecycle state and [`ParamId`]; loading a record preserves the destination module's flag
/// exactly as configured. This mirrors tensor parameters, whose values and ids are recorded but
/// whose `require_grad` configuration comes from the destination module.
///
/// # Devices
///
/// Flags are host-side control state and do not reside on a compute device. Moving or forking a
/// module preserves a flag unchanged, and flags do not contribute entries to [`Module::devices`].
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

impl super::sealed::Sealed for Flag {
    fn is_active(&self) -> bool {
        self.is_enabled()
    }
}

impl ParameterValue for Flag {}

impl Param<Flag> {
    /// Create an identified flag with a fresh parameter id.
    pub fn from_bool(value: bool) -> Self {
        Self::initialized(ParamId::new(), Flag::new(value))
    }

    /// Return this parameter with its flag set to the given value.
    pub fn with_value(self, value: bool) -> Self {
        let mut flag = self.map(|flag| flag.with(value));
        flag.is_active = value;
        flag
    }
}

impl Module for Param<Flag> {
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
        let enabled = self.is_active;
        let mut flag = Self::initialized(self.id, Flag::new(false));
        flag.is_active = enabled;
        flag
    }

    fn from_inner(module: Self) -> Self {
        let enabled = module.is_active;
        let mut flag = Self::initialized(module.id, Flag::new(enabled));
        flag.is_active = enabled;
        flag
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
    use crate::test_device;

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
    fn enabling_a_flag_during_validation_is_preserved_by_train() {
        let flag = Param::<Flag>::from_bool(false).valid().with_value(true);

        assert!(flag.is_enabled());
        assert!(flag.is_active);

        let flag = flag.train();

        assert!(flag.is_enabled());
        assert!(flag.is_active);
    }

    #[test]
    fn mapping_a_validation_flag_preserves_what_train_restores() {
        let flag = Param::<Flag>::from_bool(true)
            .valid()
            .map(|flag| flag.with(false));

        assert!(!flag.is_enabled());
        assert!(flag.is_active);

        let flag = flag.train();

        assert!(flag.is_enabled());
        assert!(flag.is_active);
    }

    #[test]
    fn flag_transitions_preserve_identity() {
        let flag = Param::<Flag>::from_bool(true);
        let id = flag.id;

        assert_eq!(flag.clone().with_value(false).id, id);
        assert_eq!(flag.valid().id, id);
        assert_eq!(Param::<Flag>::from_inner(flag).id, id);
    }

    #[test]
    fn flags_are_device_independent() {
        let flag = Param::<Flag>::from_bool(true);
        let id = flag.id;
        let device = test_device();

        let moved = flag.clone().to_device(&device);
        assert!(moved.is_enabled());
        assert_eq!(moved.id, id);
        assert!(moved.devices().is_empty());

        let forked = flag.fork(&device);
        assert!(forked.is_enabled());
        assert_eq!(forked.id, id);
        assert!(forked.devices().is_empty());
    }

    #[test]
    fn no_grad_does_not_disable_a_flag() {
        let flag = Param::<Flag>::from_bool(true).no_grad();

        assert!(flag.is_enabled());
    }

    #[cfg(feature = "autodiff")]
    #[test]
    fn validation_state_is_restored_only_by_train() {
        let device = test_device().autodiff();
        let valid = Param::<Flag>::from_bool(true).valid();

        assert!(!valid.is_enabled());
        assert!(!valid.clone().to_device(&device).is_enabled());
        assert!(!valid.clone().fork(&device).is_enabled());
        assert!(valid.train().is_enabled());
    }
}
