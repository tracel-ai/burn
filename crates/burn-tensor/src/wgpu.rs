//! Initialize wgpu through [`Device::wgpu_options`].
//!
//! Native applications can call `WgpuOptions::init`. Async initialization
//! works on native and browser targets:
//!
//! ```no_run
//! # async fn example() -> Result<(), burn_tensor::wgpu::WgpuInitError> {
//! use burn_tensor::{Device, Tensor};
//!
//! let device = Device::wgpu_options().tasks_max(32).init_async().await?;
//! let input = Tensor::<2>::ones([2, 3], &device);
//! # Ok(())
//! # }
//! ```
//!
//! Supply a [`WgpuSetup`](crate::wgpu::WgpuSetup) with
//! [`.setup(...)`](crate::wgpu::WgpuOptions::setup) to use
//! an existing application's device and queue. Register it once, then pass or
//! clone the returned Burn device into models. Repeated registration of the same
//! handles is not detected. Registering a setup does not
//! change which runtime [`Device::default`] selects.
//!
//! [`wgpu`](crate::wgpu::wgpu) exposes the exact wgpu version whose handles this runtime accepts.
//! Sharing handles does not convert tensors into buffers or wait for GPU work:
//! flush pending Burn work before submitting external consumers on the shared
//! queue, and keep shared allocations alive until those consumers finish.

use burn_dispatch::devices::wgpu as runtime;
pub use runtime::{MemoryConfiguration, WgpuBackend, WgpuInitError, WgpuSetup, wgpu};

use crate::{Device, DeviceKind, device::wgpu_device};

impl Device {
    /// Options for initializing Burn's wgpu runtime or registering an existing setup.
    ///
    /// Starts with the default device selector, automatic graphics API, and runtime
    /// defaults. Call `init` or `init_async` before creating tensors on the same selector.
    /// An initialized runtime cannot be reconfigured. Clone its device to reuse it.
    pub fn wgpu_options() -> WgpuOptions {
        WgpuOptions::default()
    }
}

#[derive(Default)]
struct Options {
    memory: Option<MemoryConfiguration>,
    tasks_max: Option<usize>,
}

impl Options {
    fn resolve(self) -> Result<runtime::RuntimeOptions, WgpuInitError> {
        let mut options = runtime::RuntimeOptions::with_tasks_max(self.tasks_max)?;
        if let Some(memory) = self.memory {
            options.memory_config = memory;
        }
        Ok(options)
    }
}

/// Options for initializing Burn's wgpu runtime.
///
/// Construct with [`Device::wgpu_options`] or [`Default::default`]. Options cover
/// device acquisition and runtime settings such as memory management and task batching.
/// Setters do not acquire a GPU. Initialization registers the selected runtime for reuse.
/// Dropping the returned device does not deregister that runtime.
#[must_use = "wgpu options do nothing until initialized"]
pub struct WgpuOptions {
    kind: DeviceKind,
    api: WgpuBackend,
    selection_explicit: bool,
    options: Options,
}

impl Default for WgpuOptions {
    fn default() -> Self {
        Self {
            kind: DeviceKind::DefaultDevice,
            api: WgpuBackend::Auto,
            selection_explicit: false,
            options: Options::default(),
        }
    }
}

impl WgpuOptions {
    /// Select an adapter kind. `Existing` identifiers cannot be initialized here;
    /// register an existing setup with [`Self::setup`] instead.
    ///
    /// Browser selection follows the runtime's power preferences; browsers do not
    /// expose native adapter indices. The default prefers a high-performance GPU.
    pub fn device_kind(mut self, kind: DeviceKind) -> Self {
        self.kind = kind;
        self.selection_explicit = true;
        self
    }

    /// Pin the graphics API, or let [`WgpuBackend::Auto`] choose it.
    ///
    /// An explicit API never falls back to another API. This selects the graphics
    /// API, not a shader compiler; the runtime chooses a supported compiler.
    /// With the `metal` feature enabled, an explicit Metal selection requires native
    /// MSL support. Initialization returns an error if it is unavailable; automatic
    /// selection permits WGSL fallback.
    pub fn graphics_api(mut self, api: WgpuBackend) -> Self {
        self.api = api;
        self.selection_explicit = true;
        self
    }

    /// Set the runtime's memory allocation policy.
    pub fn memory_config(mut self, config: MemoryConfiguration) -> Self {
        self.options.memory = Some(config);
        self
    }

    /// Set the maximum tasks aggregated into a GPU command. Must be nonzero.
    /// Overrides `CUBECL_WGPU_MAX_TASKS`, including a malformed environment value.
    pub fn tasks_max(mut self, tasks: usize) -> Self {
        self.options.tasks_max = Some(tasks);
        self
    }

    /// Register existing wgpu handles instead of acquiring another device.
    ///
    /// Memory and task options carry over. Explicit calls to [`Self::device_kind`]
    /// or [`Self::graphics_api`] conflict with a supplied setup, even when those
    /// options name defaults; [`WgpuSetupOptions::init`] returns an error.
    ///
    /// Handles must belong to one coherent setup. Compiler selection follows the
    /// same backend and adapter checks as acquired setups. The supplied device
    /// must enable the selected compiler's required features, including native
    /// extensions for SPIR-V or MSL; registration does not validate all of them.
    pub fn setup(self, setup: WgpuSetup) -> WgpuSetupOptions {
        WgpuSetupOptions {
            setup,
            options: self.options,
            selection_explicit: self.selection_explicit,
        }
    }

    fn into_request(self) -> Result<(runtime::WgpuDevice, runtime::RuntimeOptions), WgpuInitError> {
        if matches!(self.kind, DeviceKind::Existing(_)) {
            return Err(WgpuInitError::InvalidConfiguration {
                message: "Existing is a runtime identity, not an adapter selector; supply a setup or reuse its device".into(),
            });
        }
        Ok((wgpu_device(self.kind, self.api), self.options.resolve()?))
    }

    /// Initialize the device synchronously on native targets.
    ///
    /// Use [`Self::init_async`] when running in a browser or an async application.
    ///
    /// # Errors
    ///
    /// Returns acquisition, configuration, or registration errors.
    ///
    /// Returns an error if the runtime for the selected device kind and graphics
    /// API has already been initialized, either explicitly or by a tensor operation.
    /// Runtime options must be set before first use.
    #[cfg(not(target_family = "wasm"))]
    pub fn init(self) -> Result<Device, WgpuInitError> {
        self.init_with_setup().map(|(device, _)| device)
    }

    /// Initialize the device asynchronously on native or browser targets.
    ///
    /// # Errors
    ///
    /// Returns acquisition, configuration, or registration errors.
    ///
    /// Returns an error if the runtime for the selected device kind and graphics
    /// API has already been initialized, either explicitly or by a tensor operation.
    /// Runtime options must be set before first use.
    pub async fn init_async(self) -> Result<Device, WgpuInitError> {
        self.init_with_setup_async().await.map(|(device, _)| device)
    }

    /// Initialize synchronously and return the exact handles used by this runtime.
    ///
    /// Share these handles with another wgpu component. Do not register the
    /// returned setup again; clone the accompanying Burn device to reuse it.
    ///
    /// # Errors
    ///
    /// Returns acquisition, configuration, or registration errors.
    ///
    /// Returns an error if the runtime for the selected device kind and graphics
    /// API has already been initialized, either explicitly or by a tensor operation.
    /// Runtime options must be set before first use.
    #[cfg(not(target_family = "wasm"))]
    pub fn init_with_setup(self) -> Result<(Device, WgpuSetup), WgpuInitError> {
        let (device, options) = self.into_request()?;
        let setup = runtime::try_init_setup::<runtime::AutoGraphicsApi>(&device, options)?;
        Ok((Device::from(device), setup))
    }

    /// Initialize asynchronously and return the exact handles used by this runtime.
    ///
    /// Available on native and browser targets. The returned device is ready for
    /// tensor operations; the setup is ready to share with another wgpu component.
    ///
    /// # Errors
    ///
    /// Returns acquisition, configuration, or registration errors.
    ///
    /// Returns an error if the runtime for the selected device kind and graphics
    /// API has already been initialized, either explicitly or by a tensor operation.
    /// Runtime options must be set before first use.
    pub async fn init_with_setup_async(self) -> Result<(Device, WgpuSetup), WgpuInitError> {
        let (device, options) = self.into_request()?;
        let setup =
            runtime::try_init_setup_async::<runtime::AutoGraphicsApi>(&device, options).await?;
        Ok((Device::from(device), setup))
    }
}

/// Runtime options for registering an existing wgpu setup.
///
/// Created by [`WgpuOptions::setup`]. Registration is synchronous even in
/// a browser, since the application has already acquired the device and queue.
#[must_use = "wgpu setup options do nothing until initialized"]
pub struct WgpuSetupOptions {
    setup: WgpuSetup,
    options: Options,
    selection_explicit: bool,
}

impl WgpuSetupOptions {
    /// Set the runtime's memory allocation policy.
    pub fn memory_config(mut self, config: MemoryConfiguration) -> Self {
        self.options.memory = Some(config);
        self
    }

    /// Set a nonzero task count, overriding `CUBECL_WGPU_MAX_TASKS`.
    pub fn tasks_max(mut self, tasks: usize) -> Self {
        self.options.tasks_max = Some(tasks);
        self
    }

    /// Register the setup and return its initialized Burn device.
    ///
    /// Each call creates a new runtime identity; repeated handles are not detected.
    /// Failed initialization does not change an existing registration.
    ///
    /// # Errors
    ///
    /// Returns an error for conflicting selection options, invalid runtime options,
    /// incompatible capabilities, or a failure to register the runtime.
    pub fn init(self) -> Result<Device, WgpuInitError> {
        if self.selection_explicit {
            return Err(WgpuInitError::InvalidConfiguration {
                message: "device_kind and graphics_api cannot be combined with an existing setup"
                    .into(),
            });
        }
        runtime::try_init_device(self.setup, self.options.resolve()?).map(Device::from)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn existing_identity_is_not_an_adapter_to_create() {
        assert!(matches!(
            Device::wgpu_options()
                .device_kind(DeviceKind::Existing(42))
                .into_request(),
            Err(WgpuInitError::InvalidConfiguration { .. })
        ));
    }

    #[test]
    fn zero_tasks_are_rejected_before_acquiring_a_gpu() {
        assert!(matches!(
            WgpuOptions::default().tasks_max(0).into_request(),
            Err(WgpuInitError::InvalidConfiguration { .. })
        ));
    }

    #[cfg(all(feature = "std", not(target_family = "wasm")))]
    #[test]
    fn explicit_tasks_override_a_malformed_environment_default() {
        // Isolate environment changes from other tests and runtime initialization.
        const CHILD: &str = "BURN_TEST_WGPU_TASKS_CHILD";
        if std::env::var_os(CHILD).is_some() {
            assert!(matches!(
                Device::wgpu_options().into_request(),
                Err(WgpuInitError::InvalidConfiguration { .. })
            ));
            assert_eq!(
                Device::wgpu_options()
                    .tasks_max(7)
                    .into_request()
                    .unwrap()
                    .1
                    .tasks_max,
                7
            );
            return;
        }
        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "wgpu::tests::explicit_tasks_override_a_malformed_environment_default",
            ])
            .env(CHILD, "1")
            .env("CUBECL_WGPU_MAX_TASKS", "invalid")
            .status()
            .unwrap();
        assert!(status.success());
    }
}
