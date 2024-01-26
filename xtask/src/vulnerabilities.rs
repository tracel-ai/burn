use std::collections::HashMap;
use std::time::Instant;

use crate::logging::init_logger;
use crate::utils::cargo::{ensure_cargo_crate_is_installed, run_cargo};
use crate::utils::rustup::{
    is_current_toolchain_nightly, rustup_add_component, rustup_get_installed_targets,
};
use crate::utils::time::format_duration;
use crate::utils::Params;
use crate::{endgroup, group};
use std::fmt;

#[derive(clap::ValueEnum, Default, Copy, Clone, PartialEq, Eq)]
pub(crate) enum VulnerabilityCheck {
    /// Run all most useful vulnerability checks.
    #[default]
    All,
    /// Run Address sanitizer (memory error detector)
    AddressSanitizer,
    /// Run LLVM Control Flow Integrity (CFI) (provides forward-edge control flow protection)
    ControlFlowIntegrity,
    /// Run newer variant of Address sanitizer (memory error detector similar to AddressSanitizer, but based on partial hardware assistance)
    HWAddressSanitizer,
    /// Run Kernel LLVM Control Flow Integrity (KCFI) (provides forward-edge control flow protection for operating systems kernels)
    KernelControlFlowIntegrity,
    /// Run Leak sanitizer (run-time memory leak detector)
    LeakSanitizer,
    /// Run memory sanitizer (detector of uninitialized reads)
    MemorySanitizer,
    /// Run another address sanitizer (like AddressSanitizer and HardwareAddressSanitizer but with lower overhead suitable for use as hardening for production binaries)
    MemTagSanitizer,
    /// Run nightly-only checks through cargo-careful `<https://crates.io/crates/cargo-careful>`
    NightlyChecks,
    /// Run SafeStack check (provides backward-edge control flow protection by separating
    /// stack into safe and unsafe regions)
    SafeStack,
    /// Run ShadowCall check (provides backward-edge control flow protection - aarch64 only)
    ShadowCallStack,
    /// Run Thread sanitizer (data race detector)
    ThreadSanitizer,
}

impl VulnerabilityCheck {
    pub(crate) fn run(&self) -> anyhow::Result<()> {
        // Setup logger
        init_logger().init();
        // Start time measurement
        let start = Instant::now();
        match self {
            Self::NightlyChecks => cargo_careful(),
            Self::AddressSanitizer => Sanitizer::Address.run_tests(),
            Self::ControlFlowIntegrity => Sanitizer::CFI.run_tests(),
            Self::HWAddressSanitizer => Sanitizer::HWAddress.run_tests(),
            Self::KernelControlFlowIntegrity => Sanitizer::KCFI.run_tests(),
            Self::LeakSanitizer => Sanitizer::Leak.run_tests(),
            Self::MemorySanitizer => Sanitizer::Memory.run_tests(),
            Self::MemTagSanitizer => Sanitizer::MemTag.run_tests(),
            Self::SafeStack => Sanitizer::SafeStack.run_tests(),
            Self::ShadowCallStack => Sanitizer::ShadowCallStack.run_tests(),
            Self::ThreadSanitizer => Sanitizer::Thread.run_tests(),
            Self::All => {
                cargo_careful();
                Sanitizer::Address.run_tests();
                Sanitizer::Leak.run_tests();
                Sanitizer::Memory.run_tests();
                Sanitizer::SafeStack.run_tests();
                Sanitizer::Thread.run_tests();
            }
        }

        // Stop time measurement
        //
        // Compute runtime duration
        let duration = start.elapsed();

        // Print duration
        info!(
            "\x1B[32;1mTime elapsed for the current execution: {}\x1B[0m",
            format_duration(&duration)
        );

        Ok(())
    }
}

/// Run cargo-careful
fn cargo_careful() {
    if is_current_toolchain_nightly() {
        ensure_cargo_crate_is_installed("cargo-careful");
        rustup_add_component("rust-src");
        // prepare careful sysroot
        group!("Cargo: careful setup");
        run_cargo(
            "careful",
            Params::from(["setup"]),
            HashMap::new(),
            "Cargo sysroot should be available",
        );
        endgroup!();
        // Run cargo careful
        group!("Cargo: run careful checks");
        run_cargo(
            "careful",
            Params::from(["test"]),
            HashMap::new(),
            "Cargo careful should be installed and it should correctly run",
        );
        endgroup!();
    } else {
        error!(
            "You must use 'cargo +nightly' to run nightly checks.
Install a nightly toolchain with 'rustup toolchain install nightly'."
        )
    }
}

// Represents the various sanitizer available in nightly compiler
// source: https://doc.rust-lang.org/beta/unstable-book/compiler-flags/sanitizer.html
#[allow(clippy::upper_case_acronyms)]
enum Sanitizer {
    Address,
    CFI,
    HWAddress,
    KCFI,
    Leak,
    Memory,
    MemTag,
    SafeStack,
    ShadowCallStack,
    Thread,
}

impl fmt::Display for Sanitizer {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Sanitizer::Address => write!(f, "AddressSanitizer"),
            Sanitizer::CFI => write!(f, "ControlFlowIntegrity"),
            Sanitizer::HWAddress => write!(f, "HWAddressSanitizer"),
            Sanitizer::KCFI => write!(f, "KernelControlFlowIntegrity"),
            Sanitizer::Leak => write!(f, "LeakSanitizer"),
            Sanitizer::Memory => write!(f, "MemorySanitizer"),
            Sanitizer::MemTag => write!(f, "MemTagSanitizer"),
            Sanitizer::SafeStack => write!(f, "SafeStack"),
            Sanitizer::ShadowCallStack => write!(f, "ShadowCallStack"),
            Sanitizer::Thread => write!(f, "ThreadSanitizer"),
        }
    }
}

impl Sanitizer {
    const DEFAULT_RUSTFLAGS: &'static str = "-Copt-level=3";

    fn run_tests(&self) {
        if is_current_toolchain_nightly() {
            group!("Sanitizer: {}", self.to_string());
            let retriever = RustupTargetRetriever;
            if self.is_target_supported(&retriever) {
                let envs = vec![
                    (
                        "RUSTFLAGS",
                        format!("{} {}", self.flags(), Sanitizer::DEFAULT_RUSTFLAGS),
                    ),
                    ("RUSTDOCFLAGS", self.flags().to_string()),
                ];

                let features = self.cargo_features();
                let mut args = vec!["--", "--color=always", "--no-capture"];
                args.extend(features);

                run_cargo(
                    "test",
                    args.into(),
                    envs.into_iter().collect(),
                    "Failed to run cargo test",
                );
            } else {
                info!("No supported target found for this sanitizer.");
            }
            endgroup!();
        } else {
            error!(
                "You must use 'cargo +nightly' to run this check.
 Install a nightly toolchain with 'rustup toolchain install nightly'."
            )
        }
    }

    fn flags(&self) -> &'static str {
        match self {
            Sanitizer::Address => "-Zsanitizer=address",
            Sanitizer::CFI => "-Zsanitizer=cfi -Clto",
            Sanitizer::HWAddress => "-Zsanitizer=hwaddress -Ctarget-feature=+tagged-globals",
            Sanitizer::KCFI => "-Zsanitizer=kcfi",
            Sanitizer::Leak => "-Zsanitizer=leak",
            Sanitizer::Memory => "-Zsanitizer=memory -Zsanitizer-memory-track-origins",
            Sanitizer::MemTag => "--Zsanitizer=memtag -Ctarget-feature=\"+mte\"",
            Sanitizer::SafeStack => "-Zsanitizer=safestack",
            Sanitizer::ShadowCallStack => "-Zsanitizer=shadow-call-stack",
            Sanitizer::Thread => "-Zsanitizer=thread",
        }
    }

    fn cargo_features(&self) -> Vec<&str> {
        match self {
            Sanitizer::CFI => vec!["-Zbuild-std", "--target x86_64-unknown-linux-gnu"],
            _ => vec![],
        }
    }

    fn supported_targets(&self) -> Vec<Target> {
        match self {
            Sanitizer::Address => vec![
                Target::Aarch64AppleDarwin,
                Target::Aarch64UnknownFuchsia,
                Target::Aarch64UnknownLinuxGnu,
                Target::X8664AppleDarwin,
                Target::X8664UnknownFuchsia,
                Target::X8664UnknownFreebsd,
                Target::X8664UnknownLinuxGnu,
            ],
            Sanitizer::CFI => vec![Target::X8664UnknownLinuxGnu],
            Sanitizer::HWAddress => {
                vec![Target::Aarch64LinuxAndroid, Target::Aarch64UnknownLinuxGnu]
            }
            Sanitizer::KCFI => vec![
                Target::Aarch64LinuxAndroid,
                Target::Aarch64UnknownLinuxGnu,
                Target::X8664LinuxAndroid,
                Target::X8664UnknownLinuxGnu,
            ],
            Sanitizer::Leak => vec![
                Target::Aarch64AppleDarwin,
                Target::Aarch64UnknownLinuxGnu,
                Target::X8664AppleDarwin,
                Target::X8664UnknownLinuxGnu,
            ],
            Sanitizer::Memory => vec![
                Target::Aarch64UnknownLinuxGnu,
                Target::X8664UnknownFreebsd,
                Target::X8664UnknownLinuxGnu,
            ],
            Sanitizer::MemTag => vec![Target::Aarch64LinuxAndroid, Target::Aarch64UnknownLinuxGnu],
            Sanitizer::SafeStack => vec![Target::X8664UnknownLinuxGnu],
            Sanitizer::ShadowCallStack => vec![Target::Aarch64LinuxAndroid],
            Sanitizer::Thread => vec![
                Target::Aarch64AppleDarwin,
                Target::Aarch64UnknownLinuxGnu,
                Target::X8664AppleDarwin,
                Target::X8664UnknownFreebsd,
                Target::X8664UnknownLinuxGnu,
            ],
        }
    }

    // Returns true if the sanitizer is supported by the currently installed targets
    fn is_target_supported<T: TargetRetriever>(&self, retriever: &T) -> bool {
        let installed_targets = retriever.get_installed_targets();
        let supported = self.supported_targets();
        installed_targets.iter().any(|installed| {
            let installed_target = Target::from_str(installed.trim()).unwrap_or(Target::Unknown);
            supported.iter().any(|target| target == &installed_target)
        })
    }
}

// Constants for target names
const AARCH64_APPLE_DARWIN: &str = "aarch64-apple-darwin";
const AARCH64_LINUX_ANDROID: &str = "aarch64-linux-android";
const AARCH64_UNKNOWN_FUCHSIA: &str = "aarch64-unknown-fuchsia";
const AARCH64_UNKNOWN_LINUX_GNU: &str = "aarch64-unknown-linux-gnu";
const X8664_APPLE_DARWIN: &str = "x86_64-apple-darwin";
const X8664_LINUX_ANDROID: &str = "x86_64-linux-android";
const X8664_UNKNOWN_FUCHSIA: &str = "x86_64-unknown-fuchsia";
const X8664_UNKNOWN_FREEBSD: &str = "x86_64-unknown-freebsd";
const X8664_UNKNOWN_LINUX_GNU: &str = "x86_64-unknown-linux-gnu";

trait TargetRetriever {
    fn get_installed_targets(&self) -> Vec<String>;
}

struct RustupTargetRetriever;

impl TargetRetriever for RustupTargetRetriever {
    fn get_installed_targets(&self) -> Vec<String> {
        rustup_get_installed_targets()
            .lines()
            .map(|s| s.to_string())
            .collect()
    }
}

// Represents Rust targets
// Remark: we list only the targets that are supported by sanitizers
#[derive(Debug, PartialEq)]
enum Target {
    Aarch64AppleDarwin,
    Aarch64LinuxAndroid,
    Aarch64UnknownFuchsia,
    Aarch64UnknownLinuxGnu,
    X8664AppleDarwin,
    X8664LinuxAndroid,
    X8664UnknownFuchsia,
    X8664UnknownFreebsd,
    X8664UnknownLinuxGnu,
    Unknown,
}

impl Target {
    fn from_str(s: &str) -> Option<Self> {
        match s {
            AARCH64_APPLE_DARWIN => Some(Self::Aarch64AppleDarwin),
            AARCH64_LINUX_ANDROID => Some(Self::Aarch64LinuxAndroid),
            AARCH64_UNKNOWN_FUCHSIA => Some(Self::Aarch64UnknownFuchsia),
            AARCH64_UNKNOWN_LINUX_GNU => Some(Self::Aarch64UnknownLinuxGnu),
            X8664_APPLE_DARWIN => Some(Self::X8664AppleDarwin),
            X8664_LINUX_ANDROID => Some(Self::X8664LinuxAndroid),
            X8664_UNKNOWN_FUCHSIA => Some(Self::X8664UnknownFuchsia),
            X8664_UNKNOWN_FREEBSD => Some(Self::X8664UnknownFreebsd),
            X8664_UNKNOWN_LINUX_GNU => Some(Self::X8664UnknownLinuxGnu),
            _ => None,
        }
    }
}

impl fmt::Display for Target {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let target_str = match self {
            Target::Aarch64AppleDarwin => AARCH64_APPLE_DARWIN,
            Target::Aarch64LinuxAndroid => AARCH64_LINUX_ANDROID,
            Target::Aarch64UnknownFuchsia => AARCH64_UNKNOWN_FUCHSIA,
            Target::Aarch64UnknownLinuxGnu => AARCH64_UNKNOWN_LINUX_GNU,
            Target::X8664AppleDarwin => X8664_APPLE_DARWIN,
            Target::X8664LinuxAndroid => X8664_LINUX_ANDROID,
            Target::X8664UnknownFuchsia => X8664_UNKNOWN_FUCHSIA,
            Target::X8664UnknownFreebsd => X8664_UNKNOWN_FREEBSD,
            Target::X8664UnknownLinuxGnu => X8664_UNKNOWN_LINUX_GNU,
            Target::Unknown => "",
        };
        write!(f, "{}", target_str)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    struct MockTargetRetriever {
        mock_data: Vec<String>,
    }

    impl MockTargetRetriever {
        fn new(mock_data: Vec<String>) -> Self {
            Self { mock_data }
        }
    }

    impl TargetRetriever for MockTargetRetriever {
        fn get_installed_targets(&self) -> Vec<String> {
            self.mock_data.clone()
        }
    }

    #[rstest]
    #[case(vec!["".to_string()], false)] // empty string
    #[case(vec!["x86_64-pc-windows-msvc".to_string()], false)] // not supported target
    #[case(vec!["x86_64-pc-windows-msvc".to_string(), "".to_string()], false)] // not supported target and empty string
    #[case(vec!["x86_64-unknown-linux-gnu".to_string()], true)] // one supported target
    #[case(vec!["aarch64-apple-darwin".to_string(), "x86_64-unknown-linux-gnu".to_string()], true)] // one unsupported target and one supported
    fn test_is_target_supported(#[case] installed_targets: Vec<String>, #[case] expected: bool) {
        let mock_retriever = MockTargetRetriever::new(installed_targets);
        let sanitizer = Sanitizer::Memory;
        assert_eq!(sanitizer.is_target_supported(&mock_retriever), expected);
    }

    #[test]
    fn test_consistency_of_fmt_and_from_str_strings() {
        let variants = vec![
            Target::Aarch64AppleDarwin,
            Target::Aarch64LinuxAndroid,
            Target::Aarch64UnknownFuchsia,
            Target::Aarch64UnknownLinuxGnu,
            Target::X8664AppleDarwin,
            Target::X8664LinuxAndroid,
            Target::X8664UnknownFuchsia,
            Target::X8664UnknownFreebsd,
            Target::X8664UnknownLinuxGnu,
        ];
        for variant in variants {
            let variant_str = format!("{}", variant);
            let parsed_variant = Target::from_str(&variant_str);
            assert_eq!(Some(variant), parsed_variant);
        }
    }
}
