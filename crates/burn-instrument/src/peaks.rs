//! What a device can do at best: the ceilings a component's [`Bounds`](crate::Bounds) are
//! divided by.

use burn_tensor::FloatDType;
use serde::{Deserialize, Serialize};

/// A device's practical peaks, measured by saturating probe kernels.
///
/// Every ratio a report prints shares these denominators, so one bad reading invalidates all of
/// them at once. A probe can fail without erroring and return a small positive number, so a
/// scorer divides by [`trusted`](Self::trusted), never by the raw readings, and a report says
/// what [`findings`](Self::findings) found.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Peaks {
    /// Copy bandwidth, bytes/s: every byte read and written back. The ceiling for traffic split
    /// evenly between the two directions, and the fallback denominator where a device reports no
    /// per-direction ceilings.
    pub copy_bytes_per_s: Option<f64>,
    /// Read bandwidth, bytes/s.
    ///
    /// A component's reads are charged at this and its writes at
    /// [`write_bytes_per_s`](Self::write_bytes_per_s), because the two differ: a bus pays to
    /// switch direction, and a store to a line the cache does not hold reads it first. A
    /// read-only stream legitimately runs above the copy ceiling.
    pub read_bytes_per_s: Option<f64>,
    /// Write bandwidth, bytes/s, in the bytes a kernel stores. See
    /// [`read_bytes_per_s`](Self::read_bytes_per_s).
    pub write_bytes_per_s: Option<f64>,
    /// Arithmetic peak, FLOP/s at 2 per multiply-add, at [`compute_dtype`](Self::compute_dtype):
    /// the tensor-core kernel for 16-bit floats, plain arithmetic otherwise.
    pub flops_per_s: Option<f64>,
    /// The dtype the arithmetic peak was measured at, which should be the one the pass computes
    /// in.
    pub compute_dtype: FloatDType,
}

impl Default for Peaks {
    /// Nothing measured.
    fn default() -> Self {
        Self {
            copy_bytes_per_s: None,
            read_bytes_per_s: None,
            write_bytes_per_s: None,
            flops_per_s: None,
            compute_dtype: FloatDType::F32,
        }
    }
}

/// What a reader must be told about a set of peaks before its numbers mean anything. See
/// [`Peaks::findings`].
///
/// A rejection dropped a reading and the components were scored on a coarser wall; a caveat
/// kept one, because nothing in the readings says which of them is wrong.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PeakFinding {
    /// The write ceiling came in above the read one, which no shared bus does, so both were
    /// dropped and bytes fall back to the copy ceiling. See [`Peaks::rejected_directions`].
    ImpossibleDirections {
        /// The read ceiling the probe reported, bytes/s.
        read_bytes_per_s: f64,
        /// The write ceiling it reported above that, bytes/s.
        write_bytes_per_s: f64,
    },
    /// The two directions predict a copy ceiling the probe did not measure. Kept and still
    /// divided by. See [`Peaks::inconsistent_ceilings`].
    UnpredictedCopyCeiling {
        /// What the copy probe measured, bytes/s.
        measured_bytes_per_s: f64,
        /// What the two directions predict, bytes/s: their harmonic mean.
        predicted_bytes_per_s: f64,
    },
    /// The arithmetic peak came in below the device's own copy bandwidth, so it was dropped and
    /// components are scored on memory alone. See [`Peaks::rejected_compute`].
    ImpossibleCompute {
        /// What the compute probe reported, FLOP/s.
        flops_per_s: f64,
    },
}

/// How far the copy ceiling may sit from what the two directions predict before
/// [`Peaks::inconsistent_ceilings`] says so.
///
/// Wide on purpose: the prediction is a two-term model of a real bus and each reading carries a
/// few percent of noise, so a tight bound would report the model's own error. On a working probe
/// set the prediction lands within a percent; the inconsistent set this exists for was 61% out.
const CEILING_TOLERANCE: f64 = 0.25;

impl Peaks {
    /// These peaks with an impossible reading dropped, so a scorer falls back to a coarser wall
    /// rather than divide by a broken denominator: an impossible arithmetic peak leaves the
    /// bytes wall, and an impossible pair of directions leaves the copy ceiling.
    ///
    /// A merely implausible reading is kept; see
    /// [`inconsistent_ceilings`](Self::inconsistent_ceilings).
    pub fn trusted(self) -> Self {
        let trusted = match self.rejected_compute() {
            Some(_) => Self {
                flops_per_s: None,
                ..self
            },
            None => self,
        };
        // Both directions go together: they are scored as a pair, so keeping the believable
        // half of an impossible pair would divide one direction by a measurement and the other
        // by nothing.
        match trusted.rejected_directions() {
            Some(_) => Self {
                read_bytes_per_s: None,
                write_bytes_per_s: None,
                ..trusted
            },
            None => trusted,
        }
    }

    /// Everything a reader must be told about these peaks, in the order a report says it: what
    /// was dropped, then what merely disagrees. Empty for a working probe set.
    ///
    /// The consistency check reads the [`trusted`](Self::trusted) peaks, so a pair already
    /// rejected is not also reported as disagreeing with the copy ceiling.
    pub fn findings(self) -> Vec<PeakFinding> {
        let mut findings = Vec::new();
        if let Some((read, write)) = self.rejected_directions() {
            findings.push(PeakFinding::ImpossibleDirections {
                read_bytes_per_s: read,
                write_bytes_per_s: write,
            });
        }
        if let Some((measured, predicted)) = self.trusted().inconsistent_ceilings() {
            findings.push(PeakFinding::UnpredictedCopyCeiling {
                measured_bytes_per_s: measured,
                predicted_bytes_per_s: predicted,
            });
        }
        if let Some(flops) = self.rejected_compute() {
            findings.push(PeakFinding::ImpossibleCompute { flops_per_s: flops });
        }
        findings
    }

    /// The arithmetic peak, when it is too low to believe: below the ridge point of one FLOP per
    /// byte moved, where no arithmetic would ever be worth doing between two loads. No
    /// accelerator is built that way, so such a reading is the probe failing.
    pub fn rejected_compute(self) -> Option<f64> {
        let compute = self.flops_per_s?;
        let memory = self.copy_bytes_per_s?;
        (compute < memory).then_some(compute)
    }

    /// The read and write ceilings, when they cannot both be true: a write ceiling above the
    /// read one. A store moves at least the bytes a load does, so this is the probe counting
    /// bytes it did not pay for, such as lines still resident in cache.
    pub fn rejected_directions(self) -> Option<(f64, f64)> {
        let read = self.read_bytes_per_s?;
        let write = self.write_bytes_per_s?;
        (write > read).then_some((read, write))
    }

    /// The measured copy ceiling and the one the two directions predict, when they are too far
    /// apart to describe one device.
    ///
    /// A copy moves half its bytes each way, so its ceiling is the harmonic mean of the two
    /// directions. This reports and does not reject: when the three disagree, nothing in the
    /// readings says which is wrong. It cannot catch all three reading low in the same
    /// proportion, since the ratio is all it tests.
    pub fn inconsistent_ceilings(self) -> Option<(f64, f64)> {
        let copy = self.copy_bytes_per_s?;
        let read = self.read_bytes_per_s?;
        let write = self.write_bytes_per_s?;
        let predicted = 2.0 / (1.0 / read + 1.0 / write);
        let apart = (predicted - copy).abs() / copy;
        (apart > CEILING_TOLERANCE).then_some((copy, predicted))
    }
}

#[cfg(feature = "cubecl")]
impl Peaks {
    /// Measure `device`'s peaks, the arithmetic one at `dtype`.
    ///
    /// The probes are saturating kernels that move clocks and thermals for whatever runs after
    /// them, so measure before the pass, not between its parts. The runtime keeps each value for
    /// the process, so a second call is free. A device with no probes (a non-cubecl one, or a
    /// runtime without timing) reports `None` for each.
    pub fn measure(device: &burn_tensor::Device, dtype: FloatDType) -> Self {
        use cubecl::throughput::{MemoryAccess, ThroughputKey, ThroughputMode};

        let memory = |access| {
            measured(
                device,
                ThroughputKey {
                    mode: ThroughputMode::memory(access),
                },
            )
        };
        Self {
            copy_bytes_per_s: memory(MemoryAccess::Copy),
            read_bytes_per_s: memory(MemoryAccess::Read),
            write_bytes_per_s: memory(MemoryAccess::Write),
            flops_per_s: compute_keys(dtype)
                .into_iter()
                .find_map(|key| measured(device, key)),
            compute_dtype: dtype,
        }
    }
}

/// The arithmetic probes for `dtype`, in the order to try them.
///
/// A 16-bit float runs on tensor cores accumulating in f32, as a matmul does, at the largest
/// tile the device implements: a runtime declines a tile it does not, so the next is tried.
/// Every other dtype is plain arithmetic.
#[cfg(feature = "cubecl")]
fn compute_keys(dtype: FloatDType) -> Vec<cubecl::throughput::ThroughputKey> {
    use cubecl::ir::{ElemType, FloatKind};
    use cubecl::throughput::{CmmaDims, ComputeCmmaConfig, ThroughputKey, ThroughputMode};

    let elem = ElemType::Float(match dtype {
        FloatDType::F64 => FloatKind::F64,
        FloatDType::F32 | FloatDType::Flex32 => FloatKind::F32,
        FloatDType::F16 => FloatKind::F16,
        FloatDType::BF16 => FloatKind::BF16,
    });
    match dtype {
        FloatDType::F16 | FloatDType::BF16 => [16, 8]
            .into_iter()
            .map(|edge| ThroughputKey {
                mode: ThroughputMode::ComputeCmma {
                    dtype: elem,
                    config: ComputeCmmaConfig {
                        accumulator_type: ElemType::Float(FloatKind::F32),
                        cmma_dims: CmmaDims {
                            m: edge,
                            n: edge,
                            k: edge,
                        },
                    },
                },
            })
            .collect(),
        FloatDType::F32 | FloatDType::F64 | FloatDType::Flex32 => vec![ThroughputKey {
            mode: ThroughputMode::ComputeDirect { dtype: elem },
        }],
    }
}

/// One peak, per second, or `None` where the device has none for `key` or reported a reading
/// that is zero or not finite.
#[cfg(feature = "cubecl")]
fn measured(device: &burn_tensor::Device, key: cubecl::throughput::ThroughputKey) -> Option<f64> {
    let value = device
        .as_dispatch()
        .performance_stats(&[key])
        .into_iter()
        .next()?
        .ok()?;
    let per_s = match key.mode.memory_probe() {
        Some(_) => value.bytes_per_s(&key),
        None => value.ops_per_s(),
    };
    (per_s.is_finite() && per_s > 0.0).then_some(per_s)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A compute peak below the copy bandwidth is a failed probe, and dividing by it would put
    /// a floor longer than the pass: 0.0045 TFLOP/s beside 166 GB/s was such a reading.
    #[test]
    fn a_compute_peak_below_the_memory_peak_is_rejected() {
        let broken = Peaks {
            copy_bytes_per_s: Some(166.1e9),
            flops_per_s: Some(4.5e9),
            ..Peaks::default()
        };
        assert_eq!(broken.rejected_compute(), Some(4.5e9));
        assert_eq!(broken.trusted().flops_per_s, None);
        assert_eq!(broken.trusted().copy_bytes_per_s, Some(166.1e9));
    }

    #[test]
    fn a_real_compute_peak_is_left_alone() {
        let measured = Peaks {
            copy_bytes_per_s: Some(198.9e9),
            flops_per_s: Some(49.3e12),
            ..Peaks::default()
        };
        assert_eq!(measured.rejected_compute(), None);
        assert_eq!(measured.trusted().flops_per_s, Some(49.3e12));
    }

    /// A write ceiling above the read one takes both directions with it, and the copy ceiling
    /// survives as the fallback.
    #[test]
    fn an_impossible_write_ceiling_takes_both_directions_with_it() {
        let broken = Peaks {
            copy_bytes_per_s: Some(156.0e9),
            read_bytes_per_s: Some(227.0e9),
            write_bytes_per_s: Some(281.0e9),
            flops_per_s: Some(49.3e12),
            compute_dtype: FloatDType::F16,
        };
        assert_eq!(broken.rejected_directions(), Some((227.0e9, 281.0e9)));
        let trusted = broken.trusted();
        assert_eq!(trusted.read_bytes_per_s, None);
        assert_eq!(trusted.write_bytes_per_s, None);
        assert_eq!(trusted.copy_bytes_per_s, Some(156.0e9));
        assert_eq!(trusted.flops_per_s, Some(49.3e12));
    }

    /// The ordering a working probe reports is left alone, and one direction alone has nothing
    /// to contradict.
    #[test]
    fn a_plausible_pair_of_directions_is_left_alone() {
        let measured = Peaks {
            copy_bytes_per_s: Some(161.0e9),
            read_bytes_per_s: Some(199.0e9),
            write_bytes_per_s: Some(137.0e9),
            ..Peaks::default()
        };
        assert_eq!(measured.rejected_directions(), None);
        assert_eq!(measured.trusted().read_bytes_per_s, Some(199.0e9));

        let half = Peaks {
            write_bytes_per_s: None,
            ..measured
        };
        assert_eq!(half.rejected_directions(), None);
    }

    /// The directions predict the copy ceiling: within a percent on a working set, 61% out on
    /// one that is wrong without being impossible.
    #[test]
    fn a_copy_ceiling_the_directions_do_not_predict_is_reported() {
        let measured = Peaks {
            copy_bytes_per_s: Some(161.0e9),
            read_bytes_per_s: Some(199.0e9),
            write_bytes_per_s: Some(137.0e9),
            ..Peaks::default()
        };
        assert_eq!(measured.inconsistent_ceilings(), None);

        let broken = Peaks {
            copy_bytes_per_s: Some(156.0e9),
            read_bytes_per_s: Some(227.0e9),
            write_bytes_per_s: Some(281.0e9),
            ..Peaks::default()
        };
        let (copy, predicted) = broken.inconsistent_ceilings().expect("61% apart");
        assert_eq!(copy, 156.0e9);
        assert!((predicted / 1e9 - 251.0).abs() < 1.0, "{predicted}");

        let no_directions = Peaks {
            read_bytes_per_s: None,
            write_bytes_per_s: None,
            ..measured
        };
        assert_eq!(no_directions.inconsistent_ceilings(), None);
    }

    /// A rejected pair is not also reported as inconsistent: one finding, in the terms that say
    /// what happened to the denominator.
    #[test]
    fn a_rejected_pair_is_not_also_reported_as_inconsistent() {
        let broken = Peaks {
            copy_bytes_per_s: Some(156.0e9),
            read_bytes_per_s: Some(227.0e9),
            write_bytes_per_s: Some(281.0e9),
            flops_per_s: Some(49.3e12),
            compute_dtype: FloatDType::F16,
        };
        assert!(broken.inconsistent_ceilings().is_some(), "both checks fire");
        assert_eq!(
            broken.findings(),
            vec![PeakFinding::ImpossibleDirections {
                read_bytes_per_s: 227.0e9,
                write_bytes_per_s: 281.0e9,
            }]
        );
    }

    #[test]
    fn findings_name_what_was_dropped_before_what_was_doubted() {
        let odd = Peaks {
            copy_bytes_per_s: Some(140.0e9),
            read_bytes_per_s: Some(234.0e9),
            write_bytes_per_s: Some(212.0e9),
            flops_per_s: Some(4.5e9),
            compute_dtype: FloatDType::F16,
        };
        let findings = odd.findings();
        assert_eq!(findings.len(), 2, "{findings:?}");
        assert!(matches!(
            findings[0],
            PeakFinding::UnpredictedCopyCeiling { .. }
        ));
        assert_eq!(
            findings[1],
            PeakFinding::ImpossibleCompute { flops_per_s: 4.5e9 }
        );
    }

    #[test]
    fn a_working_probe_set_reports_nothing() {
        let measured = Peaks {
            copy_bytes_per_s: Some(161.0e9),
            read_bytes_per_s: Some(199.0e9),
            write_bytes_per_s: Some(137.0e9),
            flops_per_s: Some(49.3e12),
            compute_dtype: FloatDType::F16,
        };
        assert_eq!(measured.findings(), Vec::new());
    }

    /// All three low in the same proportion is what the ratio cannot see; stated so nobody
    /// reaches for this check expecting it.
    #[test]
    fn three_ceilings_low_together_pass_every_check() {
        let low = Peaks {
            copy_bytes_per_s: Some(161.0e9 * 0.7),
            read_bytes_per_s: Some(199.0e9 * 0.7),
            write_bytes_per_s: Some(137.0e9 * 0.7),
            ..Peaks::default()
        };
        assert_eq!(low.rejected_directions(), None);
        assert_eq!(low.inconsistent_ceilings(), None);
    }

    /// Without a copy reading there is nothing to judge an arithmetic one against.
    #[test]
    fn nothing_is_rejected_when_there_is_nothing_to_compare_against() {
        let partial = Peaks {
            flops_per_s: Some(4.5e9),
            ..Peaks::default()
        };
        assert_eq!(partial.rejected_compute(), None);
        assert_eq!(partial.trusted().flops_per_s, Some(4.5e9));
    }
}
