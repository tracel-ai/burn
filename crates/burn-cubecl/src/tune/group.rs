use serde::Deserialize;

/// How far up an autotune's running order a group sits.
///
/// A **cutoff**, not a tie-break. The tuner measures the highest-priority group
/// whose candidates can serve the problem and never reaches a lower one; groups
/// sharing a priority are measured together, in one round, as a real race. A
/// negative priority takes the group out of the autotune entirely.
///
/// The number only means anything relative to the others in the same autotune.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Deserialize)]
#[serde(transparent)]
pub struct GroupPriority(pub i8);

impl GroupPriority {
    /// What a family's groups ship at: equal, so their candidates are raced and a kernel that
    /// loses is not chosen.
    pub const DEFAULT: Self = Self(2);

    /// A group taken out of its autotune: cubecl reads a negative priority as
    /// "not in the autotune", so nothing in it is compiled or measured.
    pub const OFF: Self = Self(-1);

    /// One above [`DEFAULT`](Self::DEFAULT): a group at it is measured alone,
    /// and the groups at the default are reached only where every candidate of
    /// it declines.
    pub const ABOVE_DEFAULT: Self = Self(3);

    /// The priority the guaranteed fallback sits at.
    ///
    /// Zero rather than something lower because cubecl reads a *negative*
    /// group priority as "not in the autotune": the fallback has to stay in it.
    /// Everything a configuration should name sits above this, so the fallback
    /// is reached only where every group declined the problem — and a group
    /// demoted to `0` ties with it rather than disappearing under it.
    pub const FALLBACK: Self = Self(0);

    /// The negative a group is named with when cubecl is to skip it entirely.
    ///
    /// One candidate sits here: the one whose *name* is its autotune's group
    /// layout ([`Autotune::finish`](super::Autotune::finish)). It is a string
    /// counted in the tuning cache's checksum, never a kernel to measure, so it
    /// must never reach a plan.
    pub const UNPLANNED: Self = Self(-1);

    /// The within-group priority cubecl reads as "not in this round": where an
    /// autotune puts a candidate that would compile a launch an earlier
    /// candidate of its list already compiles ([`priority_of`](super::priority_of)).
    pub const OUT_OF_THE_ROUND: i8 = -1;

    /// The priority as cubecl's tuner reads it.
    pub fn get(self) -> i8 {
        self.0
    }
}

impl Default for GroupPriority {
    fn default() -> Self {
        Self::DEFAULT
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The fallback is under every group a configuration ships and still in the autotune; the
    /// candidate naming the layout is out of every plan.
    #[test]
    fn the_fallback_sits_under_the_groups_and_inside_the_autotune() {
        assert!(GroupPriority::FALLBACK < GroupPriority::DEFAULT);
        assert!(GroupPriority::FALLBACK.get() >= 0);
        assert!(GroupPriority::UNPLANNED.get() < 0);
    }
}
