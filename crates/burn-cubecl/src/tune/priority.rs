//! The priority band: where a candidate goes in an autotune round for one problem
//! ([`CandidatePriority`]), what a family says of each of its candidates to place it
//! ([`TuneCandidate`]), and the one rule that takes a candidate out of a round ([`priority_of`]).

use core::fmt::Display;
use core::time::Duration;

/// Where a candidate goes in a round **for one problem**: the candidates a
/// problem suits best are tried first, and a bound can end the round before the
/// rest are compiled.
///
/// An order and nothing more. A candidate is never lowered out of the round by
/// its priority — a round that finds no candidate close enough to peak reaches
/// every one of them — so a priority can be wrong about a candidate without the
/// round being wrong about the winner. The band is wide so a family can lower a
/// candidate several times over, and so [`LAST`](Self::LAST) stays a statement
/// of its own.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct CandidatePriority(i8);

impl CandidatePriority {
    /// Tried first: where a candidate sits when nothing about the problem
    /// lowers it.
    pub const FIRST: Self = Self(64);

    /// Tried after every other candidate of the group: what a family says of a
    /// candidate whose launch it can argue against without a clock.
    pub const LAST: Self = Self(0);

    /// Tried after every candidate whose time was predicted, and before the ones at
    /// [`LAST`](Self::LAST): what a family says of a candidate that prepares but whose time the
    /// model cannot predict on this device — a peak the device did not measure, an estimate that
    /// leaves its grid unstated. Untimed candidates keep their list's order among themselves.
    pub const UNPREDICTED: Self = Self(1);

    /// The lowest a predicted time places a candidate: one level above
    /// [`UNPREDICTED`](Self::UNPREDICTED), so a prediction, however slow, is tried before no
    /// prediction.
    const SLOWEST_PREDICTED: Self = Self(Self::UNPREDICTED.0 + 1);

    /// This priority, `levels` lower — saturating at [`LAST`](Self::LAST).
    pub fn lowered(self, levels: u32) -> Self {
        Self(
            i8::try_from(i32::from(self.0) - levels.min(i8::MAX as u32) as i32)
                .unwrap_or(Self::LAST.0)
                .max(Self::LAST.0),
        )
    }

    /// How much slower than predicted one level lower is: a candidate predicted within this of
    /// another sits at its level, where no prediction tells them apart and the list's order does.
    pub const LEVEL_RATIO: f64 = 1.1;

    /// The priority of a candidate predicted to take `predicted` on a problem no kernel runs
    /// under `floor`: [`FIRST`](Self::FIRST), lowered a level for each
    /// [`LEVEL_RATIO`](Self::LEVEL_RATIO) its prediction sits above the floor, saturating one
    /// level above [`UNPREDICTED`](Self::UNPREDICTED). A prediction under the floor, or a floor of
    /// nothing, is first.
    pub fn above_floor(predicted: Duration, floor: Duration) -> Self {
        if floor.is_zero() {
            return Self::FIRST;
        }
        let ratio = predicted.as_secs_f64() / floor.as_secs_f64();
        let levels = ratio.ln() / Self::LEVEL_RATIO.ln();
        // A `NaN` or negative level casts to zero, an infinite one saturates.
        Self::FIRST
            .lowered(levels.round() as u32)
            .max(Self::SLOWEST_PREDICTED)
    }

    /// The priority as cubecl's tuner reads it: a group or candidate priority, never negative,
    /// since cubecl reads a negative one as out of the table.
    pub fn get(self) -> i8 {
        self.0
    }
}

/// One candidate from a family's list: the setting it launches, and what
/// [`priority_of`] needs to place it in a round.
///
/// A family implements this to say, of a single candidate, *what this candidate
/// would do with that problem* — [`priority`](Self::priority), how well its cuts suit
/// the extents, and [`blueprint`](Self::blueprint), which launch it would
/// compile. Both are statements about the candidate **alone**, which is what
/// makes them testable; reading them against the other candidates is
/// [`priority_of`]'s job, and lives there so no family has to get it right twice.
///
/// Every method is answered from a [`Key`](Self::Key) and nothing else, because
/// a key is the whole of what cubecl hands a priority function, and two
/// problems sharing a key share a record.
pub trait TuneCandidate {
    /// The setting a candidate launches. `Display` because a table names its
    /// candidates by it, and that name is what a warm cache is keyed by.
    type Strategy: Display;
    /// How a problem is told apart from another — this family's autotune key.
    type Key;
    /// What a launch is identified by, for [`blueprint`](Self::blueprint). `()`
    /// where a family cannot derive one before the launch.
    type Blueprint: PartialEq;

    /// The setting this candidate launches, which names it.
    fn strategy(&self) -> &Self::Strategy;

    /// How well this candidate's cuts suit the problem `key` describes, judged
    /// on its own: what its cube covers against what the problem asks for.
    ///
    /// A **judgement**, and it may only rest on arithmetic the key affords. It
    /// decides how soon the candidate is tried, never whether it is: a
    /// candidate this lowers is still measured by a round that finds nothing
    /// better.
    fn priority(&self, key: &Self::Key) -> CandidatePriority;

    /// The launch this candidate would compile for `key`, or `None` where the
    /// family cannot derive one without the tensors.
    ///
    /// **Not a judgement but a proof, and it exists for one job**: two
    /// candidates whose settings clamp onto a single launch are one kernel
    /// under two names, and a round that measures both spends part of itself
    /// confirming a tie before persisting whichever label was tried first.
    /// [`priority_of`] leaves the second out of the round.
    ///
    /// Answering `None` costs a family only that tie; a wrong answer is a
    /// kernel nothing measures, so a family that cannot be sure says `None`.
    fn blueprint(&self, _key: &Self::Key) -> Option<Self::Blueprint> {
        None
    }
}

/// A candidate of a family that has not judged its candidates: the setting, and
/// nothing said about any problem.
///
/// Every candidate sits at [`CandidatePriority::FIRST`] and none claims a
/// launch, so a round over these is the list in the order the family wrote it.
/// It is the honest shape for a family whose candidates have not been judged
/// yet, rather than a priority invented to fill the slot: a priority that is
/// wrong costs a compile, and a duplicate claim that is wrong costs a kernel.
///
/// The key is `()` because there is nothing to answer from. A family that
/// starts prioritizing its candidates writes its own candidate type against its
/// own key, the way the matmul arms do, and this goes away with the last one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Unprioritized<S>(S);

impl<S> Unprioritized<S> {
    /// This setting as an unprioritized candidate.
    pub fn of(strategy: S) -> Self {
        Self(strategy)
    }
}

impl<S: Display> TuneCandidate for Unprioritized<S> {
    type Strategy = S;
    type Key = ();
    type Blueprint = ();

    fn strategy(&self) -> &S {
        &self.0
    }

    /// One priority for every candidate: the family says nothing, so the round
    /// is the list.
    fn priority(&self, _key: &()) -> CandidatePriority {
        CandidatePriority::FIRST
    }
}

/// Where `candidates[position]` goes in a round for `key`, or `None` where it
/// would compile a launch an earlier candidate of its list already compiles.
///
/// Earlier and not later, so the candidate a tie resolves to is the one the
/// list leads with — the canonical setting, which a pin runs.
///
/// A free function over the list, because the one thing it adds to a
/// candidate's own [`priority`](TuneCandidate::priority) is a statement about
/// the list.
pub fn priority_of<C: TuneCandidate>(
    candidates: &[C],
    position: usize,
    key: &C::Key,
) -> Option<CandidatePriority> {
    let candidate = &candidates[position];
    let compiled = candidate.blueprint(key);
    let repeats_an_earlier_candidate = compiled.is_some()
        && candidates[..position]
            .iter()
            .any(|earlier| earlier.blueprint(key) == compiled);
    (!repeats_an_earlier_candidate).then(|| candidate.priority(key))
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Candidate {
        lowered: u32,
        compiles: u8,
    }

    impl TuneCandidate for Candidate {
        type Strategy = u32;
        type Key = ();
        type Blueprint = u8;
        fn strategy(&self) -> &u32 {
            &self.lowered
        }
        fn priority(&self, _key: &()) -> CandidatePriority {
            CandidatePriority::FIRST.lowered(self.lowered)
        }
        fn blueprint(&self, _key: &()) -> Option<u8> {
            Some(self.compiles)
        }
    }

    fn candidate(lowered: u32, compiles: u8) -> Candidate {
        Candidate { lowered, compiles }
    }

    /// A priority orders and never removes: every candidate that compiles its own
    /// launch is in the round, however low the family placed it.
    #[test]
    fn every_distinct_candidate_is_in_the_round() {
        let candidates = [candidate(0, 1), candidate(7, 2), candidate(u32::MAX, 3)];
        for position in 0..candidates.len() {
            assert!(priority_of(&candidates, position, &()).is_some());
        }
        assert_eq!(
            priority_of(&candidates, 2, &()),
            Some(CandidatePriority::LAST)
        );
    }

    /// A candidate that compiles an earlier candidate's launch is out of the
    /// round, and the earlier one keeps its place: a tie resolves to the
    /// setting the list leads with.
    #[test]
    fn a_repeat_of_an_earlier_candidate_is_out_of_the_round() {
        let candidates = [candidate(3, 1), candidate(0, 1), candidate(1, 2)];
        assert_eq!(
            priority_of(&candidates, 0, &()),
            Some(CandidatePriority::FIRST.lowered(3))
        );
        assert_eq!(priority_of(&candidates, 1, &()), None);
        assert_eq!(
            priority_of(&candidates, 2, &()),
            Some(CandidatePriority::FIRST.lowered(1))
        );
    }

    /// Lowering saturates at `LAST` and never goes negative, because cubecl
    /// reads a negative priority as "not in the table" — and a candidate
    /// priority that could reach it would be a judgement removing a candidate.
    #[test]
    fn a_lowered_priority_stays_in_the_round() {
        assert_eq!(
            CandidatePriority::FIRST.lowered(u32::MAX),
            CandidatePriority::LAST
        );
        assert!(CandidatePriority::LAST.get() >= 0);
    }

    /// A prediction a level's ratio over the floor is a level lower; two levels' ratio, two.
    #[test]
    fn each_level_ratio_above_the_floor_is_a_level_lower() {
        let floor = Duration::from_micros(100);
        let at = |ratio: f64| CandidatePriority::above_floor(floor.mul_f64(ratio), floor);
        assert_eq!(at(1.0), CandidatePriority::FIRST);
        assert_eq!(at(1.1), CandidatePriority::FIRST.lowered(1));
        assert_eq!(at(1.1 * 1.1), CandidatePriority::FIRST.lowered(2));
    }

    /// A prediction however far above the floor is tried before a candidate whose time was not
    /// predicted, which is tried before one its routine declines.
    #[test]
    fn a_slow_prediction_is_tried_before_no_prediction() {
        let floor = Duration::from_micros(100);
        let slowest = CandidatePriority::above_floor(floor.mul_f64(1e6), floor);
        assert!(slowest > CandidatePriority::UNPREDICTED, "{slowest:?}");
        assert!(CandidatePriority::UNPREDICTED > CandidatePriority::LAST);
    }

    /// A prediction the model puts under the floor, or a floor of nothing, says nothing against
    /// the candidate: it is first.
    #[test]
    fn a_prediction_under_the_floor_is_first() {
        let floor = Duration::from_micros(100);
        assert_eq!(
            CandidatePriority::above_floor(floor / 2, floor),
            CandidatePriority::FIRST
        );
        assert_eq!(
            CandidatePriority::above_floor(floor, Duration::ZERO),
            CandidatePriority::FIRST
        );
    }
}
