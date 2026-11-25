//! What every family's autotune is built from.
//!
//! A family decides *which* candidates it holds; the shape they are held in is
//! the same everywhere, and lives here so that no two families drift into
//! two conventions. [`Autotune`] is that shape: a group per place a kernel comes
//! from, at the priority the caller's configuration
//! gives it, and a floor underneath them all.
//!
//! Group priority is a **cutoff**, not a tie-break: cubecl measures the
//! highest-priority group whose candidates launch and stops there. Shipped, the
//! groups of an autotune are equal, so they are measured together.
//!
//! Inside the groups a round is an **order**, and it is decided per problem. A
//! family says of each candidate how well its cuts suit the key
//! ([`TuneCandidate::priority`], the priority band); the round tries the candidates best first and
//! stops at the first one that runs close enough to the device's roofline
//! ([`bounded`](Autotune::bounded)), so the candidates after it are never
//! compiled. A candidate that turns out to be far behind stops being sampled
//! after its second run (cubecl's speed factor). A candidate's
//! priority is therefore a claim about which candidate to *try first*, never
//! about which can win: a wrong priority costs a compile, and a round with no
//! bound compiles the whole group.
//!
//! One claim is a proof rather than a judgement, and it takes a candidate out
//! of the round: a setting that compiles a launch an earlier candidate already
//! compiles *is* that candidate ([`TuneCandidate::blueprint`], read by
//! [`priority_of`]).
//!
//! # What the type holds that the plumbing could not
//!
//! * **[`Autotune::group`] registers a group and names it in the same call**, and
//!   there is no other way to get one. A group's priority reaches a *warm*
//!   tuning cache only through the layout tunable, whose name carries the
//!   priorities — so a group an autotune used but forgot to name would be one the
//!   file could demote with no visible effect.
//! * **[`Autotune::new`] fixes what every family benchmarks on**: clones of the
//!   real call inputs, unless the autotune measures on inputs of its own
//!   ([`Autotune::measured_on`]), as a cache write does.
//! * **An autotune of one kernel still exists.** The group is what a second kernel
//!   joins, and a family with no group cannot be taken out of a run by a file.
//!   It costs one sample on first touch and a cache hit after.
//! * **Editing a priority invalidates the winners measured under the old one**,
//!   and cubecl would not notice on its own: it checksums an autotune by its
//!   candidate *names* and stores a winner as an *index*, so a priority change
//!   moves neither. Each autotune therefore carries a candidate whose name *is*
//!   its group layout, in a group cubecl never plans — a string in the
//!   checksum, and no measurement.
//! * **A `LocalTuner`'s candidate set is built once per device** — it is keyed
//!   by the device's `CubeTuneId` — so anything the device reports may decide
//!   what is in a candidate list. The problem may not: the list exists before
//!   any key arrives, and what moves with the shape is the candidates'
//!   priority.

use super::roofline::{Bounded, ReadCold};
use super::{CandidatePriority, GroupPriority, Roofline, TuneCandidate, priority_of};
use core::fmt::Display;
use cubecl::tune::{
    AutotuneKey, CloneInputGenerator, InputGenerator, KeyGenerator, OrderedSettings, Patience,
    Tunable, TunableSet, TuneGroup, TuneInputs,
};
use std::sync::Arc;

/// A family's autotune: its configured groups, its candidates, and the
/// floor under them.
///
/// The type exists for one invariant. A group's priority reaches a *warm*
/// tuning cache only through the layout tunable [`finish`](Self::finish)
/// appends, whose name carries the priorities — so a group registered as a
/// candidate's home but missing from that name is a group a configuration can
/// demote with no effect anyone can see. Registering a group and naming it are
/// therefore the same act: [`group`](Self::group) does both, and nothing else
/// can do either.
///
/// `Id` is what its candidates are identified by ([`Identity`]): `()` for an autotune whose winner is
/// only ever run again.
pub struct Autotune<K: AutotuneKey, I: TuneInputs, O: 'static, Id = ()> {
    set: TunableSet<K, I, O, Id>,
    /// The groups a file can name, and what the candidates are measured on.
    layout: GroupLayout,
    /// The group under all of them, which no file reaches.
    floor: TuneGroup<K>,
}

/// One candidate's launch, as a tunable runs it.
type Launch<I, O> = dyn for<'a> Fn(<I as TuneInputs>::At<'a>) -> Result<O, String> + Send + Sync;

/// What an autotune's candidates are identified by: every tunable of a cubecl set carries one, so a
/// candidate the caller only names — a floor, a foreign routine, the group layout — is given the
/// identity its name makes.
pub trait Identity: Sized {
    /// A tunable running `func`, named `name` and identified by nothing more.
    fn named<K, I, O, Func, Err>(name: &str, func: Func) -> Tunable<K, I, O, Self>
    where
        I: TuneInputs,
        O: 'static,
        Err: Into<String> + 'static,
        Func: for<'a> Fn(<I as TuneInputs>::At<'a>) -> Result<O, Err> + Send + Sync + 'static;
}

/// An autotune whose winner is only ever run again: its candidates are named and nothing more.
impl Identity for () {
    fn named<K, I, O, Func, Err>(name: &str, func: Func) -> Tunable<K, I, O, Self>
    where
        I: TuneInputs,
        O: 'static,
        Err: Into<String> + 'static,
        Func: for<'a> Fn(<I as TuneInputs>::At<'a>) -> Result<O, Err> + Send + Sync + 'static,
    {
        Tunable::new(name, func)
    }
}

impl<K: AutotuneKey, I: TuneInputs, O: 'static, Id: Identity> Autotune<K, I, O, Id> {
    /// An empty autotune keyed by `key_gen`.
    ///
    /// Candidates are measured against clones of the real call inputs, which is
    /// the whole crate's answer: every family's benchmark inputs *are* the
    /// problem the caller brought, so a winner is chosen on the shape that will
    /// run. A candidate that writes through its operands cannot be measured
    /// this way, since every sample would land in the caller's state: the
    /// cache write measures on inputs of its own ([`measured_on`](Self::measured_on)).
    pub fn new(key_gen: impl KeyGenerator<K, I>) -> Self {
        Self::with_inputs(key_gen, CloneInputGenerator, None)
    }

    /// [`new`](Self::new), measured on what `input_gen` makes of the call
    /// instead of a clone of it: for a family whose launch reads a **value**
    /// on device that its key does not name, and so cannot be trusted to hold
    /// what the key says when the autotune measures, or that writes through its
    /// operands.
    ///
    /// Decode attention is one: the attended length lives on device, a
    /// build that discards launches never advances it, and a round measured on whatever it held
    /// elects the kernel that is fastest over an empty cache, whatever it
    /// costs over a full one. A cache write would land every sample in
    /// the caller's cache.
    ///
    /// `inputs` names what the candidates run on, and joins the layout
    /// tunable's name ([`finish`](Self::finish)): a winner measured some
    /// other way is not an answer to this autotune, and only a name reaches a warm
    /// cache.
    pub fn measured_on(
        key_gen: impl KeyGenerator<K, I>,
        inputs: &'static str,
        input_gen: impl InputGenerator<K, I>,
    ) -> Self {
        Self::with_inputs(key_gen, input_gen, Some(inputs))
    }

    fn with_inputs(
        key_gen: impl KeyGenerator<K, I>,
        input_gen: impl InputGenerator<K, I>,
        measured: Option<&'static str>,
    ) -> Self {
        Self {
            set: TunableSet::new(key_gen, input_gen),
            layout: GroupLayout {
                groups: Vec::new(),
                measured,
            },
            floor: GroupLayout::priority_group("floor", GroupPriority::FALLBACK),
        }
    }

    /// Register the group `name` at the priority the configuration gave it, and
    /// take the handle its candidates join.
    pub fn group(&mut self, name: &'static str, priority: GroupPriority) -> TuneGroup<K> {
        self.layout.groups.push((name, priority));
        GroupLayout::priority_group(name, priority)
    }

    /// Add a candidate to `group`, tried with the highest-priority candidates
    /// of its round.
    ///
    /// A candidate nobody can argue about goes first, not last: what a
    /// priority buys is the compiles a bound saves, and a round that started with the
    /// candidates it knows least about would have to compile them before
    /// finding out.
    pub fn candidate(self, group: &TuneGroup<K>, tunable: Tunable<K, I, O, Id>) -> Self {
        self.at(group, tunable, |_| Some(CandidatePriority::FIRST))
    }

    /// Add a candidate to `group`, tried after every prioritized candidate of
    /// its round.
    ///
    /// For a kernel this crate did not write and cannot prioritize — a routine of
    /// cubek's — measured only where the family's own candidates fall short of
    /// the roofline: a bounded round that finds its answer among them never
    /// compiles this one, and a round that does not reaches it still.
    pub fn candidate_after(self, group: &TuneGroup<K>, tunable: Tunable<K, I, O, Id>) -> Self {
        self.at(group, tunable, |_| Some(CandidatePriority::LAST))
    }

    /// Add a family's whole candidate list to `group`, each entry a candidate,
    /// named by its setting, tried in the order [`priority_of`] gives **for the
    /// problem**, and run by `run`.
    ///
    /// The list arrives whole because a candidate's place is not a candidate's
    /// to state — one candidate can say how well it suits a problem, and only
    /// the list can say that an earlier candidate already compiles its launch.
    /// `run` is where a family puts what only it knows: how a candidate of this
    /// list launches.
    ///
    /// A candidate is named by `Display` over its setting, so the label a warm
    /// cache is keyed by and the settings the candidate runs cannot drift
    /// apart. The order costs that cache nothing: cubecl checksums an autotune by
    /// its candidates' names and stores a winner as an index into the list, and
    /// a priority moves neither.
    pub fn prioritized<C, Run>(self, group: &TuneGroup<K>, candidates: Vec<C>, run: Run) -> Self
    where
        C: TuneCandidate<Key = K> + Send + Sync + 'static,
        Run: for<'a> Fn(&C, <I as TuneInputs>::At<'a>) -> Result<O, String> + Send + Sync + 'static,
    {
        self.listed(group, candidates, run, |candidate, launch| {
            Id::named(&candidate.strategy().to_string(), launch)
        })
    }

    /// [`prioritized`](Self::prioritized), each candidate identified by what `identify` names it,
    /// and named by that: the value cubecl's tuner hands back once the candidate wins a key
    /// ([`LocalTuner::fastest_identity`](cubecl::tune::LocalTuner::fastest_identity)), for a caller that builds
    /// something else from the winner.
    pub fn identified<C, Identify, Run>(
        self,
        group: &TuneGroup<K>,
        candidates: Vec<C>,
        identify: Identify,
        run: Run,
    ) -> Self
    where
        C: TuneCandidate<Key = K> + Send + Sync + 'static,
        Id: Display,
        Identify: Fn(&C) -> Id,
        Run: for<'a> Fn(&C, <I as TuneInputs>::At<'a>) -> Result<O, String> + Send + Sync + 'static,
    {
        self.listed(group, candidates, run, |candidate, launch| {
            Tunable::identified(identify(candidate), launch)
        })
    }

    /// Every candidate of the list in `group`, run by `run`, placed per key by [`priority_of`],
    /// and made a tunable by `tunable` around its launch.
    fn listed<C, Run>(
        self,
        group: &TuneGroup<K>,
        candidates: Vec<C>,
        run: Run,
        tunable: impl Fn(&C, Box<Launch<I, O>>) -> Tunable<K, I, O, Id>,
    ) -> Self
    where
        C: TuneCandidate<Key = K> + Send + Sync + 'static,
        Run: for<'a> Fn(&C, <I as TuneInputs>::At<'a>) -> Result<O, String> + Send + Sync + 'static,
    {
        let candidates: Arc<[C]> = candidates.into();
        let run = Arc::new(run);
        let mut autotune = self;
        for position in 0..candidates.len() {
            let (prioritized_against, running, run) =
                (candidates.clone(), candidates.clone(), run.clone());
            let launch: Box<Launch<I, O>> = Box::new(move |inputs| run(&running[position], inputs));
            autotune = autotune.at(group, tunable(&candidates[position], launch), move |key| {
                priority_of(&prioritized_against, position, key)
            });
        }
        autotune
    }

    /// Add a candidate to `group` at the priority `priority` gives it per key,
    /// or out of the round where it gives none — the one place a within-group
    /// priority is set, so [`candidate`](Self::candidate) and
    /// [`prioritized`](Self::prioritized) cannot disagree about what one means.
    fn at(
        self,
        group: &TuneGroup<K>,
        tunable: Tunable<K, I, O, Id>,
        priority: impl Fn(&K) -> Option<CandidatePriority> + Send + Sync + 'static,
    ) -> Self {
        Self {
            set: self.set.with(tunable.group(group, move |key| {
                priority(key).map_or(GroupPriority::OUT_OF_THE_ROUND, CandidatePriority::get)
            })),
            ..self
        }
    }

    /// Say what a problem costs, so a round can stop at the first candidate
    /// that runs close enough to the device's peak.
    ///
    /// How close is the configured autotune level's
    /// (`Thresholds::for_level`): the roofline time `roofline` states,
    /// divided by the level's fraction of peak, is the time under which a
    /// candidate ends the round. At `full` every candidate is measured. The
    /// peaks are cubecl's own probes, cached per device.
    ///
    /// Without this, a group's order decides nothing: every candidate of it is
    /// compiled and the fastest wins, which is a priority's cost with none of
    /// its return.
    ///
    /// The same statement makes the round **read cold**. A problem whose bytes
    /// fit the device's last-level cache would otherwise be measured from it —
    /// a tuner samples one candidate several times over the same operands,
    /// warm-up first — and a memory-bound candidate then reads as several times
    /// the bus, meets this very bound on the first candidate tried, and ends
    /// the round with a winner chosen on a read production never gets. So
    /// before every measured sample of such a problem the autotune writes past the
    /// cache ([`cold_read_bytes`](crate::tune::cold_read_bytes)), and the candidates are timed the way a decode
    /// step meets them: after every other weight of the model.
    pub fn bounded<F>(self, roofline: F) -> Self
    where
        F: for<'a> Fn(&K, &<I as TuneInputs>::At<'a>) -> Roofline + Send + Sync + 'static,
    {
        let roofline = Arc::new(roofline);
        Self {
            set: self
                .set
                .with_bounds(Arc::new(Bounded(roofline.clone())))
                .with_eviction(Arc::new(ReadCold(roofline))),
            ..self
        }
    }

    /// The cold read of [`bounded`](Self::bounded) without the early stop.
    ///
    /// A round that ends at the first candidate under the limit is only sound
    /// while every candidate's time is real. A candidate timed at **zero**
    /// clears any limit, so the round stops at it and persists that candidate
    /// with the rest unmeasured; the attention autotunes have met such a
    /// candidate. The limit is not what fails, the reading is.
    ///
    /// So the stop waits until a zero cannot be reported as a time. The cold
    /// read stays, because that reason is untouched: a problem inside the
    /// last-level cache is otherwise measured from it, and production reads it
    /// after every other weight of the model.
    pub fn read_cold<F>(self, roofline: F) -> Self
    where
        F: for<'a> Fn(&K, &<I as TuneInputs>::At<'a>) -> Roofline + Send + Sync + 'static,
    {
        Self {
            set: self
                .set
                .with_eviction(Arc::new(ReadCold(Arc::new(roofline)))),
            ..self
        }
    }

    /// Add the candidate that answers a key every group declined — and that a
    /// file taking every group out of the autotune still reaches, because a plan
    /// cubecl cannot fill is a panic rather than an empty result.
    pub fn floor(self, tunable: Tunable<K, I, O, Id>) -> Self {
        let tunable = tunable.group(&self.floor, GroupLayout::within_group);
        Self {
            set: self.set.with(tunable),
            ..self
        }
    }

    /// The set, closed with the candidate that names the group layout.
    ///
    /// cubecl invalidates a persisted tuning cache by checksumming its
    /// candidates' **names**, and a cached winner is an *index* into the
    /// candidate list. A configuration that reorders the groups moves
    /// neither, so without this candidate the file's whole purpose fails
    /// silently: a developer demotes a group, restarts, and every key tuned
    /// before the edit goes on launching the winner measured under the old
    /// order. Writing the layout into a name is what makes the edit reach the
    /// cache.
    ///
    /// The candidate sits in a group cubecl never plans (the `GroupPriority`
    /// below every planned one), so it costs a string and no measurement.
    pub fn finish(self) -> TunableSet<K, I, O, Id> {
        let unplanned = GroupLayout::priority_group("group-layout", GroupPriority::UNPLANNED);
        self.set.with(
            Id::named(&self.layout.to_string(), |_| {
                Err::<O, String>("the group layout is a name, not a kernel".to_string())
            })
            .group(&unplanned, GroupLayout::within_group),
        )
    }
}

/// An autotune's groups as a file can name them, and what its candidates are measured on: what the
/// candidate naming the layout is named by ([`Autotune::finish`]), as one name.
struct GroupLayout {
    /// Every group a file can name, in the order the family registered them.
    groups: Vec<(&'static str, GroupPriority)>,
    /// What the candidates are measured on, where it is not a clone of the call
    /// — named in the layout tunable for the reason
    /// [`measured_on`](Autotune::measured_on) gives.
    measured: Option<&'static str>,
}

impl GroupLayout {
    /// When a round stops measuring a group: after [`min_measured`](Patience::min_measured)
    /// candidates in priority order, once [`max_misses`](Patience::max_misses) in a row fail to
    /// beat the leader. The minimum also holds the roofline's early stop back, so a family of
    /// that many candidates or fewer is always raced whole.
    const PATIENCE: Patience = Patience {
        min_measured: 10,
        max_misses: 5,
    };

    /// A group at a fixed priority, whatever the key. Ordered: its candidates'
    /// priorities say what to try first, and [`PATIENCE`](Self::PATIENCE) how far down that
    /// order the round goes.
    fn priority_group<K: AutotuneKey>(name: &str, priority: GroupPriority) -> TuneGroup<K> {
        let priority = priority.get();
        TuneGroup::ordered(
            name,
            move |_key| priority,
            OrderedSettings::default().with_patience(Self::PATIENCE),
        )
    }

    /// The priority a candidate of a one-candidate group sits at — the floor's,
    /// the layout tunable's. Its group is the whole of its placement, so any
    /// priority does.
    fn within_group<K>(_key: &K) -> i8 {
        CandidatePriority::LAST.get()
    }
}

impl Display for GroupLayout {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("groups")?;
        for (group, priority) in &self.groups {
            write!(f, "-{group}{}", priority.get())?;
        }
        if let Some(inputs) = self.measured {
            write!(f, "-measured-{inputs}")?;
        }
        // How far a round goes decides its winner as the priorities do, so a winner chosen
        // under another patience is not an answer to this autotune.
        let patience = Self::PATIENCE;
        write!(
            f,
            "-patience{}x{}",
            patience.min_measured, patience.max_misses
        )
    }
}

/// What an autotune's candidate is identified by where the autotune is asked which kernel won a key: a
/// routine candidate by its routine and setting, which the tuner hands back once it is the fastest
/// ([`LocalTuner::fastest_identity`](cubecl::tune::LocalTuner::fastest_identity)), and every other
/// candidate — a foreign kernel, the floor named apart, the group layout — by its name alone.
///
/// `S` is the family's routine settings: one value naming the routine and its setting.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CandidateIdentity<S> {
    /// A routine of the family, at its setting.
    Routine(S),
    /// Any other candidate, by its name alone.
    Named(String),
}

impl<S> CandidateIdentity<S> {
    /// A routine candidate's identity: its routine and setting.
    pub fn new<C: TuneCandidate>(candidate: &C) -> Self
    where
        C::Strategy: Copy + Into<S>,
    {
        Self::Routine((*candidate.strategy()).into())
    }

    /// The routine this identifies, or `None` for another kernel.
    pub fn routine(self) -> Option<S> {
        match self {
            Self::Routine(settings) => Some(settings),
            Self::Named(_) => None,
        }
    }
}

/// The candidate's name: its setting's, for a routine candidate, the same name a candidate listed
/// without an identity has, which is what a persisted winner is checksummed by.
impl<S: Display> Display for CandidateIdentity<S> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Routine(settings) => settings.fmt(f),
            Self::Named(name) => f.write_str(name),
        }
    }
}

impl<S: Display> Identity for CandidateIdentity<S> {
    fn named<K, I, O, Func, Err>(name: &str, func: Func) -> Tunable<K, I, O, Self>
    where
        I: TuneInputs,
        O: 'static,
        Err: Into<String> + 'static,
        Func: for<'a> Fn(<I as TuneInputs>::At<'a>) -> Result<O, Err> + Send + Sync + 'static,
    {
        Tunable::identified(Self::Named(name.to_string()), func)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A group layout is a candidate *name*, so editing a configuration
    /// reaches the persisted tuning cache.
    ///
    /// cubecl invalidates that cache by checksumming candidate names and stores
    /// a winner as an index into the candidate list; a priority change moves
    /// neither. Two layouts that named the same candidate would leave a demoted
    /// group still launching for every key tuned before the edit — the file's
    /// whole purpose, failing silently. Every family's autotune depends on this
    /// one name, which is why it is tested here rather than once a family.
    #[test]
    fn a_group_layout_names_itself() {
        let shipped = GroupPriority::DEFAULT;
        let raced = GroupLayout {
            groups: vec![("tile", shipped), ("cubek", shipped)],
            measured: None,
        }
        .to_string();
        assert_ne!(
            raced,
            GroupLayout {
                groups: vec![("tile", shipped), ("cubek", GroupPriority::UNPLANNED)],
                measured: None
            }
            .to_string()
        );
        // A one-group autotune and the first group of a two-group one are
        // different layouts: a family that fell back to the other's name would
        // read the other's records.
        assert_ne!(
            raced,
            GroupLayout {
                groups: vec![("tile", shipped)],
                measured: None
            }
            .to_string()
        );
        assert_ne!(
            GroupLayout {
                groups: vec![("tile", shipped)],
                measured: None
            }
            .to_string(),
            GroupLayout {
                groups: vec![("cubek", shipped)],
                measured: None
            }
            .to_string()
        );
        // How a autotune measures is part of what it answered: a winner measured on
        // clones of a live call is not one measured at capacity.
        assert_ne!(
            raced,
            GroupLayout {
                groups: vec![("tile", shipped), ("cubek", shipped)],
                measured: Some("at-capacity")
            }
            .to_string()
        );
    }
}
