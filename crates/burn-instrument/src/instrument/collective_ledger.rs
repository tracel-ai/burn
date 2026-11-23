//! The collectives a run issued, each with its parts and the waits on them, kept by id until the
//! run is over and settled into what each cost every member.

use std::collections::HashMap;
use std::time::Duration;

use cubecl::logging::{Communication, Execution, ExecutionId, ExecutionKind, WaitTarget};
use cubecl::profile::ProfileDuration;

use super::members::{Member, MemberList, RanOn};
use crate::breakdown::{CollectiveMemberTime, CollectivePath, CollectiveTime};

/// Every collective a run issued, in issue order, and the parts and waits measured under each.
#[derive(Default)]
pub struct CollectiveLedger {
    collectives: Vec<IssuedCollective>,
    /// Where each collective sits in `collectives`.
    by_id: HashMap<ExecutionId, usize>,
    /// The parts and waits issued under a collective on a member, by id, awaiting their
    /// measurement.
    shares: HashMap<ExecutionId, Share>,
    /// The collective of the last part issued on each member: what a wait for every queued
    /// collective waits for last.
    last_part_on: HashMap<Member, usize>,
}

/// A collective as it was issued, and the measurements its parts and waits handed over.
struct IssuedCollective {
    kind: CollectiveKind,
    measured: Vec<(Share, ProfileDuration)>,
}

/// What collectives are counted together under: the same operation, path and size.
#[derive(Clone, PartialEq, Eq)]
struct CollectiveKind {
    operation: String,
    path: CollectivePath,
    bytes: u64,
}

/// A part or a wait of one collective, on one member.
#[derive(Clone, Copy)]
struct Share {
    collective: usize,
    member: Member,
    role: ShareRole,
}

/// Which of a member's two costs a measurement is.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ShareRole {
    /// The member's part, on its collective stream.
    Part,
    /// Its compute stream blocked on the part.
    Wait,
}

impl CollectiveLedger {
    /// Keep `execution` where it is a collective, or a part or wait of one placed on a member.
    pub fn issued(&mut self, execution: &Execution, members: &MemberList) {
        match &execution.kind {
            ExecutionKind::Communication(communication) => {
                self.collective(execution, communication)
            }
            ExecutionKind::Part { place } => self.part(execution, members.ran_on(place)),
            ExecutionKind::Wait { place, awaits } => {
                self.wait(execution, members.ran_on(place), awaits)
            }
            ExecutionKind::Kernel { .. } => {}
        }
    }

    /// Keep the unread measurement of a part or wait this ledger placed.
    pub fn measured(&mut self, execution: &Execution, profile: ProfileDuration) {
        if let Some(share) = self.shares.get(&execution.id).copied() {
            self.collectives[share.collective]
                .measured
                .push((share, profile));
        }
    }

    /// Read every measurement back, waiting for each, and sum the collectives of each kind: how
    /// many ran and what they cost each of `members`, in the order each kind was first issued.
    pub fn settle(self, members: &MemberList) -> Vec<CollectiveTime> {
        let mut settled: Vec<CollectiveTime> = Vec::new();
        for collective in self.collectives {
            let row = match settled.iter().position(|row| collective.kind.matches(row)) {
                Some(row) => row,
                None => {
                    settled.push(collective.kind.row(members.count()));
                    settled.len() - 1
                }
            };
            let row = &mut settled[row];
            row.count += 1;
            for (share, profile) in collective.measured {
                let length = cubecl::future::block_on(profile.resolve())
                    .map(|ticks| ticks.duration())
                    .unwrap_or_default();
                if let Some(member) = row.members.get_mut(share.member.0) {
                    share.role.add(member, length);
                }
            }
        }
        settled
    }

    fn collective(&mut self, execution: &Execution, communication: &Communication) {
        let Communication::Collective {
            operation,
            bytes,
            path,
            ..
        } = communication
        else {
            return;
        };
        self.by_id.insert(execution.id, self.collectives.len());
        self.collectives.push(IssuedCollective {
            kind: CollectiveKind {
                operation: operation.to_string(),
                path: path.into(),
                bytes: *bytes,
            },
            measured: Vec::new(),
        });
    }

    fn part(&mut self, execution: &Execution, ran_on: RanOn) {
        let collective = execution.parent.and_then(|parent| self.by_id.get(&parent));
        let (Some(&collective), RanOn::Member(member)) = (collective, ran_on) else {
            return;
        };
        self.last_part_on.insert(member, collective);
        self.shares.insert(
            execution.id,
            Share {
                collective,
                member,
                role: ShareRole::Part,
            },
        );
    }

    fn wait(&mut self, execution: &Execution, ran_on: RanOn, awaits: &WaitTarget) {
        let RanOn::Member(member) = ran_on else {
            return;
        };
        let collective = match awaits {
            WaitTarget::Part(part) => self.shares.get(part).map(|share| share.collective),
            WaitTarget::QueuedCollectives => self.last_part_on.get(&member).copied(),
        };
        if let Some(collective) = collective {
            self.shares.insert(
                execution.id,
                Share {
                    collective,
                    member,
                    role: ShareRole::Wait,
                },
            );
        }
    }
}

impl CollectiveKind {
    fn matches(&self, row: &CollectiveTime) -> bool {
        self.operation == row.operation && self.path == row.path && self.bytes == row.bytes
    }

    /// A row of this kind with nothing counted yet, over `members` members.
    fn row(&self, members: usize) -> CollectiveTime {
        CollectiveTime {
            operation: self.operation.clone(),
            path: self.path.clone(),
            bytes: self.bytes,
            count: 0,
            members: vec![CollectiveMemberTime::default(); members],
        }
    }
}

impl ShareRole {
    fn add(self, member: &mut CollectiveMemberTime, length: Duration) {
        match self {
            ShareRole::Part => member.part += length,
            ShareRole::Wait => member.exposed += length,
        }
    }
}

impl From<&cubecl::logging::CollectivePath> for CollectivePath {
    fn from(path: &cubecl::logging::CollectivePath) -> Self {
        match path {
            cubecl::logging::CollectivePath::Native(transport) => CollectivePath::Native {
                transport: format!("{transport:?}"),
            },
            cubecl::logging::CollectivePath::Emulated { why } => {
                CollectivePath::Emulated { why: why.clone() }
            }
            cubecl::logging::CollectivePath::Skipped => CollectivePath::Skipped,
        }
    }
}
