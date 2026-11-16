//! Which member of the profiled device an execution ran on.

use burn_tensor::Device;
use cubecl::RuntimeId;
use cubecl::logging::ExecutionPlace;
use cubecl_common::device::DeviceId;

/// A member of the profiled device, by its index in member order: `0` on a device of one.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Member(pub usize);

/// The device a window ran on, as its execution's place names it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum RanOn {
    /// The one device a device of one runs everything on.
    Device,
    /// A member of a sharded compute device.
    Member(Member),
    /// A device that is no member of the sharded compute device profiled, so it fills no
    /// member's gaps and opens none.
    OffMembers,
}

/// The devices of a sharded compute device's members, in member order: what places an execution
/// on a member. Empty for a device of one.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MemberList {
    /// Each member's id as its runtime hands it out, which is what an execution's place names.
    devices: Vec<DeviceId>,
}

impl MemberList {
    /// The members of `device` where it is a sharded compute device of cubecl members.
    pub fn of(device: &Device) -> Self {
        Self::over(device.as_dispatch().shard_member_ids())
    }

    /// The members whose cubecl device ids are `ids`, in member order.
    ///
    /// A device named twice is kept once: its executions name one place, so they cannot be told
    /// apart by member, and the device is timed as one.
    pub fn over(ids: Vec<DeviceId>) -> Self {
        let mut devices: Vec<DeviceId> = Vec::with_capacity(ids.len());
        for id in ids.into_iter().map(RuntimeId::strip) {
            if !devices.contains(&id) {
                devices.push(id);
            }
        }
        Self { devices }
    }

    /// How many members there are to time apart, one for a device of one.
    pub fn count(&self) -> usize {
        self.devices.len().max(1)
    }

    /// Whether the device has more than one member, so each is timed apart.
    pub fn several(&self) -> bool {
        self.devices.len() > 1
    }

    /// Where an execution placed at `place` ran.
    pub fn ran_on(&self, place: &ExecutionPlace) -> RanOn {
        if !self.several() {
            return RanOn::Device;
        }
        let device = RuntimeId::strip(place.service.device);
        self.devices
            .iter()
            .position(|member| *member == device)
            .map_or(RanOn::OffMembers, |index| RanOn::Member(Member(index)))
    }
}

#[cfg(test)]
mod tests {
    use cubecl::logging::StreamRole;
    use cubecl_common::device::ServiceId;

    use super::*;

    /// A member is found by its device whatever runtime tag its id carries: the shard names its
    /// members as cubecl devices, which stamp their runtime, and a place names the runtime's own.
    #[test]
    fn an_execution_is_placed_on_the_member_whose_device_it_names() {
        let members = MemberList::over(vec![stamped(3), stamped(5)]);

        assert_eq!(members.count(), 2);
        assert_eq!(members.ran_on(&place(5)), RanOn::Member(Member(1)));
        assert_eq!(members.ran_on(&place(3)), RanOn::Member(Member(0)));
        assert_eq!(members.ran_on(&place(4)), RanOn::OffMembers);
    }

    /// A sharded compute device whose members are one device is timed as one.
    #[test]
    fn a_device_named_twice_is_one_member() {
        let members = MemberList::over(vec![stamped(3), stamped(3)]);

        assert!(!members.several());
        assert_eq!(members.ran_on(&place(4)), RanOn::Device);
    }

    fn stamped(index_id: u16) -> DeviceId {
        RuntimeId::Hip.stamp(DeviceId::new(0, index_id))
    }

    fn place(index_id: u16) -> ExecutionPlace {
        ExecutionPlace {
            service: ServiceId::of::<()>(DeviceId::new(0, index_id)),
            stream: StreamRole::Collective,
        }
    }
}
