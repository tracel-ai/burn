//! Which member of the profiled device an execution ran on.

use burn_tensor::Device;
use cubecl::logging::ExecutionPlace;
use cubecl_common::device::ServiceId;

/// A member of the profiled device, by its index in member order: `0` on a device of one.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Member(pub usize);

/// The device a window ran on, as its execution's place names it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum RanOn {
    /// The one device a device of one runs everything on.
    Device,
    /// A member of a sharded compute device, or one of the devices a run is timed across.
    Member(Member),
    /// A device that is no member of what was profiled, so it fills no member's gaps and opens
    /// none.
    OffMembers,
}

/// The cubecl devices a run is timed on one by one, in member order: what places an execution on
/// a member. Empty for a device of one.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MemberList {
    /// Each member's device with the runtime serving it, as an execution's place names it.
    services: Vec<ServiceId>,
}

impl MemberList {
    /// The members of a run on `devices`: each cubecl device in order, a sharded compute device's
    /// members in member order. None when that is one device at most.
    ///
    /// Opens each member's client, so it syncs `devices` first: opening one waits on the device's
    /// worker while holding cubecl's registry, which work still running there can need.
    pub fn of(devices: &[Device]) -> Self {
        let ids: Vec<_> = devices
            .iter()
            .flat_map(|device| device.as_dispatch().member_ids())
            .collect();
        if ids.len() < 2 {
            return Self::default();
        }
        for device in devices {
            device
                .sync()
                .expect("the device answers before its members are reached");
        }
        Self::over(
            ids.into_iter()
                .map(|id| cubecl::Device::from_id(id).client().service_id())
                .collect(),
        )
    }

    /// The members whose devices are served at `services`, in member order.
    ///
    /// A device named twice is kept once: its executions name one place, so they cannot be told
    /// apart by member, and the device is timed as one.
    pub fn over(services: Vec<ServiceId>) -> Self {
        let mut kept: Vec<ServiceId> = Vec::with_capacity(services.len());
        for service in services {
            if !kept.contains(&service) {
                kept.push(service);
            }
        }
        Self { services: kept }
    }

    /// How many members there are to time apart, one for a device of one.
    pub fn count(&self) -> usize {
        self.services.len().max(1)
    }

    /// Whether the device has more than one member, so each is timed apart.
    pub fn several(&self) -> bool {
        self.services.len() > 1
    }

    /// Where an execution placed at `place` ran.
    pub fn ran_on(&self, place: &ExecutionPlace) -> RanOn {
        if !self.several() {
            return RanOn::Device;
        }
        self.services
            .iter()
            .position(|service| *service == place.service)
            .map_or(RanOn::OffMembers, |index| RanOn::Member(Member(index)))
    }
}

#[cfg(test)]
mod tests {
    use cubecl::logging::StreamRole;
    use cubecl_common::device::DeviceId;

    use super::*;

    /// A member is found by its device and the runtime serving it: two runtimes number their
    /// devices alike.
    #[test]
    fn an_execution_is_placed_on_the_member_whose_service_it_names() {
        let members = MemberList::over(vec![service::<u8>(0), service::<u16>(0)]);

        assert_eq!(members.count(), 2);
        assert_eq!(
            members.ran_on(&place(service::<u16>(0))),
            RanOn::Member(Member(1))
        );
        assert_eq!(
            members.ran_on(&place(service::<u8>(0))),
            RanOn::Member(Member(0))
        );
        assert_eq!(members.ran_on(&place(service::<u8>(1))), RanOn::OffMembers);
    }

    /// A sharded compute device whose members are one device is timed as one.
    #[test]
    fn a_device_named_twice_is_one_member() {
        let members = MemberList::over(vec![service::<u8>(3), service::<u8>(3)]);

        assert!(!members.several());
        assert_eq!(members.ran_on(&place(service::<u8>(4))), RanOn::Device);
    }

    /// The device `index_id` served by a runtime named `S`.
    fn service<S: 'static>(index_id: u16) -> ServiceId {
        ServiceId::of::<S>(DeviceId::new(0, index_id))
    }

    fn place(service: ServiceId) -> ExecutionPlace {
        ExecutionPlace {
            service,
            stream: StreamRole::Collective,
        }
    }
}
