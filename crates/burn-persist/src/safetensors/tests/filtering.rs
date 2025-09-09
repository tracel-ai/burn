use crate::{ModulePersist, PathFilter, SafetensorsPersister};

use super::round_trip::ComplexModule;

type TestBackend = burn_ndarray::NdArray;

#[test]
#[cfg(target_has_atomic = "ptr")]
fn filtered_export_import() {
    let device = Default::default();
    let module1 = ComplexModule::<TestBackend>::new(&device);
    let mut module2 = ComplexModule::<TestBackend>::new_zeros(&device);

    // Export only encoder tensors
    let mut save_persister = SafetensorsPersister::from_bytes(None)
        .filter(PathFilter::new().with_regex(r"^encoder\..*"));
    module1.collect_to(&mut save_persister).unwrap();

    // Import filtered tensors - need to allow partial since we only saved encoder tensors
    let mut load_persister = SafetensorsPersister::from_bytes(None).allow_partial(true);
    if let SafetensorsPersister::Memory(ref mut p) = load_persister {
        if let SafetensorsPersister::Memory(ref p_save) = save_persister {
            // Get Arc and extract data
            let data_arc = p_save.data().unwrap();
            p.set_data(data_arc.as_ref().clone());
        }
    }
    let result = module2.apply_from(&mut load_persister).unwrap();

    assert!(result.is_success());
    assert_eq!(result.applied.len(), 3); // encoder.weight, encoder.bias, encoder.norm
    assert!(result.missing.len() > 0); // decoder and layers tensors are missing
}
