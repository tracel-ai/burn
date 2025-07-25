#[cfg(test)]
mod tests {
    use burn::{
        module::Module,
        nn::{Linear, LinearConfig},
        prelude::Backend,
        record::FullPrecisionSettings,
        record::Recorder,
    };
    use burn_import::safetensors::{
        AdapterType, LoadArgs, SafetensorsFileRecorder, to_safetensors,
    };
    use std::fs::File;
    use std::io::Write;

    #[derive(Module, Debug)]
    struct Mlp<B: Backend> {
        l1: Linear<B>,
        l2: Linear<B>,
    }

    type B = burn_ndarray::NdArray;

    #[test]
    fn serialization() {
        let device = burn_ndarray::NdArrayDevice::Cpu;
        let model = Mlp {
            l1: LinearConfig::new(3, 2).init::<B>(&device),
            l2: LinearConfig::new(7, 1).init::<B>(&device),
        };

        let serialized = to_safetensors(model.clone()).unwrap();
        let mut file = File::create("model.safetensors").unwrap();
        file.write_all(&serialized).unwrap();
        let record = SafetensorsFileRecorder::<FullPrecisionSettings>::default()
            .load(
                LoadArgs::new("model.safetensors".into()).with_adapter_type(AdapterType::NoAdapter),
                &device,
            )
            .expect("Should decode state successfully");
        std::fs::remove_file("model.safetensors").unwrap();

        let model_deserialized = Mlp {
            l1: LinearConfig::new(3, 2).init::<B>(&device),
            l2: LinearConfig::new(7, 1).init::<B>(&device),
        }
        .load_record(record);

        assert!(
            model_deserialized
                .l1
                .weight
                .val()
                .all_close(model.l1.weight.val(), None, None)
        );
        assert!(model_deserialized.l1.bias.unwrap().val().all_close(
            model.l1.bias.unwrap().val(),
            None,
            None,
        ));
        assert!(
            model_deserialized
                .l2
                .weight
                .val()
                .all_close(model.l2.weight.val(), None, None)
        );
        assert!(model_deserialized.l2.bias.unwrap().val().all_close(
            model.l2.bias.unwrap().val(),
            None,
            None,
        ));
    }
}
