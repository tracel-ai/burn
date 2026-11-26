use burn::{optim, tensor::Tensor};

#[test]
fn derive_record_state_through_burn() {
    use burn_store::burn_pack::Scalar as StoredScalar;
    use optim::RecordState;

    #[allow(
        dead_code,
        reason = "creates a competing method for the generated conversion"
    )]
    trait CustomInto {
        fn into(self) -> usize;
    }

    impl CustomInto for usize {
        fn into(self) -> usize {
            self
        }
    }

    #[derive(optim::RecordState)]
    struct History {
        step: usize,
        best_step: Option<usize>,
        moments: Vec<Tensor<1>>,
    }

    let history = History {
        step: 7,
        best_step: Some(3),
        moments: Vec::new(),
    };
    let mut sink = optim::StateSink::default();
    history.state_flatten("history", &mut sink);
    assert_eq!(
        sink.scalars,
        [
            (String::from("history.step"), StoredScalar::UInt(7)),
            (String::from("history.best_step"), StoredScalar::UInt(3)),
            (String::from("history.moments.len"), StoredScalar::UInt(0)),
        ]
    );
}
