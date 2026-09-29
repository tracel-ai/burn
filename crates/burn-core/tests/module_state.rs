use burn::module::Module;
use burn_core as burn;

#[derive(Module, Debug)]
struct Envelope<M> {
    state: M,
}

#[derive(Module, Debug)]
enum Choice<M> {
    Value(M),
    Other(usize),
}

fn validation<M: Module>(module: &M) -> M {
    module.valid()
}

fn training<M: Module>(module: M) -> M {
    module.train()
}

#[test]
fn stateless_modules_transition_without_autodiff() {
    #[derive(Module, Debug)]
    struct Settings {
        value: usize,
    }

    let settings = training(validation(&Envelope {
        state: Settings { value: 42 },
    }));
    assert_eq!(settings.state.value, 42);

    let choice = training(validation(&Choice::<Settings>::Other(7)));
    assert!(matches!(choice, Choice::Other(7)));
}

#[cfg(feature = "autodiff")]
mod autodiff {
    use super::*;
    use burn::module::{Flag, Lora, Param, ParamId, RunningState};
    use burn::tensor::{Bool, Device, Int, Tensor};

    #[derive(Module, Debug)]
    struct State {
        weight: Param<Tensor<1>>,
        indices: Param<Tensor<1, Int>>,
        mask: Param<Tensor<1, Bool>>,
        raw: Tensor<1>,
        running: RunningState<Tensor<1>>,
        #[module(skip)]
        skipped: Tensor<1>,
    }

    type Nested = Envelope<(Vec<Option<[State; 1]>>, Choice<Param<Flag>>)>;

    fn state(module: &Nested) -> &State {
        &module.state.0[0].as_ref().unwrap()[0]
    }

    fn flag(module: &Nested) -> &Param<Flag> {
        let Choice::Value(flag) = &module.state.1 else {
            panic!("expected the flag variant")
        };
        flag
    }

    #[test]
    fn nested_transitions_preserve_specialized_state_hooks() {
        let device = Device::flex().autodiff();
        let running = RunningState::new(Tensor::zeros([1], &device));
        running.update(Tensor::from_floats([7.0], &device));
        let module = Envelope {
            state: (
                vec![Some([State {
                    weight: Param::from_tensor(Tensor::ones([1], &device)),
                    indices: Param::initialized(ParamId::new(), Tensor::from_ints([1], &device)),
                    mask: Param::initialized(ParamId::new(), Tensor::from_bool([true], &device)),
                    raw: Tensor::ones([1], &device),
                    running,
                    skipped: Tensor::ones([1], &device),
                }])],
                Choice::Value(Param::from_bool(true)),
            ),
        };

        let valid = validation(&module);
        assert!(!state(&valid).weight.is_require_grad());
        assert!(!state(&valid).indices.val().is_autodiff());
        assert!(!state(&valid).mask.val().is_autodiff());
        // Raw tensors are not visited by parameter mappers, but still transition.
        assert!(!state(&valid).raw.is_autodiff());
        assert!(state(&valid).skipped.is_autodiff());
        assert!(!state(&valid).running.value().is_autodiff());
        assert_eq!(state(&valid).running.value().into_scalar::<f32>(), 7.0);
        assert!(!flag(&valid).is_enabled());
        assert_eq!(flag(&valid).id, flag(&module).id);
        assert!(state(&module).weight.is_require_grad());
        assert!(flag(&module).is_enabled());

        // train() must also flush pending running-state updates.
        state(&valid)
            .running
            .update(Tensor::from_floats([9.0], &Device::flex()));
        let trained = training(valid);
        assert!(state(&trained).weight.is_require_grad());
        assert_eq!(state(&trained).weight.id, state(&module).weight.id);
        assert!(state(&trained).indices.val().is_autodiff());
        assert!(state(&trained).mask.val().is_autodiff());
        assert!(state(&trained).raw.is_autodiff());
        assert!(!state(&trained).raw.is_require_grad());
        assert!(state(&trained).running.value().is_autodiff());
        assert_eq!(state(&trained).running.value().into_scalar::<f32>(), 9.0);
        assert!(flag(&trained).is_enabled());
    }

    #[test]
    fn training_enables_nested_adapter_parameters_without_unfreezing_the_base() {
        let device = Device::flex();
        let weight =
            Param::from_tensor(Tensor::<2>::ones([4, 4], &device)).apply_lora(Lora::new(2, 4.0));
        let adapter = weight.adapter().unwrap();
        let a_id = adapter.a.id;
        let b_id = adapter.b.id;

        let trained = training(weight);
        assert!(trained.base().is_autodiff());
        assert!(!trained.base().is_require_grad());
        let adapter = trained.adapter().unwrap();
        assert_eq!(adapter.a.id, a_id);
        assert_eq!(adapter.b.id, b_id);
        assert!(adapter.a.is_require_grad());
        assert!(adapter.b.is_require_grad());
    }
}
