use burn::module::{Module, ModuleDisplay, ModuleMapper, ModuleVisitor, Param};
use burn::tensor::{Device, Tensor};
use burn_core as burn;
use core::marker::PhantomData;

#[derive(Clone, Debug, PartialEq)]
struct Config(usize);

#[derive(Module, Debug)]
struct StateOnly<T> {
    #[module(skip)]
    config: T,
}

trait Forward: Module {
    fn forward(&self, input: Tensor<1>) -> Tensor<1>;
}

#[derive(Module, Debug)]
struct Scale {
    weight: Param<Tensor<1>>,
}

impl Forward for Scale {
    fn forward(&self, input: Tensor<1>) -> Tensor<1> {
        input * self.weight.val()
    }
}

#[derive(Module, Debug)]
struct Residual<M: Forward, C> {
    inner: M,
    #[module(skip)]
    config: C,
    #[module(skip)]
    cached: Option<M>,
}

impl<M: Forward, C> Residual<M, C> {
    fn forward(&self, input: Tensor<1>) -> Tensor<1> {
        self.inner.forward(input.clone()) + input
    }
}

#[derive(Module, Debug)]
struct TypedBlock<T> {
    weight: Param<Tensor<1>>,
    #[module(skip)]
    marker: PhantomData<T>,
}

#[derive(Module, Debug)]
struct Wrapped<T> {
    inner: TypedBlock<T>,
}

trait HasLayer {
    type Layer;
}

#[derive(Debug)]
struct Layers;
impl HasLayer for Layers {
    type Layer = Scale;
}

#[derive(Module, Debug)]
// Sharing the associated type's name must not make this field recursive.
struct Layer<T: HasLayer>
where
    T::Layer: core::fmt::Debug,
{
    inner: T::Layer,
}

#[derive(Module, Debug)]
struct Tree<T> {
    layer: T,
    children: Vec<Self>,
    next: Option<Box<Tree<T>>>,
}

#[derive(Module, Debug)]
enum Recursive<M> {
    Leaf(M),
    Branch(Box<Self>),
}

fn scale(device: &Device) -> Scale {
    Scale {
        weight: Param::from_tensor(Tensor::ones([2], device)),
    }
}

#[test]
fn generic_state_does_not_require_module() {
    let module = StateOnly { config: Config(7) };
    assert_eq!(module.num_params(), 0);
    assert_eq!(module.valid().train().config, Config(7));
}

#[test]
fn generic_module_can_also_be_used_in_skipped_state() {
    let device = Device::flex();
    let module = Residual {
        inner: scale(&device),
        config: Config(7),
        cached: Some(scale(&device)),
    };
    assert_eq!(module.num_params(), 2);
    let output = module.forward(Tensor::ones([2], &device));
    assert_eq!(
        output.to_data().try_to_vec::<f32>().unwrap(),
        vec![2.0, 2.0]
    );
}

#[test]
fn module_bounds_apply_to_fields_instead_of_all_type_arguments() {
    let device = Device::flex();
    #[derive(Debug)]
    struct NotClone;
    let module = Wrapped::<NotClone> {
        inner: TypedBlock {
            weight: scale(&device).weight,
            marker: PhantomData,
        },
    };
    assert_eq!(module.num_params(), 2);
    let associated = Layer::<Layers> {
        inner: scale(&device),
    };
    assert_eq!(associated.num_params(), 2);
}

#[test]
fn recursive_children_are_visited_and_mapped() {
    let device = Device::flex();
    let leaf = || Tree {
        layer: scale(&device),
        children: vec![],
        next: None,
    };
    let module = Tree {
        layer: scale(&device),
        children: vec![leaf()],
        next: Some(Box::new(leaf())),
    };

    #[derive(Default)]
    struct Paths {
        path: Vec<String>,
        leaves: Vec<String>,
    }
    impl ModuleVisitor for Paths {
        fn enter_module(&mut self, name: &str, _: &str) {
            self.path.push(name.into());
        }
        fn exit_module(&mut self, _: &str, _: &str) {
            self.path.pop();
        }
        fn visit_float<const D: usize>(&mut self, _: &Param<Tensor<D>>) {
            self.leaves.push(self.path.join("."));
        }
    }
    struct Double;
    impl ModuleMapper for Double {
        fn map_float<const D: usize>(&mut self, param: Param<Tensor<D>>) -> Param<Tensor<D>> {
            param.map(|tensor| tensor * 2.0)
        }
    }
    let mut paths = Paths::default();
    module.visit(&mut paths);
    assert_eq!(
        paths.leaves,
        [
            "layer.weight",
            "children.0.layer.weight",
            "next.layer.weight"
        ]
    );
    assert_eq!(module.num_params(), 6);
    let mapped = module.clone().map(&mut Double);
    let restored = module.load_record(mapped.into_record());
    for layer in [
        &restored.layer,
        &restored.children[0].layer,
        &restored.next.as_ref().unwrap().layer,
    ] {
        assert_eq!(
            layer.weight.val().to_data().try_to_vec::<f32>().unwrap(),
            vec![2.0, 2.0]
        );
    }
}

#[test]
fn recursive_enum_can_own_a_boxed_child() {
    let device = Device::flex();
    let module = Recursive::Branch(Box::new(Recursive::Leaf(scale(&device))));
    assert_eq!(module.num_params(), 2);
    assert!(module.to_string().contains("Leaf"));
}

#[test]
fn recursive_containers_infer_bounds_on_other_children() {
    #[derive(Module, Debug)]
    struct Tree<M> {
        children: Vec<(M, Self)>,
    }

    let device = Device::flex();
    let module = Tree {
        children: vec![(scale(&device), Tree { children: vec![] })],
    };
    assert_eq!(module.num_params(), 2);
}

#[test]
fn enum_variants_can_hold_skipped_generic_state() {
    #[derive(Module, Debug)]
    enum Adapter<M: Module, C> {
        Layer(M),
        Cached(#[module(skip)] M),
        Config(#[module(skip)] C),
    }
    let device = Device::flex();
    let layer = Adapter::<Scale, Config>::Layer(scale(&device));
    let cached = Adapter::<Scale, Config>::Cached(scale(&device));
    let config = Adapter::<Scale, Config>::Config(Config(7));
    assert_eq!(layer.num_params(), 2);
    for module in [cached, config] {
        assert_eq!(module.num_params(), 0);
        assert!(module.devices().is_empty());
        assert_eq!(module.valid().train().to_string(), module.to_string());
    }
}

#[test]
fn tuple_structs_support_generic_modules_and_state() {
    #[derive(Module, Debug)]
    struct Adapter<M, C>(M, #[module(skip)] C);
    let device = Device::flex();
    let module = Adapter(scale(&device), Config(7));
    assert_eq!(module.num_params(), 2);
    let module = module.valid();
    assert_eq!(module.1, Config(7));
    assert!(module.to_string().contains("Config(7)"));
}

#[test]
fn boxes_preserve_custom_module_display() {
    use burn::module::{Content, DisplaySettings};

    #[derive(Module, Debug)]
    #[module(custom_display)]
    struct Custom;

    impl ModuleDisplay for Custom {
        fn custom_settings(&self) -> Option<DisplaySettings> {
            Some(DisplaySettings::new().with_new_line_after_attribute(false))
        }

        fn custom_content(&self, content: Content) -> Option<Content> {
            content.add("label", "custom").optional()
        }
    }

    assert_eq!(
        ModuleDisplay::format(&Box::new(Custom), Default::default()),
        Custom.to_string()
    );
}
