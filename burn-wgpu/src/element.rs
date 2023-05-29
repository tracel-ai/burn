use burn_tensor::Element;

pub trait FloatElement: Element {
    fn type_name() -> &'static str;
}
pub trait IntElement: Element {
    fn type_name() -> &'static str;
}
