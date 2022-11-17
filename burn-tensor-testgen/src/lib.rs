use proc_macro::TokenStream;
use quote::{format_ident, quote};

#[proc_macro_attribute]
pub fn testgen(attr: TokenStream, item: TokenStream) -> TokenStream {
    let item: proc_macro2::TokenStream = proc_macro2::TokenStream::from(item);
    let attr: proc_macro2::TokenStream = proc_macro2::TokenStream::from(attr);
    let macro_ident = format_ident!("testgen_{}", attr.to_string());

    let macro_gen = quote! {
        #[macro_export]
        macro_rules! #macro_ident {
            () => {
                mod #attr {
                    use super::*;

                    type TestADBackend = burn_tensor::backend::ADBackendDecorator<TestBackend>;
                    type TestADTensor<const D: usize> = burn_tensor::Tensor<TestADBackend, D>;

                    #item
                }
            };
        }
    };

    let test_gen = quote! {
        #[cfg(test)]
        use crate::tests::TestBackend;
        #[cfg(test)]
        use crate as burn_tensor;
        #[cfg(test)]
        type TestADBackend = burn_tensor::backend::ADBackendDecorator<TestBackend>;
        #[cfg(test)]
        type TestTensor<const D: usize> = burn_tensor::Tensor<TestBackend, D>;
        #[cfg(test)]
        type TestADTensor<const D: usize> = burn_tensor::Tensor<TestADBackend, D>;
        #[cfg(test)]
        #item
    };

    quote! {
        #test_gen
        #macro_gen
    }
    .into()
}
