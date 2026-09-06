//! Compile literal equations into einsum execution stages. The public macro wrapper in
//! burn-tensor forwards `$crate`, keeping the expansion independent of dependency renaming.

use burn_einsum::ELLIPSIS;
use proc_macro2::{Span, TokenStream};
use quote::{quote, quote_spanned};
use syn::{
    Expr, Ident, LitStr, Path, Token,
    parse::{Parse, ParseStream},
    punctuated::Punctuated,
    spanned::Spanned,
};

/// The crate path, literal equation, and operands forwarded by the public wrapper.
pub(crate) struct EinsumInput {
    krate: Path,
    equation: LitStr,
    operands: Vec<Expr>,
}

impl Parse for EinsumInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let krate = input.parse()?;
        input.parse::<Token![,]>()?;
        let equation = input.parse()?;
        let operands = if input.is_empty() {
            Vec::new()
        } else {
            input.parse::<Token![,]>()?;
            Punctuated::<Expr, Token![,]>::parse_terminated(input)?
                .into_iter()
                .collect()
        };
        Ok(Self {
            krate,
            equation,
            operands,
        })
    }
}

/// Parse and validate the literal now, then emit axis descriptors and a contraction stage for
/// each remaining operand. Shapes determine the concrete permutations and matmul dimensions
/// when the generated code runs; the equation never needs to be parsed again at runtime.
pub(crate) fn expand(input: EinsumInput) -> syn::Result<TokenStream> {
    let EinsumInput {
        krate,
        equation,
        operands,
    } = input;
    let parsed = burn_einsum::parse(&equation.value())
        .map_err(|error| syn::Error::new(equation.span(), error.to_string()))?;
    if parsed.inputs.len() != operands.len() {
        return Err(syn::Error::new(
            equation.span(),
            format!(
                "einsum equation describes {} operand(s), but {} were supplied",
                parsed.inputs.len(),
                operands.len()
            ),
        ));
    }

    // Bind operands in argument order and evaluate each expression once. Mixed-site names
    // cannot capture identifiers inside an operand expression supplied by the caller.
    let names: Vec<_> = (0..operands.len())
        .map(|index| Ident::new(&format!("__einsum_operand_{index}"), Span::mixed_site()))
        .collect();
    let bindings =
        operands
            .iter()
            .zip(&names)
            .zip(&parsed.inputs)
            .map(|((operand, name), labels)| {
                let rank_check = if labels.contains(&ELLIPSIS) {
                    // Ellipsis rank is checked during alignment. This still ensures the argument
                    // is a Tensor, through the same path as the exact-rank check below.
                    quote! { let _ = #krate::Tensor::dims(&#name); }
                } else {
                    // Burn represents scalar values with Tensor<1> of shape [1].
                    let rank = labels.len().max(1);
                    quote! { let _: [usize; #rank] = #krate::Tensor::dims(&#name); }
                };
                quote_spanned! { operand.span() =>
                    let #name = #operand;
                    #rank_check
                }
            });
    let input_labels = parsed
        .inputs
        .iter()
        .map(|labels| quote! { &[#(#labels),*] });
    let output_labels = &parsed.output;

    // When no operand has an ellipsis its expansion is empty, including an ellipsis in the
    // output. Otherwise an output ellipsis needs the contextual Tensor<D> result type.
    let finish = match parsed.output_rank() {
        Some(rank) => {
            let rank = rank.max(1);
            quote! { .finish::<#rank>() }
        }
        None => quote! { .finish() },
    };
    let contractions = (1..operands.len()).map(|_| quote! { .contract_next() });
    let expansion_doc = format!(
        "Einsum `{}`: bind and check {} operand(s), align their axes from the compiled label \
         descriptors, perform {} left-to-right pairwise contraction(s), then restore the output \
         axes. Each pair reduces axes no longer needed by later operands and lowers to \
         multiplication or batched matrix multiplication according to the runtime dimensions.",
        equation.value(),
        operands.len(),
        operands.len() - 1,
    );

    Ok(quote! {{
        #[doc = #expansion_doc]
        const _: () = ();
        #(#bindings)*
        #krate::__einsum::Execution::new(
            &[#(#input_labels),*],
            &[#(#output_labels),*],
            [#(#names.into()),*],
        )
        #(#contractions)*
        #finish
    }})
}

#[cfg(test)]
mod tests {
    use super::*;

    fn expanded(input: TokenStream) -> String {
        let input = syn::parse2(input).unwrap();
        let expanded = expand(input).unwrap();
        // Check that every tested expansion is syntactically a Rust expression.
        syn::parse2::<syn::ExprBlock>(expanded.clone()).unwrap();
        expanded.to_string()
    }

    #[test]
    fn compiles_labels_and_unrolls_left_to_right_stages() {
        let expanded = expanded(quote! { burn, "ij,jk,kl->il", a, b, c, });
        assert_eq!(expanded.matches("contract_next").count(), 2);
        assert_eq!(expanded.matches("[usize ; 2usize]").count(), 3);
        assert!(expanded.contains("finish :: < 2usize >"));
        assert!(expanded.contains("[34u8 , 35u8]"));
        assert!(expanded.contains("[34u8 , 37u8]"));
        assert!(!expanded.contains(":: parse"));
        assert!(expanded.contains("batched matrix multiplication"));
    }

    #[test]
    fn binds_operand_expressions_once_and_preserves_crate_path() {
        let expanded = expanded(quote! {
            ::renamed_burn, "ij,jk->ik", build_left(), build_right()
        });
        assert_eq!(expanded.matches("build_left").count(), 1);
        assert_eq!(expanded.matches("build_right").count(), 1);
        assert!(expanded.contains(":: renamed_burn :: Tensor :: dims"));
        assert!(expanded.contains(":: renamed_burn :: __einsum :: Execution :: new"));
    }

    #[test]
    fn scalar_input_and_output_use_rank_one() {
        let expanded = expanded(quote! { burn, "->", scalar });
        assert!(expanded.contains("[usize ; 1usize]"));
        assert!(expanded.contains("finish :: < 1usize >"));
        assert!(!expanded.contains("contract_next"));
    }

    #[test]
    fn single_operand_can_reorder_implicit_output() {
        let expanded = expanded(quote! { burn, "ji", matrix });
        assert!(!expanded.contains("contract_next"));
        assert!(expanded.contains("finish :: < 2usize >"));
    }

    #[test]
    fn output_ellipsis_uses_contextual_rank() {
        let expanded = expanded(quote! { burn, "...ij,...jk->...ik", a, b });
        assert!(expanded.contains(". finish ()"));
        assert!(!expanded.contains("[usize ;"));
    }

    #[test]
    fn reduced_ellipsis_has_exact_output_rank() {
        let expanded = expanded(quote! { burn, "...ij->j", a });
        assert!(expanded.contains("finish :: < 1usize >"));
    }

    #[test]
    fn zero_width_output_ellipsis_has_exact_output_rank() {
        let expanded = expanded(quote! { burn, "ij->...ji", a });
        assert!(expanded.contains("finish :: < 2usize >"));
    }

    #[test]
    fn rejects_invalid_equation_at_expansion() {
        let input = syn::parse2(quote! { burn, "ij->ii", a }).unwrap();
        assert!(expand(input).is_err());
    }

    #[test]
    fn rejects_operand_count_mismatch_at_expansion() {
        let input = syn::parse2(quote! { burn, "ij,jk->ik", a }).unwrap();
        assert_eq!(
            expand(input).unwrap_err().to_string(),
            "einsum equation describes 2 operand(s), but 1 were supplied"
        );
    }

    #[test]
    fn rejects_missing_operands_at_expansion() {
        let input = syn::parse2(quote! { burn, "i->i" }).unwrap();
        assert!(expand(input).is_err());
    }

    #[test]
    fn rejects_non_literal_equation() {
        assert!(syn::parse2::<EinsumInput>(quote! { burn, equation, a }).is_err());
    }
}
