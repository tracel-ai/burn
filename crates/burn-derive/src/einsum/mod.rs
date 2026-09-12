//! Compile literal equations into tensor operations. The public macro wrapper in
//! burn-tensor forwards `$crate`, keeping the expansion independent of dependency renaming.

use burn_einsum::{Axis, ELLIPSIS};
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

/// Parse and plan the equation during expansion, then emit the planned tensor operations.
/// Only tensor dimensions and the number of axes represented by an ellipsis remain dynamic.
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
    let plan = parsed.plan();

    // Bind operands in argument order and evaluate each expression once. Mixed-site names
    // cannot capture identifiers inside an operand expression supplied by the caller.
    let names: Vec<_> = (0..operands.len())
        .map(|index| local(&format!("operand_{index}")))
        .collect();
    let bindings =
        operands
            .iter()
            .zip(&names)
            .zip(&parsed.inputs)
            .map(|((operand, name), labels)| {
                let rank_check = if labels.contains(&ELLIPSIS) {
                    // Ellipsis rank is checked while preparing the operands.
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
    let prepared = local("prepared");
    let width = local("ellipsis_width");
    let output_rank = local("output_rank");
    let result = local("result");
    let last_use = local("last_use");
    let input_ranks = plan.inputs.iter().map(|input| {
        let named_rank = input.named_rank;
        let has_ellipsis = input.has_ellipsis;
        quote! { (#named_rank, #has_ellipsis) }
    });
    let output_dimensions = plan.output_dimensions;
    let total_dimensions = plan.total_dimensions;
    let ellipsis = match plan.ellipsis {
        Some(axis) => quote! { ::core::option::Option::Some(#axis) },
        None => quote! { ::core::option::Option::None },
    };
    let alignments = plan
        .inputs
        .iter()
        .zip(&names)
        .enumerate()
        .map(|(index, (input, name))| {
            let local_width = local(&format!("input_width_{index}"));
            let named_rank = input.named_rank;
            let width_binding = input.has_ellipsis.then(|| {
                quote! { let #local_width = #name.shape().len() - #named_rank; }
            });
            let diagonals = input.diagonals.iter().map(|diagonal| {
                let first = axis_expr(&diagonal.first, &local_width);
                let second = axis_expr(&diagonal.second, &local_width);
                let permutation = axes_expr(&diagonal.permutation, &local_width, &krate);
                let restore = axes_expr(&diagonal.restore, &local_width, &krate);
                quote! { let #name = #name.diagonal(#first, #second, #permutation, #restore); }
            });
            let permutation = axes_expr(&input.permutation, &local_width, &krate);
            let present = &input.axes;
            let shape = local("aligned_shape");
            quote! {
                let #name = #prepared.operands.next().expect("einsum operand count was checked");
                #width_binding
                #(#diagonals)*
                let #name = #name.permute(#permutation);
                let #name = {
                    let #shape = #krate::__einsum::alignment_shape(
                        &#name.shape(), &[#(#present),*], #ellipsis, #width,
                    );
                    #name.reshape(#shape)
                };
            }
        });
    let contractions = plan.contractions.iter().zip(&names[1..]).enumerate().map(
        |(index, (step, right))| {
            let left_reduce = reduce(&result, &step.left_reduce, &width, &krate);
            let right_reduce = reduce(right, &step.right_reduce, &width, &krate);
            let operation = match &step.matmul {
                None => quote! { #result.mul(#right) },
                Some(matmul) => {
                    let swap = matmul
                        .swap
                        .then(|| quote! { let (#result, #right) = (#right, #result); });
                    let shared = axes_expr(&matmul.shared, &width, &krate);
                    let left_axes = axes_expr(&matmul.left, &width, &krate);
                    let right_axes = axes_expr(&matmul.right, &width, &krate);
                    let contraction = axes_expr(&matmul.contraction, &width, &krate);
                    let left_permutation = axes_expr(&matmul.left_permutation, &width, &krate);
                    let right_permutation = axes_expr(&matmul.right_permutation, &width, &krate);
                    let output_permutation = axes_expr(&matmul.output_permutation, &width, &krate);
                    let shared_name = local("shared");
                    let left_name = local("left_axes");
                    let right_name = local("right_axes");
                    let contraction_name = local("contraction");
                    let left_shape = local("left_shape");
                    let right_shape = local("right_shape");
                    let shapes = local("matmul_shapes");
                    quote! {{
                        #swap
                        let #shared_name = #shared;
                        let #left_name = #left_axes;
                        let #right_name = #right_axes;
                        let #contraction_name = #contraction;
                        let #left_shape = #result.shape();
                        let #right_shape = #right.shape();
                        if #krate::__einsum::can_matmul(
                            &#left_shape, &#right_shape, #shared_name,
                            #left_name, #right_name, #contraction_name,
                        ) {
                            let #shapes = #krate::__einsum::matmul_shapes(
                                &#left_shape, &#right_shape, #shared_name,
                                #left_name, #right_name, #contraction_name,
                            );
                            #result.permute(#left_permutation).reshape(#shapes.left)
                                .matmul(#right.permute(#right_permutation).reshape(#shapes.right))
                                .reshape(#shapes.output).permute(#output_permutation)
                        } else {
                            #krate::__einsum::broadcast_contract(#result, #right, #contraction_name)
                        }
                    }}
                }
            };
            let operation = if step.deferred_reduce.is_empty() {
                operation
            } else {
                let operand_index = index + 1;
                let deferred = axes_expr(&step.deferred_reduce, &width, &krate);
                let deferred_name = local("deferred_reduce");
                let mandatory = step
                    .matmul
                    .as_ref()
                    .map_or(&[][..], |matmul| matmul.contraction.as_slice());
                let mandatory = axes_expr(mandatory, &width, &krate);
                quote! {{
                    let #deferred_name = #deferred;
                    if #krate::__einsum::can_contract_early(
                        &#result.shape(), &#right.shape(), #deferred_name,
                        &#last_use, #operand_index,
                    ) {
                        #krate::__einsum::contract_early(
                            #result, #right, #mandatory, #deferred_name,
                            &#last_use, #operand_index,
                        )
                    } else {
                        #operation
                    }
                }}
            };
            quote! {
                let #result = {
                    #left_reduce
                    #right_reduce
                    #operation
                };
            }
        },
    );
    let first = &names[0];
    let final_reduce = reduce(&result, &plan.final_reduce, &width, &krate);

    // When no operand has an ellipsis its expansion is empty, including an ellipsis in the
    // output. Otherwise an output ellipsis needs the contextual Tensor<D> result type.
    let finish = match parsed.output_rank() {
        Some(rank) => {
            let rank = rank.max(1);
            quote! { #result.finish::<#rank>(#output_rank) }
        }
        None => quote! { #result.finish(#output_rank) },
    };
    let expansion_doc = format!(
        "Einsum `{}`: the equation is compiled into diagonal extraction, axis permutations, \
         reshapes, reductions, and {} left-to-right pairwise contraction(s). Each contraction \
         emits multiplication or batched matrix multiplication with its planned axis ordering. \
         Tensor sizes determine reshape dimensions and broadcasting branches at runtime.",
        equation.value(),
        operands.len() - 1,
    );

    Ok(quote! {{
        #[doc = #expansion_doc]
        const _: () = ();
        #(#bindings)*
        let mut #prepared = #krate::__einsum::prepare(
            [#(#names.into()),*], &[#(#input_ranks),*],
            #output_dimensions, #total_dimensions, #ellipsis,
        );
        let #width = #prepared.ellipsis_width;
        let #output_rank = #prepared.output_rank;
        #(#alignments)*
        let #last_use = #krate::__einsum::validate_broadcast(&[#(&#names),*]);
        let #result = #first;
        #(#contractions)*
        #final_reduce
        #finish
    }})
}

fn local(name: &str) -> Ident {
    Ident::new(&format!("__einsum_{name}"), Span::mixed_site())
}

/// Emit a literal slice for fixed axes; only ellipsis equations need runtime expansion.
fn axes_expr(axes: &[Axis], width: &Ident, krate: &Path) -> TokenStream {
    if axes.iter().all(|axis| matches!(axis, Axis::Index(_))) {
        let indices = axes.iter().map(|axis| match axis {
            Axis::Index(index) => index,
            _ => unreachable!(),
        });
        quote! { &[#(#indices),*] }
    } else {
        let axes = axes.iter().map(|axis| match axis {
            Axis::Index(index) => quote! { #krate::__einsum::Axis::Index(#index) },
            Axis::AfterEllipsis(index) => quote! { #krate::__einsum::Axis::AfterEllipsis(#index) },
            Axis::Ellipsis(start) => quote! { #krate::__einsum::Axis::Ellipsis(#start) },
        });
        quote! { &#krate::__einsum::axes(&[#(#axes),*], #width) }
    }
}

fn axis_expr(axis: &Axis, width: &Ident) -> TokenStream {
    match axis {
        Axis::Index(index) => quote! { #index },
        Axis::AfterEllipsis(index) => quote! { #index + #width },
        Axis::Ellipsis(_) => unreachable!("a diagonal always refers to a named axis"),
    }
}

fn reduce(name: &Ident, axes: &[Axis], width: &Ident, krate: &Path) -> TokenStream {
    if axes.is_empty() {
        TokenStream::new()
    } else {
        let axes = axes_expr(axes, width, krate);
        quote! { let #name = #name.sum_dims(#axes); }
    }
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
    fn compiles_layouts_and_unrolls_left_to_right_operations() {
        let expanded = expanded(quote! { burn, "ij,jk,kl->il", a, b, c, });
        assert_eq!(expanded.matches(". matmul (").count(), 2);
        assert_eq!(expanded.matches("[usize ; 2usize]").count(), 3);
        assert!(expanded.contains("finish :: < 2usize >"));
        assert!(expanded.contains(". permute ("));
        assert!(expanded.contains(". reshape ("));
        assert!(!expanded.contains("contract_next"));
        assert!(!expanded.contains("Execution"));
        assert!(!expanded.contains("Plan"));
        assert!(!expanded.contains(":: parse"));
        assert!(expanded.contains("batched matrix multiplication"));
    }

    #[test]
    fn matrix_product_emits_literal_axes_and_direct_matmul_chain() {
        let expanded = expanded(quote! { burn, "ij,jk->ik", a, b });
        assert!(expanded.contains("let __einsum_contraction = & [2usize]"));
        assert!(expanded.contains("let __einsum_left_axes = & [0usize]"));
        assert!(expanded.contains("let __einsum_right_axes = & [1usize]"));
        assert!(expanded.contains(
            "__einsum_result . permute (& [0usize , 2usize , 1usize]) . reshape (__einsum_matmul_shapes . left)"
        ));
        assert!(expanded.contains(
            ". matmul (__einsum_operand_1 . permute (& [2usize , 1usize , 0usize]) . reshape (__einsum_matmul_shapes . right))"
        ));
        assert!(expanded.contains(
            ". reshape (__einsum_matmul_shapes . output) . permute (& [0usize , 2usize , 1usize])"
        ));
        assert!(!expanded.contains(":: axes"));
        assert!(!expanded.contains("Axis ::"));
        assert!(!expanded.contains("can_contract_early"));
    }

    #[test]
    fn later_broadcast_axes_can_contract_early() {
        let expanded = expanded(quote! { burn, "ij,jk,j->ik", a, b, weights });
        assert_eq!(expanded.matches("can_contract_early").count(), 1);
        assert!(
            expanded.contains("let __einsum_last_use = burn :: __einsum :: validate_broadcast")
        );
        assert!(expanded.contains("let __einsum_deferred_reduce = & [2usize]"));
        let early = quote! {
            burn::__einsum::contract_early(
                __einsum_result, __einsum_operand_1, &[], __einsum_deferred_reduce,
                &__einsum_last_use, 1usize,
            )
        };
        assert!(expanded.contains(&early.to_string()));
    }

    #[test]
    fn diagonal_and_reduction_are_emitted_as_operations() {
        let expanded = expanded(quote! { burn, "ii->", matrix });
        assert!(
            expanded.contains(". diagonal (0usize , 1usize , & [0usize , 1usize] , & [0usize])")
        );
        assert!(expanded.contains(". sum_dims (& [0usize])"));
        assert!(!expanded.contains(". matmul ("));
    }

    #[test]
    fn binds_operand_expressions_once_and_preserves_crate_path() {
        let expanded = expanded(quote! {
            ::renamed_burn, "ij,jk->ik", build_left(), build_right()
        });
        assert_eq!(expanded.matches("build_left").count(), 1);
        assert_eq!(expanded.matches("build_right").count(), 1);
        assert!(expanded.contains(":: renamed_burn :: Tensor :: dims"));
        assert!(expanded.contains(":: renamed_burn :: __einsum :: prepare"));
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
        assert!(expanded.contains(". finish (__einsum_output_rank)"));
        assert!(!expanded.contains("[usize ;"));
    }

    #[test]
    fn ellipsis_keeps_only_axis_offsets_and_dimensions_dynamic() {
        let expanded = expanded(quote! { burn, "i...i->...", a });
        let diagonal = quote! {
            .diagonal(
                0usize, 1usize + __einsum_input_width_0,
                &burn::__einsum::axes(&[
                    burn::__einsum::Axis::Ellipsis(1usize),
                    burn::__einsum::Axis::Index(0usize),
                    burn::__einsum::Axis::AfterEllipsis(1usize)
                ], __einsum_input_width_0),
                &burn::__einsum::axes(&[
                    burn::__einsum::Axis::AfterEllipsis(0usize),
                    burn::__einsum::Axis::Ellipsis(0usize)
                ], __einsum_input_width_0)
            )
        };
        assert!(expanded.contains(&diagonal.to_string()));
        assert!(expanded.contains(":: axes"));
        assert!(expanded.contains("Axis :: Ellipsis"));
        assert!(!expanded.contains(":: parse"));
        assert!(!expanded.contains("Plan"));
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
