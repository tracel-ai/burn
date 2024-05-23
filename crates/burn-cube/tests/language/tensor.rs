use burn_cube::{cube, Numeric, Tensor};

#[cube]
fn kernel<T: Numeric>(input: Tensor<T>) {
    // TODO: not the prettiest to be forced to put T even if useless
    let _shape = Tensor::<T>::shape(input, 1u32);
    let _stride = Tensor::<T>::stride(input, 1u32);
    let _length = Tensor::<T>::len(input);
}

mod tests {
    use super::*;
    use burn_cube::{dialect::Item, CubeContext, CubeElem, F32};

    type ElemType = F32;

    #[test]
    fn cube_support_tensor_metadata() {
        let mut context = CubeContext::root();
        let input = context.input(0, Item::new(ElemType::as_elem()));

        kernel_expand::<ElemType>(&mut context, input);
        assert_eq!(
            format!("{:?}", context.into_scope().operations),
            "[Metadata(Shape { \
                dim: ConstantScalar(1.0, UInt), \
                var: GlobalInputArray(0, Item { \
                    elem: Float(F32), \
                    vectorization: 1 \
                }), \
                out: Local(0, Item { \
                    elem: UInt, \
                    vectorization: 1 \
                }, 0) \
            }), \
            Metadata(Stride { \
                dim: ConstantScalar(1.0, UInt), \
                var: GlobalInputArray(0, Item { \
                    elem: Float(F32), \
                    vectorization: 1 \
                }), \
                out: Local(1, Item { \
                    elem: UInt, \
                    vectorization: 1 \
                }, 0) \
            }), \
            Metadata(ArrayLength { \
                var: GlobalInputArray(0, Item { \
                    elem: Float(F32), \
                    vectorization: 1 \
                }), \
                out: Local(2, Item { \
                    elem: UInt, \
                    vectorization: 1 \
                }, 0) \
            })]"
        );
    }
}
