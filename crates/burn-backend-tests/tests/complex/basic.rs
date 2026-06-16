use burn_tensor::{Bool, ComplexScalar, IndexingUpdateOp, Int, Tensor, TensorData};

#[test]
fn test_complex_zeros() {
    let tensor = TestTensor::<2>::zeros([2, 3], &Default::default());
    let data = tensor.into_data();

    let expected = TensorData::from([
        [
            ComplexScalar::<f32> {
                real: 0.0,
                imag: 0.0,
            },
            ComplexScalar::<f32> {
                real: 0.0,
                imag: 0.0,
            },
            ComplexScalar::<f32> {
                real: 0.0,
                imag: 0.0,
            },
        ],
        [
            ComplexScalar::<f32> {
                real: 0.0,
                imag: 0.0,
            },
            ComplexScalar::<f32> {
                real: 0.0,
                imag: 0.0,
            },
            ComplexScalar::<f32> {
                real: 0.0,
                imag: 0.0,
            },
        ],
    ]);

    data.assert_eq(&expected, false);
}

#[test]
fn test_complex_ones() {
    let tensor = TestTensor::<2>::ones([2, 2], &Default::default());
    let data = tensor.into_data();

    let expected = TensorData::from([
        [
            ComplexScalar::<f32> {
                real: 1.0,
                imag: 0.0,
            },
            ComplexScalar::<f32> {
                real: 1.0,
                imag: 0.0,
            },
        ],
        [
            ComplexScalar::<f32> {
                real: 1.0,
                imag: 0.0,
            },
            ComplexScalar::<f32> {
                real: 1.0,
                imag: 0.0,
            },
        ],
    ]);

    data.assert_eq(&expected, false);
}

#[test]
fn test_complex_from_data() {
    let data = TensorData::from([[
        ComplexScalar::<f32> {
            real: 1.0,
            imag: 2.0,
        },
        ComplexScalar::<f32> {
            real: 3.0,
            imag: 4.0,
        },
    ]]);

    let tensor = TestTensor::<2>::from_data(data.clone(), &Default::default());
    let result = tensor.into_data();

    result.assert_eq(&data, false);
}

#[test]
fn test_complex_reshape() {
    let tensor = TestTensor::<2>::from_data(
        TensorData::from([[
            ComplexScalar::<f32> {
                real: 1.0,
                imag: 2.0,
            },
            ComplexScalar::<f32> {
                real: 3.0,
                imag: 4.0,
            },
            ComplexScalar::<f32> {
                real: 5.0,
                imag: 6.0,
            },
            ComplexScalar::<f32> {
                real: 7.0,
                imag: 8.0,
            },
        ]]),
        &Default::default(),
    );

    let result: TestTensor<3> = tensor.reshape([2, 2, 1]);
    let data = result.into_data();

    let expected = TensorData::from([
        [
            [ComplexScalar::<f32> {
                real: 1.0,
                imag: 2.0,
            }],
            [ComplexScalar::<f32> {
                real: 3.0,
                imag: 4.0,
            }],
        ],
        [
            [ComplexScalar::<f32> {
                real: 5.0,
                imag: 6.0,
            }],
            [ComplexScalar::<f32> {
                real: 7.0,
                imag: 8.0,
            }],
        ],
    ]);

    data.assert_eq(&expected, false);
}

#[test]
fn test_complex_transpose() {
    let tensor = TestTensor::<2>::from_data(
        TensorData::from([
            [
                ComplexScalar::<f32> {
                    real: 1.0,
                    imag: 2.0,
                },
                ComplexScalar::<f32> {
                    real: 3.0,
                    imag: 4.0,
                },
            ],
            [
                ComplexScalar::<f32> {
                    real: 5.0,
                    imag: 6.0,
                },
                ComplexScalar::<f32> {
                    real: 7.0,
                    imag: 8.0,
                },
            ],
        ]),
        &Default::default(),
    );

    let result = tensor.transpose();
    let data = result.into_data();

    let expected = TensorData::from([
        [
            ComplexScalar::<f32> {
                real: 1.0,
                imag: 2.0,
            },
            ComplexScalar::<f32> {
                real: 5.0,
                imag: 6.0,
            },
        ],
        [
            ComplexScalar::<f32> {
                real: 3.0,
                imag: 4.0,
            },
            ComplexScalar::<f32> {
                real: 7.0,
                imag: 8.0,
            },
        ],
    ]);

    data.assert_eq(&expected, false);
}

#[test]
fn complex_cast() {
    let tensor = TestTensor::<2>::from_data(
        TensorData::from([[
            ComplexScalar::<f32> {
                real: 1.0,
                imag: 2.0,
            },
            ComplexScalar::<f32> {
                real: 3.0,
                imag: 4.0,
            },
        ]]),
        &Default::default(),
    );

    let result = tensor.cast(burn_backend::ComplexDType::Complex64);
    let data = result.into_data();

    let expected = TensorData::from([[
        ComplexScalar::<f64> {
            real: 1.0,
            imag: 2.0,
        },
        ComplexScalar::<f64> {
            real: 3.0,
            imag: 4.0,
        },
    ]]);

    data.assert_eq(&expected, false);
}

#[test]
fn complex_cast_to_float() {
    let tensor = TestTensor::<2>::from_data(
        TensorData::from([[
            ComplexScalar::<f32> {
                real: 1.0,
                imag: 2.0,
            },
            ComplexScalar::<f32> {
                real: 3.0,
                imag: 4.0,
            },
        ]]),
        &Default::default(),
    );

    let result = tensor.cast(burn_backend::FloatDType::F32);
    let data = result.into_data();

    let expected = TensorData::from([[
        1.0,
        3.0,
    ]]);

    data.assert_eq(&expected, false);
}
#[test]
fn test_complex_swap_dims() {
    let tensor = TestTensor::<2>::from_data(
        TensorData::from([
            [
                ComplexScalar::<f32> {
                    real: 1.0,
                    imag: 2.0,
                },
                ComplexScalar::<f32> {
                    real: 3.0,
                    imag: 4.0,
                },
            ],
            [
                ComplexScalar::<f32> {
                    real: 5.0,
                    imag: 6.0,
                },
                ComplexScalar::<f32> {
                    real: 7.0,
                    imag: 8.0,
                },
            ],
        ]),
        &Default::default(),
    );

    let data = tensor.swap_dims(0, 1).into_data();

    let expected = TensorData::from([
        [
            ComplexScalar::<f32> {
                real: 1.0,
                imag: 2.0,
            },
            ComplexScalar::<f32> {
                real: 5.0,
                imag: 6.0,
            },
        ],
        [
            ComplexScalar::<f32> {
                real: 3.0,
                imag: 4.0,
            },
            ComplexScalar::<f32> {
                real: 7.0,
                imag: 8.0,
            },
        ],
    ]);

    data.assert_eq(&expected, false);
}

#[test]
fn test_complex_slice() {
    let tensor = TestTensor::<2>::from_data(
        TensorData::from([
            [
                ComplexScalar::<f32> {
                    real: 1.0,
                    imag: 1.0,
                },
                ComplexScalar::<f32> {
                    real: 2.0,
                    imag: 2.0,
                },
                ComplexScalar::<f32> {
                    real: 3.0,
                    imag: 3.0,
                },
            ],
            [
                ComplexScalar::<f32> {
                    real: 4.0,
                    imag: 4.0,
                },
                ComplexScalar::<f32> {
                    real: 5.0,
                    imag: 5.0,
                },
                ComplexScalar::<f32> {
                    real: 6.0,
                    imag: 6.0,
                },
            ],
        ]),
        &Default::default(),
    );

    let data = tensor.slice([0..1, 1..3]).into_data();

    let expected = TensorData::from([[
        ComplexScalar::<f32> {
            real: 2.0,
            imag: 2.0,
        },
        ComplexScalar::<f32> {
            real: 3.0,
            imag: 3.0,
        },
    ]]);

    data.assert_eq(&expected, false);
}

#[test]
fn test_complex_repeat_dim() {
    let tensor = TestTensor::<2>::from_data(
        TensorData::from([[
            ComplexScalar::<f32> {
                real: 1.0,
                imag: 2.0,
            },
            ComplexScalar::<f32> {
                real: 3.0,
                imag: 4.0,
            },
        ]]),
        &Default::default(),
    );

    let data = tensor.repeat_dim(0, 2).into_data();

    let expected = TensorData::from([
        [
            ComplexScalar::<f32> {
                real: 1.0,
                imag: 2.0,
            },
            ComplexScalar::<f32> {
                real: 3.0,
                imag: 4.0,
            },
        ],
        [
            ComplexScalar::<f32> {
                real: 1.0,
                imag: 2.0,
            },
            ComplexScalar::<f32> {
                real: 3.0,
                imag: 4.0,
            },
        ],
    ]);

    data.assert_eq(&expected, false);
}

#[test]
fn test_complex_cat() {
    let t1 = TestTensor::<2>::from_data(
        TensorData::from([[
            ComplexScalar::<f32> {
                real: 1.0,
                imag: 2.0,
            },
            ComplexScalar::<f32> {
                real: 3.0,
                imag: 4.0,
            },
        ]]),
        &Default::default(),
    );
    let t2 = TestTensor::<2>::from_data(
        TensorData::from([[
            ComplexScalar::<f32> {
                real: 5.0,
                imag: 6.0,
            },
            ComplexScalar::<f32> {
                real: 7.0,
                imag: 8.0,
            },
        ]]),
        &Default::default(),
    );

    let data = TestTensor::<2>::cat(vec![t1, t2], 0).into_data();

    let expected = TensorData::from([
        [
            ComplexScalar::<f32> {
                real: 1.0,
                imag: 2.0,
            },
            ComplexScalar::<f32> {
                real: 3.0,
                imag: 4.0,
            },
        ],
        [
            ComplexScalar::<f32> {
                real: 5.0,
                imag: 6.0,
            },
            ComplexScalar::<f32> {
                real: 7.0,
                imag: 8.0,
            },
        ],
    ]);

    data.assert_eq(&expected, false);
}

#[test]
fn test_complex_equal() {
    let t1 = TestTensor::<1>::from_data(
        TensorData::from([
            ComplexScalar::<f32> {
                real: 1.0,
                imag: 2.0,
            },
            ComplexScalar::<f32> {
                real: 3.0,
                imag: 4.0,
            },
        ]),
        &Default::default(),
    );
    let t2 = TestTensor::<1>::from_data(
        TensorData::from([
            ComplexScalar::<f32> {
                real: 1.0,
                imag: 2.0,
            },
            ComplexScalar::<f32> {
                real: 0.0,
                imag: 0.0,
            },
        ]),
        &Default::default(),
    );

    t1.equal(t2)
        .into_data()
        .assert_eq(&TensorData::from([true, false]), false);
}

#[test]
fn test_complex_not_equal() {
    let t1 = TestTensor::<1>::from_data(
        TensorData::from([
            ComplexScalar::<f32> {
                real: 1.0,
                imag: 2.0,
            },
            ComplexScalar::<f32> {
                real: 3.0,
                imag: 4.0,
            },
        ]),
        &Default::default(),
    );
    let t2 = TestTensor::<1>::from_data(
        TensorData::from([
            ComplexScalar::<f32> {
                real: 1.0,
                imag: 2.0,
            },
            ComplexScalar::<f32> {
                real: 0.0,
                imag: 0.0,
            },
        ]),
        &Default::default(),
    );

    t1.not_equal(t2)
        .into_data()
        .assert_eq(&TensorData::from([false, true]), false);
}

#[test]
fn test_complex_any() {
    TestTensor::<1>::zeros([2], &Default::default())
        .any()
        .into_data()
        .assert_eq(&TensorData::from([false]), false);

    let tensor = TestTensor::<1>::from_data(
        TensorData::from([
            ComplexScalar::<f32> {
                real: 0.0,
                imag: 0.0,
            },
            ComplexScalar::<f32> {
                real: 1.0,
                imag: 0.0,
            },
        ]),
        &Default::default(),
    );
    tensor
        .any()
        .into_data()
        .assert_eq(&TensorData::from([true]), false);
}

#[test]
fn test_complex_any_dim() {
    let tensor = TestTensor::<2>::from_data(
        TensorData::from([
            [
                ComplexScalar::<f32> {
                    real: 0.0,
                    imag: 0.0,
                },
                ComplexScalar::<f32> {
                    real: 1.0,
                    imag: 0.0,
                },
            ],
            [
                ComplexScalar::<f32> {
                    real: 0.0,
                    imag: 0.0,
                },
                ComplexScalar::<f32> {
                    real: 0.0,
                    imag: 0.0,
                },
            ],
        ]),
        &Default::default(),
    );

    // col 0: both zero -> false; col 1: has non-zero -> true
    tensor
        .any_dim(0)
        .into_data()
        .assert_eq(&TensorData::from([[false, true]]), false);
}

#[test]
fn test_complex_all() {
    let tensor = TestTensor::<1>::from_data(
        TensorData::from([
            ComplexScalar::<f32> {
                real: 1.0,
                imag: 0.0,
            },
            ComplexScalar::<f32> {
                real: 0.0,
                imag: 2.0,
            },
        ]),
        &Default::default(),
    );
    tensor
        .all()
        .into_data()
        .assert_eq(&TensorData::from([true]), false);

    let tensor = TestTensor::<1>::from_data(
        TensorData::from([
            ComplexScalar::<f32> {
                real: 1.0,
                imag: 0.0,
            },
            ComplexScalar::<f32> {
                real: 0.0,
                imag: 0.0,
            },
        ]),
        &Default::default(),
    );
    tensor
        .all()
        .into_data()
        .assert_eq(&TensorData::from([false]), false);
}

#[test]
fn test_complex_all_dim() {
    let tensor = TestTensor::<2>::from_data(
        TensorData::from([
            [
                ComplexScalar::<f32> {
                    real: 1.0,
                    imag: 0.0,
                },
                ComplexScalar::<f32> {
                    real: 1.0,
                    imag: 0.0,
                },
            ],
            [
                ComplexScalar::<f32> {
                    real: 0.0,
                    imag: 0.0,
                },
                ComplexScalar::<f32> {
                    real: 1.0,
                    imag: 0.0,
                },
            ],
        ]),
        &Default::default(),
    );

    // col 0: has zero -> false; col 1: all non-zero -> true
    tensor
        .all_dim(0)
        .into_data()
        .assert_eq(&TensorData::from([[false, true]]), false);
}

#[test]
fn test_complex_permute() {
    let tensor = TestTensor::<2>::from_data(
        TensorData::from([
            [
                ComplexScalar::<f32> {
                    real: 1.0,
                    imag: 2.0,
                },
                ComplexScalar::<f32> {
                    real: 3.0,
                    imag: 4.0,
                },
            ],
            [
                ComplexScalar::<f32> {
                    real: 5.0,
                    imag: 6.0,
                },
                ComplexScalar::<f32> {
                    real: 7.0,
                    imag: 8.0,
                },
            ],
        ]),
        &Default::default(),
    );

    // permute([1, 0]) on [2, 2] is equivalent to transpose
    let data = tensor.permute([1, 0]).into_data();

    let expected = TensorData::from([
        [
            ComplexScalar::<f32> {
                real: 1.0,
                imag: 2.0,
            },
            ComplexScalar::<f32> {
                real: 5.0,
                imag: 6.0,
            },
        ],
        [
            ComplexScalar::<f32> {
                real: 3.0,
                imag: 4.0,
            },
            ComplexScalar::<f32> {
                real: 7.0,
                imag: 8.0,
            },
        ],
    ]);

    data.assert_eq(&expected, false);
}

#[test]
fn test_complex_expand() {
    let tensor = TestTensor::<2>::from_data(
        TensorData::from([[
            ComplexScalar::<f32> {
                real: 1.0,
                imag: 2.0,
            },
            ComplexScalar::<f32> {
                real: 3.0,
                imag: 4.0,
            },
        ]]),
        &Default::default(),
    );

    // shape [1, 2] -> expand to [3, 2]
    let data = tensor.expand([3, 2]).into_data();

    let expected = TensorData::from([
        [
            ComplexScalar::<f32> {
                real: 1.0,
                imag: 2.0,
            },
            ComplexScalar::<f32> {
                real: 3.0,
                imag: 4.0,
            },
        ],
        [
            ComplexScalar::<f32> {
                real: 1.0,
                imag: 2.0,
            },
            ComplexScalar::<f32> {
                real: 3.0,
                imag: 4.0,
            },
        ],
        [
            ComplexScalar::<f32> {
                real: 1.0,
                imag: 2.0,
            },
            ComplexScalar::<f32> {
                real: 3.0,
                imag: 4.0,
            },
        ],
    ]);

    data.assert_eq(&expected, false);
}

#[test]
fn test_complex_flip() {
    let tensor = TestTensor::<2>::from_data(
        TensorData::from([
            [
                ComplexScalar::<f32> {
                    real: 1.0,
                    imag: 2.0,
                },
                ComplexScalar::<f32> {
                    real: 3.0,
                    imag: 4.0,
                },
            ],
            [
                ComplexScalar::<f32> {
                    real: 5.0,
                    imag: 6.0,
                },
                ComplexScalar::<f32> {
                    real: 7.0,
                    imag: 8.0,
                },
            ],
        ]),
        &Default::default(),
    );

    let data = tensor.flip([0]).into_data();

    let expected = TensorData::from([
        [
            ComplexScalar::<f32> {
                real: 5.0,
                imag: 6.0,
            },
            ComplexScalar::<f32> {
                real: 7.0,
                imag: 8.0,
            },
        ],
        [
            ComplexScalar::<f32> {
                real: 1.0,
                imag: 2.0,
            },
            ComplexScalar::<f32> {
                real: 3.0,
                imag: 4.0,
            },
        ],
    ]);

    data.assert_eq(&expected, false);
}

#[test]
fn test_complex_unfold() {
    // shape [4] -> unfold(0, size=2, step=1) -> shape [3, 2]
    let tensor = TestTensor::<1>::from_data(
        TensorData::from([
            ComplexScalar::<f32> {
                real: 1.0,
                imag: 0.0,
            },
            ComplexScalar::<f32> {
                real: 2.0,
                imag: 0.0,
            },
            ComplexScalar::<f32> {
                real: 3.0,
                imag: 0.0,
            },
            ComplexScalar::<f32> {
                real: 4.0,
                imag: 0.0,
            },
        ]),
        &Default::default(),
    );

    let result: TestTensor<2> = tensor.unfold(0, 2, 1);
    let data = result.into_data();

    let expected = TensorData::from([
        [
            ComplexScalar::<f32> {
                real: 1.0,
                imag: 0.0,
            },
            ComplexScalar::<f32> {
                real: 2.0,
                imag: 0.0,
            },
        ],
        [
            ComplexScalar::<f32> {
                real: 2.0,
                imag: 0.0,
            },
            ComplexScalar::<f32> {
                real: 3.0,
                imag: 0.0,
            },
        ],
        [
            ComplexScalar::<f32> {
                real: 3.0,
                imag: 0.0,
            },
            ComplexScalar::<f32> {
                real: 4.0,
                imag: 0.0,
            },
        ],
    ]);

    data.assert_eq(&expected, false);
}

#[test]
fn test_complex_slice_assign() {
    let tensor = TestTensor::<2>::from_data(
        TensorData::from([
            [
                ComplexScalar::<f32> {
                    real: 1.0,
                    imag: 1.0,
                },
                ComplexScalar::<f32> {
                    real: 2.0,
                    imag: 2.0,
                },
            ],
            [
                ComplexScalar::<f32> {
                    real: 3.0,
                    imag: 3.0,
                },
                ComplexScalar::<f32> {
                    real: 4.0,
                    imag: 4.0,
                },
            ],
        ]),
        &Default::default(),
    );
    let values = TestTensor::<2>::from_data(
        TensorData::from([[
            ComplexScalar::<f32> {
                real: 9.0,
                imag: 9.0,
            },
            ComplexScalar::<f32> {
                real: 8.0,
                imag: 8.0,
            },
        ]]),
        &Default::default(),
    );

    let data = tensor.slice_assign([0..1, 0..2], values).into_data();

    let expected = TensorData::from([
        [
            ComplexScalar::<f32> {
                real: 9.0,
                imag: 9.0,
            },
            ComplexScalar::<f32> {
                real: 8.0,
                imag: 8.0,
            },
        ],
        [
            ComplexScalar::<f32> {
                real: 3.0,
                imag: 3.0,
            },
            ComplexScalar::<f32> {
                real: 4.0,
                imag: 4.0,
            },
        ],
    ]);

    data.assert_eq(&expected, false);
}

#[test]
fn test_complex_select() {
    let tensor = TestTensor::<2>::from_data(
        TensorData::from([
            [
                ComplexScalar::<f32> {
                    real: 1.0,
                    imag: 2.0,
                },
                ComplexScalar::<f32> {
                    real: 3.0,
                    imag: 4.0,
                },
            ],
            [
                ComplexScalar::<f32> {
                    real: 5.0,
                    imag: 6.0,
                },
                ComplexScalar::<f32> {
                    real: 7.0,
                    imag: 8.0,
                },
            ],
            [
                ComplexScalar::<f32> {
                    real: 9.0,
                    imag: 10.0,
                },
                ComplexScalar::<f32> {
                    real: 11.0,
                    imag: 12.0,
                },
            ],
        ]),
        &Default::default(),
    );

    let indices = Tensor::<1, Int>::from_ints([2, 0], &Default::default());
    let data = tensor.select(0, indices).into_data();

    let expected = TensorData::from([
        [
            ComplexScalar::<f32> {
                real: 9.0,
                imag: 10.0,
            },
            ComplexScalar::<f32> {
                real: 11.0,
                imag: 12.0,
            },
        ],
        [
            ComplexScalar::<f32> {
                real: 1.0,
                imag: 2.0,
            },
            ComplexScalar::<f32> {
                real: 3.0,
                imag: 4.0,
            },
        ],
    ]);

    data.assert_eq(&expected, false);
}

#[test]
fn test_complex_full() {
    // Fills via scalar; the real part is propagated (imag always 0 via ElementConversion).
    let fill = ComplexScalar::<f32> {
        real: 3.0,
        imag: 0.0,
    };
    let tensor = TestTensor::<2>::full([2, 2], fill, &Default::default());
    let data = tensor.into_data();

    let expected = TensorData::from([
        [
            ComplexScalar::<f32> {
                real: 3.0,
                imag: 0.0,
            },
            ComplexScalar::<f32> {
                real: 3.0,
                imag: 0.0,
            },
        ],
        [
            ComplexScalar::<f32> {
                real: 3.0,
                imag: 0.0,
            },
            ComplexScalar::<f32> {
                real: 3.0,
                imag: 0.0,
            },
        ],
    ]);

    data.assert_eq(&expected, false);
}

#[test]
fn test_complex_equal_elem() {
    // The scalar comparison converts through ElementConversion (imag always 0),
    // so this exercises the real-part equality path.
    let tensor = TestTensor::<1>::from_data(
        TensorData::from([
            ComplexScalar::<f32> {
                real: 1.0,
                imag: 0.0,
            },
            ComplexScalar::<f32> {
                real: 3.0,
                imag: 0.0,
            },
            ComplexScalar::<f32> {
                real: 1.0,
                imag: 0.0,
            },
        ]),
        &Default::default(),
    );

    let result = tensor
        .equal_elem(ComplexScalar::<f32> {
            real: 1.0,
            imag: 0.0,
        })
        .into_data();

    result.assert_eq(&TensorData::from([true, false, true]), false);
}

#[test]
fn test_complex_not_equal_elem() {
    // The scalar comparison converts through ElementConversion (imag always 0),
    // so this exercises the real-part inequality path.
    let tensor = TestTensor::<1>::from_data(
        TensorData::from([
            ComplexScalar::<f32> {
                real: 1.0,
                imag: 0.0,
            },
            ComplexScalar::<f32> {
                real: 3.0,
                imag: 0.0,
            },
            ComplexScalar::<f32> {
                real: 1.0,
                imag: 0.0,
            },
        ]),
        &Default::default(),
    );

    let result = tensor
        .not_equal_elem(ComplexScalar::<f32> {
            real: 1.0,
            imag: 0.0,
        })
        .into_data();

    result.assert_eq(&TensorData::from([false, true, false]), false);
}

#[test]
fn test_complex_mask_where() {
    let tensor = TestTensor::<2>::from_data(
        TensorData::from([
            [
                ComplexScalar::<f32> {
                    real: 1.0,
                    imag: 1.0,
                },
                ComplexScalar::<f32> {
                    real: 2.0,
                    imag: 2.0,
                },
            ],
            [
                ComplexScalar::<f32> {
                    real: 3.0,
                    imag: 3.0,
                },
                ComplexScalar::<f32> {
                    real: 4.0,
                    imag: 4.0,
                },
            ],
        ]),
        &Default::default(),
    );
    let source = TestTensor::<2>::from_data(
        TensorData::from([
            [
                ComplexScalar::<f32> {
                    real: 9.0,
                    imag: 9.0,
                },
                ComplexScalar::<f32> {
                    real: 9.0,
                    imag: 9.0,
                },
            ],
            [
                ComplexScalar::<f32> {
                    real: 9.0,
                    imag: 9.0,
                },
                ComplexScalar::<f32> {
                    real: 9.0,
                    imag: 9.0,
                },
            ],
        ]),
        &Default::default(),
    );
    let mask = Tensor::<2, Bool>::from_data([[true, false], [false, true]], &Default::default());

    let data = tensor.mask_where(mask, source).into_data();

    let expected = TensorData::from([
        [
            ComplexScalar::<f32> {
                real: 9.0,
                imag: 9.0,
            },
            ComplexScalar::<f32> {
                real: 2.0,
                imag: 2.0,
            },
        ],
        [
            ComplexScalar::<f32> {
                real: 3.0,
                imag: 3.0,
            },
            ComplexScalar::<f32> {
                real: 9.0,
                imag: 9.0,
            },
        ],
    ]);

    data.assert_eq(&expected, false);
}

#[test]
fn test_complex_mask_fill() {
    let tensor = TestTensor::<2>::from_data(
        TensorData::from([
            [
                ComplexScalar::<f32> {
                    real: 1.0,
                    imag: 1.0,
                },
                ComplexScalar::<f32> {
                    real: 2.0,
                    imag: 2.0,
                },
            ],
            [
                ComplexScalar::<f32> {
                    real: 3.0,
                    imag: 3.0,
                },
                ComplexScalar::<f32> {
                    real: 4.0,
                    imag: 4.0,
                },
            ],
        ]),
        &Default::default(),
    );
    let mask = Tensor::<2, Bool>::from_data([[true, false], [false, true]], &Default::default());

    let data = tensor
        .mask_fill(
            mask,
            ComplexScalar::<f32> {
                real: 0.0,
                imag: 0.0,
            },
        )
        .into_data();

    let expected = TensorData::from([
        [
            ComplexScalar::<f32> {
                real: 0.0,
                imag: 0.0,
            },
            ComplexScalar::<f32> {
                real: 2.0,
                imag: 2.0,
            },
        ],
        [
            ComplexScalar::<f32> {
                real: 3.0,
                imag: 3.0,
            },
            ComplexScalar::<f32> {
                real: 0.0,
                imag: 0.0,
            },
        ],
    ]);

    data.assert_eq(&expected, false);
}

#[test]
fn test_complex_gather() {
    // tensor shape [2, 3]; output[i, j] = tensor[indices[i, j], j] for dim=0
    let tensor = TestTensor::<2>::from_data(
        TensorData::from([
            [
                ComplexScalar::<f32> {
                    real: 1.0,
                    imag: 1.0,
                },
                ComplexScalar::<f32> {
                    real: 2.0,
                    imag: 2.0,
                },
                ComplexScalar::<f32> {
                    real: 3.0,
                    imag: 3.0,
                },
            ],
            [
                ComplexScalar::<f32> {
                    real: 4.0,
                    imag: 4.0,
                },
                ComplexScalar::<f32> {
                    real: 5.0,
                    imag: 5.0,
                },
                ComplexScalar::<f32> {
                    real: 6.0,
                    imag: 6.0,
                },
            ],
        ]),
        &Default::default(),
    );

    let indices = Tensor::<2, Int>::from_ints([[0, 1, 0], [1, 0, 1]], &Default::default());
    let data = tensor.gather(0, indices).into_data();

    let expected = TensorData::from([
        [
            ComplexScalar::<f32> {
                real: 1.0,
                imag: 1.0,
            },
            ComplexScalar::<f32> {
                real: 5.0,
                imag: 5.0,
            },
            ComplexScalar::<f32> {
                real: 3.0,
                imag: 3.0,
            },
        ],
        [
            ComplexScalar::<f32> {
                real: 4.0,
                imag: 4.0,
            },
            ComplexScalar::<f32> {
                real: 2.0,
                imag: 2.0,
            },
            ComplexScalar::<f32> {
                real: 6.0,
                imag: 6.0,
            },
        ],
    ]);

    data.assert_eq(&expected, false);
}

#[test]
fn test_complex_scatter_add() {
    // Start with zeros [2, 2]; scatter-add values along dim 0.
    // output[indices[i, j], j] += values[i, j]
    let tensor = TestTensor::<2>::zeros([2, 2], &Default::default());
    let indices = Tensor::<2, Int>::from_ints([[0, 0], [1, 0]], &Default::default());
    let values = TestTensor::<2>::from_data(
        TensorData::from([
            [
                ComplexScalar::<f32> {
                    real: 1.0,
                    imag: 1.0,
                },
                ComplexScalar::<f32> {
                    real: 2.0,
                    imag: 2.0,
                },
            ],
            [
                ComplexScalar::<f32> {
                    real: 3.0,
                    imag: 3.0,
                },
                ComplexScalar::<f32> {
                    real: 4.0,
                    imag: 4.0,
                },
            ],
        ]),
        &Default::default(),
    );

    let data = tensor
        .scatter(0, indices, values, IndexingUpdateOp::Add)
        .into_data();

    // out[0, 0] += 1+1i; out[0, 1] += 2+2i + 4+4i = 6+6i; out[1, 0] += 3+3i
    let expected = TensorData::from([
        [
            ComplexScalar::<f32> {
                real: 1.0,
                imag: 1.0,
            },
            ComplexScalar::<f32> {
                real: 6.0,
                imag: 6.0,
            },
        ],
        [
            ComplexScalar::<f32> {
                real: 3.0,
                imag: 3.0,
            },
            ComplexScalar::<f32> {
                real: 0.0,
                imag: 0.0,
            },
        ],
    ]);

    data.assert_eq(&expected, false);
}

#[test]
fn test_complex_select_assign_add() {
    // tensor [3, 2]; add values to rows 0 and 2
    let tensor = TestTensor::<2>::from_data(
        TensorData::from([
            [
                ComplexScalar::<f32> {
                    real: 1.0,
                    imag: 1.0,
                },
                ComplexScalar::<f32> {
                    real: 2.0,
                    imag: 2.0,
                },
            ],
            [
                ComplexScalar::<f32> {
                    real: 3.0,
                    imag: 3.0,
                },
                ComplexScalar::<f32> {
                    real: 4.0,
                    imag: 4.0,
                },
            ],
            [
                ComplexScalar::<f32> {
                    real: 5.0,
                    imag: 5.0,
                },
                ComplexScalar::<f32> {
                    real: 6.0,
                    imag: 6.0,
                },
            ],
        ]),
        &Default::default(),
    );
    let indices = Tensor::<1, Int>::from_ints([0, 2], &Default::default());
    let values = TestTensor::<2>::from_data(
        TensorData::from([
            [
                ComplexScalar::<f32> {
                    real: 10.0,
                    imag: 10.0,
                },
                ComplexScalar::<f32> {
                    real: 20.0,
                    imag: 20.0,
                },
            ],
            [
                ComplexScalar::<f32> {
                    real: 30.0,
                    imag: 30.0,
                },
                ComplexScalar::<f32> {
                    real: 40.0,
                    imag: 40.0,
                },
            ],
        ]),
        &Default::default(),
    );

    let data = tensor
        .select_assign(0, indices, values, IndexingUpdateOp::Add)
        .into_data();

    let expected = TensorData::from([
        [
            ComplexScalar::<f32> {
                real: 11.0,
                imag: 11.0,
            },
            ComplexScalar::<f32> {
                real: 22.0,
                imag: 22.0,
            },
        ],
        [
            ComplexScalar::<f32> {
                real: 3.0,
                imag: 3.0,
            },
            ComplexScalar::<f32> {
                real: 4.0,
                imag: 4.0,
            },
        ],
        [
            ComplexScalar::<f32> {
                real: 35.0,
                imag: 35.0,
            },
            ComplexScalar::<f32> {
                real: 46.0,
                imag: 46.0,
            },
        ],
    ]);

    data.assert_eq(&expected, false);
}

#[test]
fn test_complex_scatter_nd() {
    let device = Default::default();

    // data: shape [3, 2], values at indices [0] and [2] are updated
    // data[0, :] = (1+1i, 2+2i), data[1, :] = (3+3i, 4+4i), data[2, :] = (5+5i, 6+6i)
    let data = TestTensor::<2>::from_data(
        TensorData::from([
            [
                ComplexScalar::<f32> {
                    real: 1.0,
                    imag: 1.0,
                },
                ComplexScalar::<f32> {
                    real: 2.0,
                    imag: 2.0,
                },
            ],
            [
                ComplexScalar::<f32> {
                    real: 3.0,
                    imag: 3.0,
                },
                ComplexScalar::<f32> {
                    real: 4.0,
                    imag: 4.0,
                },
            ],
            [
                ComplexScalar::<f32> {
                    real: 5.0,
                    imag: 5.0,
                },
                ComplexScalar::<f32> {
                    real: 6.0,
                    imag: 6.0,
                },
            ],
        ]),
        &device,
    );

    // indices: shape [2, 1] — two index tuples each of depth 1 (selects a row)
    let indices: Tensor<2, Int> = Tensor::from_data(TensorData::from([[0i64], [2i64]]), &device);

    // values: shape [2, 2] matching the selected rows
    let values = TestTensor::<2>::from_data(
        TensorData::from([
            [
                ComplexScalar::<f32> {
                    real: 10.0,
                    imag: 10.0,
                },
                ComplexScalar::<f32> {
                    real: 20.0,
                    imag: 20.0,
                },
            ],
            [
                ComplexScalar::<f32> {
                    real: 30.0,
                    imag: 30.0,
                },
                ComplexScalar::<f32> {
                    real: 40.0,
                    imag: 40.0,
                },
            ],
        ]),
        &device,
    );

    let result = data.scatter_nd::<2, 2>(indices, values, IndexingUpdateOp::Assign);
    let result_data = result.into_data();

    let expected = TensorData::from([
        [
            ComplexScalar::<f32> {
                real: 10.0,
                imag: 10.0,
            },
            ComplexScalar::<f32> {
                real: 20.0,
                imag: 20.0,
            },
        ],
        [
            ComplexScalar::<f32> {
                real: 3.0,
                imag: 3.0,
            },
            ComplexScalar::<f32> {
                real: 4.0,
                imag: 4.0,
            },
        ],
        [
            ComplexScalar::<f32> {
                real: 30.0,
                imag: 30.0,
            },
            ComplexScalar::<f32> {
                real: 40.0,
                imag: 40.0,
            },
        ],
    ]);

    result_data.assert_eq(&expected, false);
}

#[test]
fn test_complex_scatter_nd_add() {
    let device = Default::default();

    // data: shape [2, 3], update specific elements using depth-2 index tuples
    let data = TestTensor::<2>::from_data(
        TensorData::from([
            [
                ComplexScalar::<f32> {
                    real: 1.0,
                    imag: 1.0,
                },
                ComplexScalar::<f32> {
                    real: 2.0,
                    imag: 2.0,
                },
                ComplexScalar::<f32> {
                    real: 3.0,
                    imag: 3.0,
                },
            ],
            [
                ComplexScalar::<f32> {
                    real: 4.0,
                    imag: 4.0,
                },
                ComplexScalar::<f32> {
                    real: 5.0,
                    imag: 5.0,
                },
                ComplexScalar::<f32> {
                    real: 6.0,
                    imag: 6.0,
                },
            ],
        ]),
        &device,
    );

    // indices: shape [2, 2] — two index tuples of depth 2 (selects individual elements)
    let indices: Tensor<2, Int> =
        Tensor::from_data(TensorData::from([[0i64, 1i64], [1i64, 2i64]]), &device);

    // values: shape [2] — one scalar per index tuple
    let values = TestTensor::<1>::from_data(
        TensorData::from([
            ComplexScalar::<f32> {
                real: 10.0,
                imag: 10.0,
            },
            ComplexScalar::<f32> {
                real: 20.0,
                imag: 20.0,
            },
        ]),
        &device,
    );

    let result = data.scatter_nd::<2, 1>(indices, values, IndexingUpdateOp::Add);
    let result_data = result.into_data();

    // data[0,1] += (10+10i), data[1,2] += (20+20i)
    let expected = TensorData::from([
        [
            ComplexScalar::<f32> {
                real: 1.0,
                imag: 1.0,
            },
            ComplexScalar::<f32> {
                real: 12.0,
                imag: 12.0,
            },
            ComplexScalar::<f32> {
                real: 3.0,
                imag: 3.0,
            },
        ],
        [
            ComplexScalar::<f32> {
                real: 4.0,
                imag: 4.0,
            },
            ComplexScalar::<f32> {
                real: 5.0,
                imag: 5.0,
            },
            ComplexScalar::<f32> {
                real: 26.0,
                imag: 26.0,
            },
        ],
    ]);

    result_data.assert_eq(&expected, false);
}

#[test]
fn test_complex_to_device() {
    let device = Default::default();
    let tensor = TestTensor::<2>::from_data(
        TensorData::from([[
            ComplexScalar::<f32> {
                real: 1.0,
                imag: 2.0,
            },
            ComplexScalar::<f32> {
                real: 3.0,
                imag: 4.0,
            },
        ]]),
        &device,
    );
    let original = tensor.clone().into_data();
    let moved = tensor.to_device(&device).into_data();

    moved.assert_eq(&original, false);
}
