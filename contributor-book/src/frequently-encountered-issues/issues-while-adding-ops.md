# Issues encountered while adding ops

Below are some of the issues that were encountered while adding ops to the project. If you encounter
an issue while adding an op that isn't listed here, and it's not obvious how to fix it, you can add
it to this list or reach out on the [Discord server](https://discord.gg/uPEBbYYDB6) if you need
help.

## Off by .000001 errors

```sh
---- fusion::base::tests::maxmin::tests::test_mean_dim_2d stdout ---- thread 'fusion::base::tests::maxmin::tests::test_mean_dim_2d' panicked at burn-wgpu/src/fusion/base.rs:185:5: assertion `left == right` failed left: Data { value: [1.0, 4.0], shape: Shape { dims: [2, 1] } } right: Data { value: [0.99999994, 3.9999998], shape: Shape { dims: [2, 1] } } ----

tests::maxmin::tests::test_mean_dim_2d stdout ---- thread 'tests::maxmin::tests::test_mean_dim_2d' panicked at burn-wgpu/src/lib.rs:49:5: assertion `left == right` failed left: Data { value: [1.0, 4.0], shape: Shape { dims: [2, 1] } } right: Data { value: [0.99999994, 3.9999998], shape: Shape { dims: [2, 1] } }
```

If you encounter this, swap out the `assert_eq!` in the failing test for
`tensor1.to_data().assert_approx_eq` with `3` as the second argument. The second arguments specifies
the level of precision: `3` is equivalent to a less than 10<sup>-3</sup> (0.001) difference between
the elements of the two tensors.
