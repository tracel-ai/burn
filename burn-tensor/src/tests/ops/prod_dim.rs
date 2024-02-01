#[burn_tensor_testgen::testgen(prod_dim)]
mod tests {
    use super::*;
    use burn_tensor::{Data, Tensor};

    #[test]
    fn prod_R1() {
        let t = Tensor::<TestBackend, 1>::from_data(Data::from([3.0, 4.0]), &Default::default());

        let t_prod = t.prod();

        t_prod.to_data().assert_approx_eq(&Data::from([12.0]), 3);
    }

    #[test]
    fn prod_dim_R2() {
        let t = Tensor::<TestBackend, 2>::from_data(
            Data::from([[9.0, 10.0], [54.0, 61.0]]),
            &Default::default(),
        );

        let t_prod = t.prod_dim(1);

        t_prod
            .to_data()
            .assert_approx_eq(&Data::from([[90.0], [3294.0]]), 3);
    }
}
