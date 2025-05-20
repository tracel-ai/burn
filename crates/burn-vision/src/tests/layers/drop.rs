#[burn_tensor_testgen::testgen(layers_drop)]
mod tests {
    use super::*;
    use crate::layers::drop::*;

    #[test]
    fn test_drop_path() {
        let device = Default::default();
        let drop_prob = 0.5;
        let scale_by_keep = true;

        let config = crate::layers::drop::DropPathConfig {
            drop_prob,
            scale_by_keep,
        };

        let module = config.init();

        let input = Tensor::<TestBackend, 4>::random(
            [2, 3, 4, 5],
            Distribution::Uniform(0.0, 1.0),
            &device,
        );
        let output = module.forward(input.clone());

        assert_eq!(input.dims(), output.dims());
    }

    #[test]
    fn test_drop_path_sample() {
        let device = Default::default();

        let n = 3;
        let shape = [n, 2, 4];

        let x = Tensor::<TestBackend, 3>::random(shape, Distribution::Uniform(0.0, 1.0), &device);

        /// No-op case: not training and drop_prob = 0.0
        let training = false;
        let drop_prob = 0.0;
        let scale_by_keep = false;
        let res = crate::layers::drop::_drop_path_sample(
            x.clone(),
            drop_prob,
            training,
            scale_by_keep,
            |shape, keep_prob, device| {
                Tensor::<TestBackend, 3>::from_data([[[1.0]], [[0.0]], [[1.0]]], device)
            },
        );
        res.to_data().assert_eq(&x.clone().to_data(), true);

        /// No-op case: training, but drop_prob = 0.0
        let training = true;
        let drop_prob = 0.0;
        let scale_by_keep = false;
        let res = crate::layers::drop::_drop_path_sample(
            x.clone(),
            drop_prob,
            training,
            scale_by_keep,
            |shape, keep_prob, device| {
                Tensor::<TestBackend, 3>::from_data([[[1.0]], [[0.0]], [[1.0]]], device)
            },
        );
        res.to_data().assert_eq(&x.clone().to_data(), true);

        /// Training, but no scaling
        let training = true;
        let drop_prob = 0.5;
        let scale_by_keep = false;
        let res = crate::layers::drop::_drop_path_sample(
            x.clone(),
            drop_prob,
            training,
            scale_by_keep,
            |shape, keep_prob, device| {
                Tensor::<TestBackend, 3>::from_data([[[1.0]], [[0.0]], [[1.0]]], device)
            },
        );
        res.to_data().assert_eq(
            &(x.clone()
                * Tensor::<TestBackend, 3>::from_data([[[1.0]], [[0.0]], [[1.0]]], &device))
            .to_data(),
            true,
        );

        /// Training, with scaling
        let training = true;
        let drop_prob = 0.5;
        let keep_prob = 1.0 - drop_prob;
        let scale_by_keep = true;
        let res = crate::layers::drop::_drop_path_sample(
            x.clone(),
            drop_prob,
            training,
            scale_by_keep,
            |shape, keep_prob, device| {
                Tensor::<TestBackend, 3>::from_data([[[1.0]], [[0.0]], [[1.0]]], device)
            },
        );
        res.to_data().assert_eq(
            &(x.clone()
                * Tensor::<TestBackend, 3>::from_data([[[1.0]], [[0.0]], [[1.0]]], &device))
            .div_scalar(keep_prob)
            .to_data(),
            true,
        );
    }

    #[test]
    fn test_droppath_module() {
        let drop_prob = 0.2;
        let config = crate::layers::drop::DropPathConfig::new().with_drop_prob(drop_prob);

        assert_eq!(config.drop_prob(), 0.2);
        assert_eq!(config.keep_prob(), 1.0 - drop_prob);
        assert!(config.scale_by_keep());

        let module = config.init();
        assert_eq!(module.drop_prob(), 0.2);
        assert_eq!(module.keep_prob(), 1.0 - drop_prob);
        assert!(module.scale_by_keep());

        let device = Default::default();
        let shape = [2, 3, 4];
        let x = Tensor::<TestBackend, 3>::random(shape, Distribution::Uniform(0.0, 1.0), &device);

        // TODO(crutcher): work out how to enable/disable training mode in tests.
        let output = module.forward(x.clone());
        assert_eq!(x.dims(), output.dims());
    }
}
