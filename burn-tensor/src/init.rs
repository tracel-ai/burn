#[macro_export]
macro_rules! zeros {
    (
        kind: $kind:ident,
        shape: $shape:expr,
        backend: $backend:expr
    ) => {{
        let shape = $crate::Shape::new($shape);
        let data = $crate::Data::zeros_(shape, $kind::default());

        match $backend {
            Backend::Tch(device) => {
                $crate::tensor::backend::tch::TchTensor::from_data(data, device)
            }
        }
    }};

    (
        $kind:ident,
        $shape:expr,
        $backend:expr
    ) => {{
        $crate::zeros!(kind: $kind, shape: $shape, backend: $backend)
    }};

    (
        kind: $kind:ident,
        shape: $shape:expr
    ) => {{
        $crate::zeros!(
            kind: $kind,
            shape: $shape,
            backend: $crate::tensor::backend::Backend::default()
        )
    }};

    (
        $kind:ident,
        $shape:expr
    ) => {{
        $crate::zeros!(kind: $kind, shape: $shape)
    }};

    (
        $shape:expr
    ) => {{
        $crate::zeros!(kind: f32, shape: $shape)
    }};
}

#[macro_export]
macro_rules! ones {
    (
        kind: $kind:ident,
        shape: $shape:expr,
        backend: $backend:expr
    ) => {{
        let shape = $crate::Shape::new($shape);
        let data = $crate::Data::ones_(shape, $kind::default());

        match $backend {
            Backend::Tch(device) => {
                $crate::tensor::backend::tch::TchTensor::from_data(data, device)
            }
        }
    }};

    (
        $kind:ident,
        $shape:expr,
        $backend:expr
    ) => {{
        $crate::ones!(kind: $kind, shape: $shape, backend: $backend)
    }};

    (
        kind: $kind:ident,
        shape: $shape:expr
    ) => {{
        $crate::ones!(
            kind: $kind,
            shape: $shape,
            backend: $crate::tensor::backend::Backend::default()
        )
    }};

    (
        $kind:ident,
        $shape:expr
    ) => {{
        $crate::ones!(kind: $kind, shape: $shape)
    }};

    (
        $shape:expr
    ) => {{
        $crate::ones!(kind: f32, shape: $shape)
    }};
}

#[macro_export]
macro_rules! random {
    (
        kind: $kind:ident,
        shape: $shape:expr,
        backend: $backend:expr,
        distribution: $distribution:expr
    ) => {{
        let shape = $crate::Shape::new($shape);
        let data = $crate::Data::sample_(shape, $distribution, $kind::default());

        match $backend {
            Backend::Tch(device) => {
                $crate::tensor::backend::tch::TchTensor::from_data(data, device)
            }
        }
    }};

    (
        kind: $kind:ident,
        shape: $shape:expr,
        backend: $backend:expr
    ) => {{
        $crate::random!(
            kind: $kind,
            shape: $shape,
            backend: $backend,
            distribution: $crate::Distribution::<$kind>::Standard
        )
    }};

    (
        kind: $kind:ident,
        shape: $shape:expr,
        backend: $backend:expr
    ) => {{
        $crate::random!(kind: $kind, shape: $shape, backend: $backend)
    }};

    (
        $kind:ident,
        $shape:expr,
        $backend:expr
    ) => {{
        $crate::random!(kind: $kind, shape: $shape, backend: $backend)
    }};

    (
        kind: $kind:ident,
        shape: $shape:expr
    ) => {{
        $crate::random!(
            kind: $kind,
            shape: $shape,
            backend: $crate::tensor::backend::Backend::default()
        )
    }};

    (
        $kind:ident,
        $shape:expr
    ) => {{
        $crate::random!(kind: $kind, shape: $shape)
    }};

    (
        $shape:expr
    ) => {{
        $crate::random!(kind: f32, shape: $shape)
    }};
}

#[cfg(test)]
mod tests {
    use crate::{
        backend::{tch::TchTensor, Backend, TchDevice},
        Data, Shape,
    };
    use crate::{random, TensorBase};

    #[test]
    fn random_macro_tch() {
        let shape = Shape::new([2, 3]);
        let data = Data::random(shape.clone());
        let tensor = TchTensor::from_data(data, TchDevice::Cpu);

        let tensor_w_random = random!(
            kind: f32,
            shape: [2, 3],
            backend: Backend::Tch(TchDevice::Cpu)
        );

        assert_ne!(tensor.to_data(), tensor_w_random.to_data());
        assert_eq!(tensor.shape(), tensor_w_random.shape());
    }

    #[test]
    fn random_macro_diffenrent_api() {
        let tensor_1 = random!(
            kind: f32,
            shape: [2, 3],
            backend: Backend::Tch(TchDevice::Cpu)
        );
        let tensor_2 = random!(f32, [2, 3], Backend::Tch(TchDevice::Cpu));
        let tensor_3 = random!(
            kind: f32,
            shape: [2, 3],
            backend: Backend::Tch(TchDevice::Cpu),
            distribution: crate::Distribution::Standard
        );
        let tensor_4 = random!(
            kind: f32,
            shape: [2, 3]
        );
        let tensor_5 = random!(f32, [2, 3]);
        let tensor_6 = random!([2, 3]);

        assert_eq!(tensor_1.shape(), tensor_2.shape());
        assert_eq!(tensor_2.shape(), tensor_3.shape());
        assert_eq!(tensor_3.shape(), tensor_4.shape());
        assert_eq!(tensor_4.shape(), tensor_5.shape());
        assert_eq!(tensor_5.shape(), tensor_6.shape());
    }

    #[test]
    fn ones_macro_diffenrent_api() {
        let tensor_1 = ones!(
            kind: f32,
            shape: [2, 3],
            backend: Backend::Tch(TchDevice::Cpu)
        );
        let tensor_2 = ones!(f32, [2, 3]);
        let tensor_3 = ones!([2, 3]);

        assert_eq!(tensor_1.to_data(), tensor_2.to_data());
        assert_eq!(tensor_1.to_data(), tensor_3.to_data());
    }

    #[test]
    fn zeros_macro_diffenrent_api() {
        let tensor_1 = zeros!(
            kind: f32,
            shape: [2, 3],
            backend: Backend::Tch(TchDevice::Cpu)
        );
        let tensor_2 = zeros!(f32, [2, 3]);
        let tensor_3 = zeros!([2, 3]);

        assert_eq!(tensor_1.to_data(), tensor_2.to_data());
        assert_eq!(tensor_1.to_data(), tensor_3.to_data());
    }
}
