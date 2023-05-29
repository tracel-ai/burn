use burn_tensor::{
    backend::Backend,
    ops::{ActivationOps, BoolTensorOps, ModuleOps, TensorOps},
    ElementConversion,
};

use crate::{
    element::{FloatElement, IntElement},
    GraphicsAPI, WGPUBackend, WGPUDevice,
};

impl<G, F, I> ActivationOps<WGPUBackend<G, F, I>> for WGPUBackend<G, F, I>
where
    G: GraphicsAPI + 'static,
    F: FloatElement,
    I: IntElement,
{
    fn relu<const D: usize>(
        tensor: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D>,
    ) -> <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D> {
        let mask = <WGPUBackend<G, F, I>>::lower_equal_elem(tensor.clone(), 0.elem());

        <WGPUBackend<G, F, I>>::mask_fill(tensor, mask, 0.elem())
    }

    fn relu_backward<const D: usize>(
        output: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D>,
        grad: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D>,
    ) -> <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D> {
        let mask = <WGPUBackend<G, F, I>>::lower_equal_elem(output, 0.elem());

        <WGPUBackend<G, F, I>>::mask_fill(grad, mask, 0.elem())
    }

    fn gelu<const D: usize>(
        tensor: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D>,
    ) -> <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D> {
        let x = <WGPUBackend<G, F, I>>::div_scalar(tensor.clone(), std::f64::consts::SQRT_2.elem());
        let x = <WGPUBackend<G, F, I>>::erf(x);
        let x = <WGPUBackend<G, F, I>>::add_scalar(x, 1i32.elem());
        let x = <WGPUBackend<G, F, I>>::mul(tensor, x);

        <WGPUBackend<G, F, I>>::div_scalar(x, 2i32.elem())
    }

    fn gelu_backward<const D: usize>(
        x: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D>,
        grad: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D>,
    ) -> <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D> {
        // Derivative of the approximate gelu implementation based on tanh.

        let constant_1 = 0.0356774;
        let constant_2 = 0.797885;
        let constant_3 = 0.0535161;
        let constant_4 = 0.398942;

        let x3 = <WGPUBackend<G, F, I>>::powf(x.clone(), 3.0);

        let c1 = <WGPUBackend<G, F, I>>::mul_scalar(x3.clone(), constant_1.elem());
        let c2 = <WGPUBackend<G, F, I>>::mul_scalar(x.clone(), constant_2.elem());
        let c3 = <WGPUBackend<G, F, I>>::mul_scalar(x3, constant_3.elem());
        let c4 = <WGPUBackend<G, F, I>>::mul_scalar(x, constant_4.elem());

        let inner1 = <WGPUBackend<G, F, I>>::add(c1, c2);
        let inner2 = <WGPUBackend<G, F, I>>::add(c3, c4);

        let tanh = <WGPUBackend<G, F, I>>::tanh(inner1);

        let sech = <WGPUBackend<G, F, I>>::powf(tanh.clone(), 2.0);
        let sech = <WGPUBackend<G, F, I>>::neg(sech);
        let sech = <WGPUBackend<G, F, I>>::add_scalar(sech, 1.elem());

        let y1 = <WGPUBackend<G, F, I>>::mul_scalar(tanh, 0.5.elem());
        let y2 = <WGPUBackend<G, F, I>>::mul(inner2, sech);
        let y2 = <WGPUBackend<G, F, I>>::add_scalar(y2, 0.5.elem());
        let y = <WGPUBackend<G, F, I>>::add(y1, y2);

        <WGPUBackend<G, F, I>>::mul(y, grad)
    }
}
