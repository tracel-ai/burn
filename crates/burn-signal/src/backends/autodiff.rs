use crate::SignalOps;
use alloc::{vec, vec::Vec};
use burn_autodiff::{
    Autodiff,
    checkpoint::{base::Checkpointer, strategy::CheckpointStrategy},
    grads::Gradients,
    ops::{Backward, Ops, OpsKind, unary},
};
use burn_core::backend::{Backend, TensorMetadata, ops::FloatTensorOps, tensor::FloatTensor};
use burn_std::Slice;

impl<B: Backend + SignalOps, C: CheckpointStrategy> SignalOps for Autodiff<B, C> {
    fn rfft(
        signal: FloatTensor<Autodiff<B, C>>,
        dim: usize,
        n: Option<usize>,
    ) -> (FloatTensor<Autodiff<B, C>>, FloatTensor<Autodiff<B, C>>) {
        #[derive(Debug)]
        struct Rfft;

        impl<B: Backend + SignalOps> Backward<B, 1> for Rfft {
            type State = (usize, Option<usize>, usize, usize, Vec<Slice>, Vec<Slice>);

            fn backward(
                self,
                ops: Ops<Self::State, 1>,
                grads: &mut Gradients,
                _checkpointer: &mut Checkpointer,
            ) {
                unary::<B, _>(ops.parents, ops.node, grads, |grad| {
                    let (dim, n, input_len, n_fft, slices_re, slices_im) = ops.state;

                    let grad_re = B::float_slice(grad.clone(), &slices_re);
                    let grad_im = B::float_slice(grad.clone(), &slices_im);

                    let grad_re = mul_interior::<B>(grad_re, dim, n_fft, 0.5);
                    let grad_im = mul_interior::<B>(grad_im, dim, n_fft, 0.5);

                    let grad = B::irfft(grad_re, grad_im, dim, n);
                    let grad = B::float_mul_scalar(grad, (n_fft as f64).into());

                    pad_to_length::<B>(grad, dim, input_len)
                });
            }
        }

        let input_len = signal.shape()[dim];
        let n_fft = n.unwrap_or(input_len);
        let signal_guard = signal.node();
        let (re, im) = B::rfft(signal.into_primitive(), dim, n);

        // In order to perform only a single irfft in the backward pass, we have to temporarily
        // bundle `re` and `im` into a single tensor. The following slice vecs are used to split
        // them back up in both the forward and the backward pass.
        let slices_re = re
            .shape()
            .iter()
            .map(|&len| Slice::from(0..len))
            .collect::<Vec<Slice>>();
        let slices_im = {
            let mut slices = slices_re.clone();
            let len = slices[0].end.unwrap();
            slices[0].start = len;
            slices[0].end = Some(2 * len);
            slices
        };
        let spectrum = B::float_cat(vec![re, im], 0);
        let state = (
            dim,
            n,
            input_len,
            n_fft,
            slices_re.clone(),
            slices_im.clone(),
        );

        let spectrum = match Rfft.prepare::<C>([signal_guard]).compute_bound().stateful() {
            OpsKind::Tracked(prep) => prep.finish(state, spectrum),
            OpsKind::UnTracked(prep) => prep.finish(spectrum),
        };

        let re = Self::float_slice(spectrum.clone(), &slices_re);
        let im = Self::float_slice(spectrum.clone(), &slices_im);

        (re, im)
    }

    fn irfft(
        spectrum_re: FloatTensor<Autodiff<B, C>>,
        spectrum_im: FloatTensor<Autodiff<B, C>>,
        dim: usize,
        n: Option<usize>,
    ) -> FloatTensor<Autodiff<B, C>> {
        #[derive(Debug)]
        struct Irfft;

        impl<B: Backend + SignalOps> Backward<B, 2> for Irfft {
            type State = (usize, Option<usize>, usize, usize);

            fn backward(
                self,
                ops: Ops<Self::State, 2>,
                grads: &mut Gradients,
                _checkpointer: &mut Checkpointer,
            ) {
                let (dim, n, input_len, n_fft) = ops.state;
                let [node_re, node_im] = ops.parents;
                let grad = grads.consume::<B>(&ops.node);

                let grad = B::float_div_scalar(grad, (n_fft as f64).into());

                let (grad_re, grad_im) = B::rfft(grad, dim, n);

                let grad_re = mul_interior::<B>(grad_re, dim, n_fft, 2.0);
                let grad_im = mul_interior::<B>(grad_im, dim, n_fft, 2.0);

                let grad_re = pad_to_length::<B>(grad_re, dim, input_len);
                let grad_im = pad_to_length::<B>(grad_im, dim, input_len);

                if let Some(node) = node_re {
                    grads.register::<B>(node.id, grad_re);
                }
                if let Some(node) = node_im {
                    grads.register::<B>(node.id, grad_im);
                }
            }
        }

        let input_len = spectrum_re.shape()[dim];
        let input_guards = [spectrum_re.node(), spectrum_im.node()];
        let signal = B::irfft(
            spectrum_re.into_primitive(),
            spectrum_im.into_primitive(),
            dim,
            n,
        );
        let n_fft = n.unwrap_or(signal.shape()[dim]);
        let state = (dim, n, input_len, n_fft);

        match Irfft.prepare::<C>(input_guards).compute_bound().stateful() {
            OpsKind::Tracked(prep) => prep.finish(state, signal),
            OpsKind::UnTracked(prep) => prep.finish(signal),
        }
    }
}

// adapted from: burn_cubecl::kernel::fft::base::pad_to_length
fn pad_to_length<B: Backend>(tensor: FloatTensor<B>, dim: usize, target: usize) -> FloatTensor<B> {
    let shape = tensor.shape();
    let current = shape[dim];
    if current == target {
        return tensor;
    }
    if current > target {
        let slices: Vec<_> = shape
            .iter()
            .enumerate()
            .map(|(i, &s)| Slice::from(if i == dim { 0..target } else { 0..s }))
            .collect();
        return B::float_slice(tensor, &slices);
    }
    let mut padded_shape = shape.clone();
    padded_shape[dim] = target;
    let padded = B::float_zeros(padded_shape, &tensor.device(), tensor.dtype().into());
    let slices: Vec<Slice> = shape.iter().map(|&s| Slice::from(0..s)).collect();
    B::float_slice_assign(padded, &slices, tensor)
}

fn mul_interior<B: Backend>(
    bins: FloatTensor<B>,
    dim: usize,
    n_fft: usize,
    factor: f64,
) -> FloatTensor<B> {
    // identify the interior bins (all bins except DC and Nyquist)
    let slices_interior: Vec<Slice> = {
        let mut ranges = bins.shape().into_ranges();

        // skip the DC bin
        ranges[dim].start += 1;

        // if `n_fft` is even, we have a Nyquist bin to skip
        if n_fft.is_multiple_of(2) {
            ranges[dim].end -= 1;
        }

        // Length-one and length-two transforms have no interior bins. Avoid an
        // empty backend slice, which some kernels cannot represent.
        if ranges[dim].start >= ranges[dim].end {
            return bins;
        }

        ranges.into_iter().map(Slice::from).collect()
    };

    // multiply only the interior bins by `factor`
    let interior = B::float_slice(bins.clone(), &slices_interior);
    let interior = B::float_mul_scalar(interior, factor.into());

    B::float_slice_assign(bins, &slices_interior, interior)
}
