use super::GradAdaptor;
use crate::{
    GradientsParams,
    grad_clipping::{L2Norm, norm_precision, scale_gradient},
};
use alloc::vec::Vec;
use burn_core::{
    module::{Module, ModuleVisitor, Param, ParamId},
    tensor::{DType, Device, Tensor},
};
use hashbrown::HashMap;

struct NormCollector {
    source: GradAdaptor,
    gradients: GradientsParams,
    norms: Vec<(Device, Vec<L2Norm>)>,
    original_dtypes: HashMap<ParamId, DType>,
}

impl ModuleVisitor for NormCollector {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
        // Removing by ID counts shared parameters once.
        let entry = match &mut self.source {
            GradAdaptor::Single(grads) => grads.remove::<D>(param.id).map(|g| {
                let device = g.device();
                let dtype = g.dtype();
                (g, device, dtype)
            }),
            GradAdaptor::Multi(grads) => {
                grads.remove_with(param.id, |a, b| norm_precision(a) + norm_precision(b))
            }
        };
        let Some((grad, device, dtype)) = entry else {
            return;
        };
        if grad.dtype() != dtype {
            self.original_dtypes.insert(param.id, dtype);
        }
        let norm = L2Norm::of(grad.clone());
        if let Some((_, norms)) = self.norms.iter_mut().find(|(d, _)| *d == device) {
            norms.push(norm);
        } else {
            self.norms.push((device, alloc::vec![norm]));
        }
        self.gradients.register(param.id, grad);
    }
}

pub(super) struct GlobalNorm {
    scales: Vec<(Device, Tensor<1>)>,
    original_dtypes: HashMap<ParamId, DType>,
}

impl GlobalNorm {
    pub(super) fn prepare<M: Module>(
        module: &M,
        source: GradAdaptor,
        limit: f32,
    ) -> (GradientsParams, Option<Self>) {
        let mut collector = NormCollector {
            source,
            gradients: GradientsParams::new(),
            norms: Vec::new(),
            original_dtypes: HashMap::new(),
        };
        module.visit(&mut collector);
        let norm = Self::from_device_norms(collector.norms, collector.original_dtypes, limit);
        (collector.gradients, norm)
    }

    fn from_device_norms(
        norms: Vec<(Device, Vec<L2Norm>)>,
        original_dtypes: HashMap<ParamId, DType>,
        limit: f32,
    ) -> Option<Self> {
        let local: Vec<_> = norms
            .into_iter()
            .map(|(device, norms)| (device, L2Norm::combine(norms)))
            .collect();
        // Reduce where the widest participating dtype is supported.
        let root = local
            .iter()
            .find(|(_, n)| n.dtype() == DType::F64)
            .or_else(|| local.first())?
            .0
            .clone();
        let (devices, partials): (Vec<_>, _) = local
            .into_iter()
            .map(|(device, norm)| {
                let dtype = norm.dtype();
                let norm = if device == root {
                    norm
                } else {
                    norm.into_device(&root)
                };
                ((device, dtype), norm)
            })
            .unzip();
        let sqrt_scale = L2Norm::combine(partials).sqrt_scale(limit);
        let scales = devices
            .into_iter()
            .map(|(device, dtype)| {
                let scale = if device == root {
                    sqrt_scale.clone()
                } else {
                    sqrt_scale.clone().cast(dtype).to_device(&device)
                };
                (device, scale)
            })
            .collect();
        Some(Self {
            scales,
            original_dtypes,
        })
    }

    pub(super) fn clip<const D: usize>(
        &self,
        id: ParamId,
        grad: Tensor<D>,
        device: &Device,
    ) -> Tensor<D> {
        let (_, sqrt_scale) = self
            .scales
            .iter()
            .find(|(d, _)| d == device)
            .expect("gradient device contributed a norm");
        let dtype = self
            .original_dtypes
            .get(&id)
            .copied()
            .unwrap_or_else(|| grad.dtype());
        scale_gradient(grad, sqrt_scale.clone(), dtype)
    }
}
