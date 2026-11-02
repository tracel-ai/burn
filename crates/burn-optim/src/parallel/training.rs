use burn_core as burn;

use alloc::{boxed::Box, collections::VecDeque, vec::Vec};
use core::any::Any;
use std::{
    mem, panic,
    sync::mpsc::{self, Receiver, Sender},
    thread::{self, Scope, ScopedJoinHandle},
};

use burn::module::Module;
use burn::module::parallel::{
    AutodiffSignal, DistributedLayeredModel, HeldStage, HiddenSignal, InferencePipeline, InputPart,
    InputSignal, LayerParallelism, LayerPlacement, MiddlePart, OutputPart, OutputSignal,
    PipelineParts, PipelineSchedule, PipelineShape, StagePass,
};
use burn::tensor::{Device, Gradients, Tensor};

use crate::{GradientsAccumulator, GradientsParams, LearningRate, ModuleOptimizer};

/// A [`DistributedLayeredModel`] trained as a pipeline: each step cuts a batch into microbatches,
/// and every part of the split runs their forward and backward passes on a thread of its own, in
/// the order its [`PipelineSchedule`] gives.
///
/// Each microbatch's loss is scaled by one over their number, so a step updates the model as one
/// step on the whole batch does when the microbatches hold as many rows each and the loss is a
/// mean over them.
///
/// ```rust,ignore
/// let schedule = PipelineSchedule::OneForwardOneBackward;
/// let mut training = TrainingPipeline::new(model, schedule, AdamConfig::new().init());
/// let losses = training.step(inputs, targets, |output, target| loss(output, target), lr);
/// let model = training.into_model();
/// ```
pub struct TrainingPipeline<M: LayerParallelism> {
    parts: PipelineParts<M>,
    placement: LayerPlacement,
    schedule: PipelineSchedule,
    optimizer: ModuleOptimizer,
}

impl<M: LayerParallelism> TrainingPipeline<M>
where
    HiddenSignal<M>: AutodiffSignal,
{
    /// `model`, built on autodiff devices, trained in `schedule`'s order and updated by
    /// `optimizer`.
    pub fn new(
        model: DistributedLayeredModel<M>,
        schedule: PipelineSchedule,
        optimizer: ModuleOptimizer,
    ) -> Self {
        let placement = model.placement().clone();
        Self {
            parts: PipelineParts::new(model.into_stages(), &placement),
            placement,
            schedule,
            optimizer,
        }
    }

    /// One training step over a batch cut into `inputs`, each with the target at its index in
    /// `targets`: every microbatch's forward and backward, then one optimizer step at
    /// `learning_rate`. Returns each microbatch's loss as `loss` computed it, before scaling.
    ///
    /// # Panics
    ///
    /// Panics when `inputs` is empty or `targets` has another length, and with the panic of
    /// `loss` or a layer when one panics.
    pub fn step<T: Send>(
        &mut self,
        inputs: Vec<InputSignal<M>>,
        targets: Vec<T>,
        loss: impl Fn(OutputSignal<M>, T) -> Tensor<1> + Sync,
        learning_rate: LearningRate,
    ) -> Vec<Tensor<1>> {
        assert!(!inputs.is_empty(), "a step needs at least one microbatch");
        assert_eq!(
            inputs.len(),
            targets.len(),
            "every microbatch needs a target"
        );
        let shape = PipelineShape {
            num_stages: self.parts.middle.len() + 2,
            num_microbatches: inputs.len(),
        };
        let passes = self.schedule.passes(shape);
        let scale = 1.0 / shape.num_microbatches as f32;
        let devices = signal_devices(&self.parts);
        let mut links = devices.windows(2).map(|pair| Link::new(&pair[0], &pair[1]));
        let mut losses = Vec::with_capacity(shape.num_microbatches);

        let gradients = thread::scope(|scope| {
            let Link { next, mut previous } = links
                .next()
                .expect("a pipeline has at least two parts, so one link");
            let input = InputWorker {
                part: self.parts.input.clone(),
                next,
            }
            .spawn(scope, &passes[0], inputs);
            let middle = self
                .parts
                .middle
                .iter()
                .zip(&passes[1..])
                .map(|(part, passes)| {
                    let link = links
                        .next()
                        .expect("every part before the output part links to the next");
                    MiddleWorker {
                        part: part.clone(),
                        previous: mem::replace(&mut previous, link.previous),
                        next: link.next,
                    }
                    .spawn(scope, passes)
                })
                .collect();
            let output = OutputWorker {
                part: self.parts.output.clone(),
                previous,
                scale,
            }
            .spawn(
                scope,
                &passes[shape.num_stages - 1],
                targets,
                &loss,
                &mut losses,
            );
            StepGradients::join(input, middle, output)
        });

        self.apply(gradients, learning_rate);
        losses
    }

    /// Run each of `batches` through the model as it stands, every part on a thread of its own,
    /// recording no graph: an evaluation between steps.
    pub fn forward(&self, batches: Vec<InputSignal<M>>) -> Vec<OutputSignal<M>> {
        InferencePipeline::from_parts(self.parts.valid()).forward(batches)
    }

    /// The model, its layers as training left them. The optimizer's state is not kept, so
    /// training resumed from this model starts it afresh.
    pub fn into_model(self) -> DistributedLayeredModel<M> {
        DistributedLayeredModel::new(M::from_layers(self.parts.layers()), &self.placement)
    }

    /// Steps the whole model at once: stepped apart, a part's parameter paths would be relative
    /// to it, and a parameter group selected by path would not match them.
    fn apply(&mut self, gradients: StepGradients, learning_rate: LearningRate) {
        let model = M::from_layers(self.parts.layers());
        let mut merged = GradientsAccumulator::new();
        for grads in gradients.into_params() {
            merged.accumulate(&model, grads);
        }
        let model = self.optimizer.step(learning_rate, model, merged.grads());
        self.parts.set_layers(model.into_layers());
    }
}

/// A step's threads joined, keeping the panic that stopped the step: once one part panics, its
/// neighbours stop with [`NeighbourStopped`].
#[derive(Default)]
struct Joined {
    panic: Option<Box<dyn Any + Send>>,
}

impl Joined {
    fn join<T>(&mut self, handle: ScopedJoinHandle<'_, T>) -> Option<T> {
        handle
            .join()
            .map_err(|panic| {
                if self
                    .panic
                    .as_ref()
                    .is_none_or(|kept| kept.is::<NeighbourStopped>())
                {
                    self.panic = Some(panic);
                }
            })
            .ok()
    }

    fn resume(self) -> ! {
        panic::resume_unwind(
            self.panic
                .expect("a part that returned nothing has panicked"),
        )
    }
}

/// How a part stops when a neighbour it waits on has stopped first.
struct NeighbourStopped;

/// The two channels between consecutive parts, seen from each end: activations go forward and
/// their gradients come back. Unbounded, since a part that blocked on a send could deadlock the
/// schedule; the schedule bounds how many wait.
struct Link<S> {
    /// The later part, as the earlier one sees it.
    next: Neighbour<S>,
    /// The earlier part, as the later one sees it.
    previous: Neighbour<S>,
}

impl<S> Link<S> {
    /// The link between a part taking its signal on `earlier` and the next, on `later`.
    fn new(earlier: &Device, later: &Device) -> Self {
        let (forward, forwarded) = mpsc::channel();
        let (back, returned) = mpsc::channel();
        Self {
            next: Neighbour {
                to: forward,
                from: returned,
                device: later.clone(),
            },
            previous: Neighbour {
                to: back,
                from: forwarded,
                device: earlier.clone(),
            },
        }
    }
}

/// A neighbouring part: where this part sends it signals, where it receives the ones it sends
/// back, and the device it takes them on.
struct Neighbour<S> {
    to: Sender<S>,
    from: Receiver<S>,
    device: Device,
}

impl<S: Module> Neighbour<S> {
    /// Moved to the neighbour's device by the sender, so the copy queues before the work this
    /// part issues next. A send to a stopped part is dropped; only a panic stops one, and the step
    /// re-raises it.
    fn send(&self, signal: S) {
        let _ = self.to.send(signal.to_device(&self.device));
    }

    /// What the neighbour sends next. A neighbour that stopped first has panicked, so this part
    /// stops too, quietly, leaving that panic as the step's.
    fn receive(&self) -> S {
        self.from
            .recv()
            .unwrap_or_else(|_| panic::resume_unwind(Box::new(NeighbourStopped)))
    }
}

/// What every part's thread accumulated over one step.
struct StepGradients {
    input: OuterGradients,
    middle: Vec<GradientsParams>,
    output: OuterGradients,
}

impl StepGradients {
    /// Every part's gradients, each keyed by its parameters' ids.
    fn into_params(self) -> impl Iterator<Item = GradientsParams> {
        [self.input.layer]
            .into_iter()
            .chain(self.input.stage)
            .chain(self.middle)
            .chain(self.output.stage)
            .chain([self.output.layer])
    }

    /// Every part's thread joined, re-raising the panic that stopped the step if one did.
    fn join(
        input: ScopedJoinHandle<'_, OuterGradients>,
        middle: Vec<ScopedJoinHandle<'_, GradientsParams>>,
        output: ScopedJoinHandle<'_, OuterGradients>,
    ) -> Self {
        let mut joined = Joined::default();
        let input = joined.join(input);
        let middle: Vec<_> = middle.into_iter().map(|part| joined.join(part)).collect();
        let output = joined.join(output);
        let gradients = || {
            Some(Self {
                input: input?,
                middle: middle.into_iter().collect::<Option<_>>()?,
                output: output?,
            })
        };
        gradients().unwrap_or_else(|| joined.resume())
    }
}

/// An outer part's gradients: its layer's, and its stage's when it holds one.
struct OuterGradients {
    layer: GradientsParams,
    stage: Option<GradientsParams>,
}

/// The input part: microbatches in, activations to the next part, gradients back from it.
struct InputWorker<M: LayerParallelism> {
    part: InputPart<M>,
    next: Neighbour<HiddenSignal<M>>,
}

impl<M: LayerParallelism> InputWorker<M>
where
    HiddenSignal<M>: AutodiffSignal,
{
    /// `run` on a thread of its own and on the part's stream.
    fn spawn<'scope>(
        self,
        scope: &'scope Scope<'scope, '_>,
        passes: &'scope [StagePass],
        inputs: Vec<InputSignal<M>>,
    ) -> ScopedJoinHandle<'scope, OuterGradients>
    where
        M: 'scope,
    {
        scope.spawn(move || self.part.stream.executes(|| self.run(passes, inputs)))
    }

    fn run(self, passes: &[StagePass], inputs: Vec<InputSignal<M>>) -> OuterGradients {
        let mut inputs = inputs.into_iter();
        let mut in_flight = VecDeque::new();
        let mut layer = GradientsAccumulator::new();
        let mut stage = GradientsAccumulator::new();
        for pass in passes {
            match pass {
                StagePass::Forward { .. } => {
                    let output = self.part.forward(
                        inputs
                            .next()
                            .expect("the schedule runs one forward per microbatch"),
                    );
                    self.next.send(output.clone());
                    in_flight.push_back(output);
                }
                StagePass::Backward { .. } => {
                    let output: HiddenSignal<M> = in_flight
                        .pop_front()
                        .expect("the schedule runs each backward after its forward");
                    let grad = self.next.receive();
                    let mut grads = backward_from(output, grad);
                    accumulate(&mut layer, &self.part.layer, &mut grads);
                    if let Some(held) = &self.part.stage {
                        accumulate(&mut stage, &held.layers, &mut grads);
                    }
                }
            }
        }
        OuterGradients {
            layer: layer.grads(),
            stage: self.part.stage.as_ref().map(|_| stage.grads()),
        }
    }
}

/// A stage between the outer parts: activations from the previous part and to the next, and
/// their gradients the other way.
struct MiddleWorker<M: LayerParallelism> {
    part: MiddlePart<M>,
    previous: Neighbour<HiddenSignal<M>>,
    next: Neighbour<HiddenSignal<M>>,
}

impl<M: LayerParallelism> MiddleWorker<M>
where
    HiddenSignal<M>: AutodiffSignal,
{
    /// `run` on a thread of its own and on the part's stream.
    fn spawn<'scope>(
        self,
        scope: &'scope Scope<'scope, '_>,
        passes: &'scope [StagePass],
    ) -> ScopedJoinHandle<'scope, GradientsParams>
    where
        M: 'scope,
    {
        scope.spawn(move || self.part.stream.executes(|| self.run(passes)))
    }

    fn run(self, passes: &[StagePass]) -> GradientsParams {
        let mut in_flight = VecDeque::new();
        let mut layers = GradientsAccumulator::new();
        for pass in passes {
            match pass {
                StagePass::Forward { .. } => {
                    let stage = &self.part.stage;
                    let input = self.previous.receive().detach_require_grad();
                    let output = stage.forward(input.clone());
                    self.next.send(output.clone());
                    in_flight.push_back(Boundary { input, output });
                }
                StagePass::Backward { .. } => {
                    let Boundary { input, output } = in_flight
                        .pop_front()
                        .expect("the schedule runs each backward after its forward");
                    let grad = self.next.receive();
                    let mut grads = backward_from(output, grad);
                    self.previous.send(input.grad(&grads));
                    accumulate(&mut layers, &self.part.stage.layers, &mut grads);
                }
            }
        }
        layers.grads()
    }
}

/// The output part: activations from the previous part, the loss, and their gradients back.
struct OutputWorker<M: LayerParallelism> {
    part: OutputPart<M>,
    previous: Neighbour<HiddenSignal<M>>,
    /// One over the number of microbatches, which each loss is scaled by.
    scale: f32,
}

impl<M: LayerParallelism> OutputWorker<M>
where
    HiddenSignal<M>: AutodiffSignal,
{
    /// `run` on a thread of its own and on the part's stream.
    fn spawn<'scope, T: Send + 'scope>(
        self,
        scope: &'scope Scope<'scope, '_>,
        passes: &'scope [StagePass],
        targets: Vec<T>,
        loss: &'scope (impl Fn(OutputSignal<M>, T) -> Tensor<1> + Sync),
        losses: &'scope mut Vec<Tensor<1>>,
    ) -> ScopedJoinHandle<'scope, OuterGradients>
    where
        M: 'scope,
    {
        scope.spawn(move || {
            self.part
                .stream
                .executes(|| self.run(passes, targets, loss, losses))
        })
    }

    /// Each microbatch's loss, before scaling, goes in `losses`.
    fn run<T>(
        self,
        passes: &[StagePass],
        targets: Vec<T>,
        loss: &impl Fn(OutputSignal<M>, T) -> Tensor<1>,
        losses: &mut Vec<Tensor<1>>,
    ) -> OuterGradients {
        let mut targets = targets.into_iter();
        let mut in_flight = VecDeque::new();
        let mut layer = GradientsAccumulator::new();
        let mut stage = GradientsAccumulator::new();
        for pass in passes {
            match pass {
                StagePass::Forward { .. } => {
                    let input = self.previous.receive().detach_require_grad();
                    let target = targets
                        .next()
                        .expect("the schedule runs one forward per microbatch");
                    let value = loss(self.part.forward(input.clone()), target);
                    losses.push(value.clone().detach());
                    in_flight.push_back(Scored {
                        input,
                        loss: value * self.scale,
                    });
                }
                StagePass::Backward { .. } => {
                    let Scored { input, loss } = in_flight
                        .pop_front()
                        .expect("the schedule runs each backward after its forward");
                    let mut grads = loss.backward();
                    self.previous.send(input.grad(&grads));
                    if let Some(held) = &self.part.stage {
                        accumulate(&mut stage, &held.layers, &mut grads);
                    }
                    accumulate(&mut layer, &self.part.layer, &mut grads);
                }
            }
        }
        OuterGradients {
            layer: layer.grads(),
            stage: self.part.stage.as_ref().map(|_| stage.grads()),
        }
    }
}

/// What a stage keeps of a microbatch between its forward and its backward.
struct Boundary<S> {
    input: S,
    output: S,
}

/// What the output part keeps of a microbatch between its forward and its backward.
struct Scored<S> {
    input: S,
    loss: Tensor<1>,
}

/// The backward of the part that produced `output`, given the gradient the next part sent back.
fn backward_from<S: AutodiffSignal>(output: S, grad: S) -> Gradients {
    output
        .grad_product(grad)
        .expect("the signal between stages carries no float tensor to train through")
        .backward()
}

fn accumulate<T: Module>(
    accumulator: &mut GradientsAccumulator<T>,
    module: &T,
    grads: &mut Gradients,
) {
    accumulator.accumulate(module, GradientsParams::from_module(grads, module));
}

/// Where each part, in forward order, takes and gives the signal between parts: an outer part
/// on its stage's device when it holds one, on its own otherwise.
fn signal_devices<M: LayerParallelism>(parts: &PipelineParts<M>) -> Vec<Device> {
    let outer = |stage: &Option<HeldStage<M>>, device: &Device| {
        stage.as_ref().map_or(device, |stage| &stage.device).clone()
    };
    core::iter::once(outer(&parts.input.stage, &parts.input.device))
        .chain(parts.middle.iter().map(|part| part.stage.device.clone()))
        .chain(core::iter::once(outer(
            &parts.output.stage,
            &parts.output.device,
        )))
        .collect()
}
