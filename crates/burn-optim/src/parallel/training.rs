use burn_core as burn;

use alloc::{boxed::Box, collections::VecDeque, vec::Vec};
use core::any::Any;
use std::{
    panic,
    sync::mpsc::{self, Receiver, Sender},
    thread::{self, ScopedJoinHandle},
};

use burn::module::Module;
use burn::module::parallel::{
    BackwardSignal, DistributedLayeredModel, HeldStage, HiddenSignal, InferencePipeline, InputPart,
    InputSignal, LayerParallelism, LayerPlacement, LayerSequence, MiddlePart, OutputPart,
    OutputSignal, PipelineParts, PipelineSchedule, PipelineShape, StagePass,
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
    HiddenSignal<M>: BackwardSignal,
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
        let mut links = Links::new(shape.num_stages - 1);
        let devices = signal_devices(&self.parts);

        let gradients = thread::scope(|scope| {
            let input = scope.spawn({
                let worker = InputWorker {
                    part: self.parts.input.clone(),
                    to_next: links.to_next.next().unwrap(),
                    from_next: links.from_next.next().unwrap(),
                    next_device: devices[1].clone(),
                };
                let passes = &passes[0];
                move || worker.part.stream.executes(|| worker.run(passes, inputs))
            });
            let middle: Vec<_> = self
                .parts
                .middle
                .iter()
                .zip(&passes[1..])
                .zip(devices.windows(3))
                .map(|((part, passes), devices)| {
                    let worker = MiddleWorker {
                        part: part.clone(),
                        from_previous: links.from_previous.next().unwrap(),
                        to_next: links.to_next.next().unwrap(),
                        to_previous: links.to_previous.next().unwrap(),
                        from_next: links.from_next.next().unwrap(),
                        previous_device: devices[0].clone(),
                        next_device: devices[2].clone(),
                    };
                    scope.spawn(move || worker.part.stream.executes(|| worker.run(passes)))
                })
                .collect();
            let output = scope.spawn({
                let worker = OutputWorker {
                    part: self.parts.output.clone(),
                    from_previous: links.from_previous.next().unwrap(),
                    to_previous: links.to_previous.next().unwrap(),
                    previous_device: devices[shape.num_stages - 2].clone(),
                };
                let passes = &passes[shape.num_stages - 1];
                let loss = &loss;
                move || {
                    worker
                        .part
                        .stream
                        .executes(|| worker.run(passes, targets, loss, scale))
                }
            });
            let mut joined = Joined::default();
            let input = joined.join(input);
            let middle: Vec<_> = middle.into_iter().map(|part| joined.join(part)).collect();
            let output = joined.join(output);
            let gradients = || {
                Some(StepGradients {
                    input: input?,
                    middle: middle.into_iter().collect::<Option<_>>()?,
                    output: output?,
                })
            };
            gradients().unwrap_or_else(|| joined.resume())
        });

        self.apply(gradients, learning_rate)
    }

    /// Run each of `batches` through the model as it stands, every part on a thread of its own,
    /// recording no graph: an evaluation between steps.
    pub fn forward(&self, batches: Vec<InputSignal<M>>) -> Vec<OutputSignal<M>> {
        InferencePipeline::from_parts(self.parts.valid()).forward(batches)
    }

    /// The model, its layers as training left them.
    pub fn into_model(self) -> DistributedLayeredModel<M> {
        let layers = self.parts.into_layers();
        let model = M::from_layers(LayerSequence {
            input: layers.input,
            hidden: layers
                .stages
                .into_iter()
                .flat_map(|stage| stage.layers)
                .collect(),
            output: layers.output,
        });
        DistributedLayeredModel::new(model, &self.placement)
    }

    /// Each part's layers stepped with the gradients its thread accumulated; the losses back.
    fn apply(&mut self, gradients: StepGradients, learning_rate: LearningRate) -> Vec<Tensor<1>> {
        let optimizer = &mut self.optimizer;
        let parts = &mut self.parts;

        let input = &mut parts.input;
        input.layer = optimizer.step(learning_rate, input.layer.clone(), gradients.input.layer);
        if let (Some(stage), Some(grads)) = (&mut input.stage, gradients.input.stage) {
            stage.layers = optimizer.step(learning_rate, stage.layers.clone(), grads);
        }
        for (part, grads) in parts.middle.iter_mut().zip(gradients.middle) {
            let stage = &mut part.stage;
            stage.layers = optimizer.step(learning_rate, stage.layers.clone(), grads);
        }
        let output = &mut parts.output;
        if let (Some(stage), Some(grads)) = (&mut output.stage, gradients.output.stage) {
            stage.layers = optimizer.step(learning_rate, stage.layers.clone(), grads);
        }
        output.layer = optimizer.step(learning_rate, output.layer.clone(), gradients.output.layer);
        gradients.output.losses
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

/// The channels between consecutive parts, handed out in part order: activations go forward
/// and their gradients come back. Unbounded, since a part that blocked on a send could deadlock
/// the schedule; the schedule bounds how many wait.
struct Links<S> {
    to_next: alloc::vec::IntoIter<Sender<S>>,
    from_previous: alloc::vec::IntoIter<Receiver<S>>,
    to_previous: alloc::vec::IntoIter<Sender<S>>,
    from_next: alloc::vec::IntoIter<Receiver<S>>,
}

impl<S> Links<S> {
    fn new(boundaries: usize) -> Self {
        let (to_next, from_previous) = (0..boundaries)
            .map(|_| mpsc::channel())
            .unzip::<_, _, Vec<_>, Vec<_>>();
        let (to_previous, from_next) = (0..boundaries)
            .map(|_| mpsc::channel())
            .unzip::<_, _, Vec<_>, Vec<_>>();
        Self {
            to_next: to_next.into_iter(),
            from_previous: from_previous.into_iter(),
            to_previous: to_previous.into_iter(),
            from_next: from_next.into_iter(),
        }
    }
}

/// What every part's thread accumulated over one step.
struct StepGradients {
    input: OuterGradients,
    middle: Vec<GradientsParams>,
    output: OutputGradients,
}

/// An outer part's gradients: its layer's, and its stage's when it holds one.
struct OuterGradients {
    layer: GradientsParams,
    stage: Option<GradientsParams>,
}

/// The output part's gradients, with each microbatch's loss.
struct OutputGradients {
    layer: GradientsParams,
    stage: Option<GradientsParams>,
    losses: Vec<Tensor<1>>,
}

/// The input part: microbatches in, activations to the next part, gradients back from it.
struct InputWorker<M: LayerParallelism> {
    part: InputPart<M>,
    to_next: Sender<HiddenSignal<M>>,
    from_next: Receiver<HiddenSignal<M>>,
    next_device: Device,
}

impl<M: LayerParallelism> InputWorker<M>
where
    HiddenSignal<M>: BackwardSignal,
{
    fn run(self, passes: &[StagePass], inputs: Vec<InputSignal<M>>) -> OuterGradients {
        let mut inputs = inputs.into_iter();
        let mut in_flight = VecDeque::new();
        let mut layer = GradientsAccumulator::new();
        let mut stage = GradientsAccumulator::new();
        for pass in passes {
            match pass {
                StagePass::Forward { .. } => {
                    let output = self.part.forward(inputs.next().unwrap());
                    send(&self.to_next, output.clone(), &self.next_device);
                    in_flight.push_back(output);
                }
                StagePass::Backward { .. } => {
                    let output: HiddenSignal<M> = in_flight.pop_front().unwrap();
                    let grad = receive(&self.from_next);
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
    from_previous: Receiver<HiddenSignal<M>>,
    to_next: Sender<HiddenSignal<M>>,
    to_previous: Sender<HiddenSignal<M>>,
    from_next: Receiver<HiddenSignal<M>>,
    previous_device: Device,
    next_device: Device,
}

impl<M: LayerParallelism> MiddleWorker<M>
where
    HiddenSignal<M>: BackwardSignal,
{
    fn run(self, passes: &[StagePass]) -> GradientsParams {
        let mut in_flight = VecDeque::new();
        let mut layers = GradientsAccumulator::new();
        for pass in passes {
            match pass {
                StagePass::Forward { .. } => {
                    let stage = &self.part.stage;
                    let input = receive(&self.from_previous).detach_require_grad();
                    let output = stage.forward(input.clone());
                    send(&self.to_next, output.clone(), &self.next_device);
                    in_flight.push_back(Boundary { input, output });
                }
                StagePass::Backward { .. } => {
                    let Boundary { input, output } = in_flight.pop_front().unwrap();
                    let grad = receive(&self.from_next);
                    let mut grads = backward_from(output, grad);
                    send(&self.to_previous, input.grad(&grads), &self.previous_device);
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
    from_previous: Receiver<HiddenSignal<M>>,
    to_previous: Sender<HiddenSignal<M>>,
    previous_device: Device,
}

impl<M: LayerParallelism> OutputWorker<M>
where
    HiddenSignal<M>: BackwardSignal,
{
    fn run<T>(
        self,
        passes: &[StagePass],
        targets: Vec<T>,
        loss: &impl Fn(OutputSignal<M>, T) -> Tensor<1>,
        scale: f32,
    ) -> OutputGradients {
        let mut targets = targets.into_iter();
        let mut in_flight = VecDeque::new();
        let mut losses = Vec::new();
        let mut layer = GradientsAccumulator::new();
        let mut stage = GradientsAccumulator::new();
        for pass in passes {
            match pass {
                StagePass::Forward { .. } => {
                    let input = receive(&self.from_previous).detach_require_grad();
                    let value = loss(self.part.forward(input.clone()), targets.next().unwrap());
                    losses.push(value.clone().detach());
                    in_flight.push_back(Scored {
                        input,
                        loss: value * scale,
                    });
                }
                StagePass::Backward { .. } => {
                    let Scored { input, loss } = in_flight.pop_front().unwrap();
                    let mut grads = loss.backward();
                    send(&self.to_previous, input.grad(&grads), &self.previous_device);
                    if let Some(held) = &self.part.stage {
                        accumulate(&mut stage, &held.layers, &mut grads);
                    }
                    accumulate(&mut layer, &self.part.layer, &mut grads);
                }
            }
        }
        OutputGradients {
            layer: layer.grads(),
            stage: self.part.stage.as_ref().map(|_| stage.grads()),
            losses,
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
fn backward_from<S: BackwardSignal>(output: S, grad: S) -> Gradients {
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

/// `signal` moved to the receiving part's `device`, then sent. The sender moves it so the copy
/// queues before the work it issues next; moved by the receiver, it would wait behind that work.
///
/// A send to a part that has already stopped, which only a panic there does, is dropped; the step
/// re-raises that panic once every part has stopped.
fn send<S: Module>(to: &Sender<S>, signal: S, device: &Device) {
    let _ = to.send(signal.to_device(device));
}

/// What the neighbour sends next. A neighbour that stopped first has panicked, so this part stops
/// too, quietly, leaving that panic as the step's.
fn receive<S>(from: &Receiver<S>) -> S {
    from.recv()
        .unwrap_or_else(|_| panic::resume_unwind(Box::new(NeighbourStopped)))
}
