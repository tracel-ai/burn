use super::{
    backend::{autodiff::ADTensor, tch::TchTensor},
    ops::*,
    Data, Distribution, Element, Shape, TensorTrait,
};

pub type Tensor<const D: usize, B> = <B as TensorType<D, B>>::T;

pub trait Backend:
    Sized
    + Default
    + Send
    + Sync
    + std::fmt::Debug
    + TensorType<1, Self>
    + TensorType<2, Self>
    + TensorType<3, Self>
    + TensorType<4, Self>
    + TensorType<5, Self>
    + TensorType<6, Self>
{
    type E: Element;
    type Device: Default + Send + Sync + std::fmt::Debug + Clone + Copy;

    fn name() -> String;
}

pub trait TensorType<const D: usize, B: Backend>
where
    B: TensorType<D, B>,
{
    type T: TensorTrait<B::E, D>
        // + TensorOpsDevice<B::E, D, B>
        + TensorCreationLike<B::E, D>
        + TensorCreationFork<B::E, D, 1, Output = Tensor<1, B>>
        + TensorCreationFork<B::E, D, 2, Output = Tensor<2, B>>
        + TensorCreationFork<B::E, D, 3, Output = Tensor<3, B>>
        + TensorCreationFork<B::E, D, 4, Output = Tensor<4, B>>
        + TensorCreationFork<B::E, D, 5, Output = Tensor<5, B>>
        + TensorCreationFork<B::E, D, 6, Output = Tensor<6, B>>
        + TensorOpsIndex<B::E, D, 1>
        + TensorOpsIndex<B::E, D, 2>
        + TensorOpsIndex<B::E, D, 3>
        + TensorOpsIndex<B::E, D, 4>
        + TensorOpsIndex<B::E, D, 5>
        + TensorOpsIndex<B::E, D, 6>;
    // + TensorOpsReshape<B::E, D, 1, Output = Tensor<1, B>>
    // + TensorOpsReshape<B::E, 1, D, Output = Tensor<D, B>>;
    // + TensorOpsReshape<B::E, 3, D, Tensor<D, B>>
    // + TensorOpsReshape<B::E, 4, D, Tensor<D, B>>
    // + TensorOpsReshape<B::E, 5, D, Tensor<D, B>>
    // + TensorOpsReshape<B::E, 6, D, Tensor<D, B>>
    //+ TensorOpsReshape<B::E, D, 1, Tensor<1, B>>
    //+ TensorOpsReshape<B::E, D, 2, Tensor<2, B>>
    //+ TensorOpsReshape<B::E, D, 3, Tensor<3, B>>
    //+ TensorOpsReshape<B::E, D, 4, Tensor<4, B>>
    //+ TensorOpsReshape<B::E, D, 5, Tensor<5, B>>
    //+ TensorOpsReshape<B::E, D, 6, Tensor<6, B>>;

    fn from_data(data: Data<B::E, D>, device: B::Device) -> Self::T;
}

pub trait Backend2:
    Sized
    + Default
    + TensorType2<1, Self>
    + TensorType2<2, Self>
    + TensorType2<3, Self>
    + TensorType2<4, Self>
    + TensorType2<5, Self>
    + TensorType2<6, Self>
{
    type Device;
    type Elem: Element;
}

pub trait TensorType2<const D: usize, B: Backend2>
where
    B: TensorType2<D, B>,
{
    type T: TensorTrait<B::Elem, D>;
}

#[derive(Default)]
pub struct ADBackend2<B: Backend2> {
    _b: B,
}

#[derive(Default)]
pub struct TchBackend2<E: Default> {
    _e: E,
}

impl<E: Element> Backend2 for TchBackend2<E> {
    type Device = tch::Device;

    type Elem = E;
}

impl<E: Element, const D: usize> TensorType2<D, Self> for TchBackend2<E> {
    type T = TchTensor<E, D>;
}

impl<B: Backend2> Backend2 for ADBackend2<B> {
    type Device = tch::Device;

    type Elem = B::Elem;
}

impl<B: Backend2, const D: usize> TensorType2<D, Self> for ADBackend2<B>
where
    B: TensorType2<D, B>,
{
    type T = ADTensor<B::Elem, D, TensorOps<D, B>>;
}

pub type TensorOps<const D: usize, B> = <B as TensorType2<D, B>>::T;

pub struct Tensor2<const D: usize, B: Backend2 + TensorType2<D, B>>
where
    B: Backend2 + TensorType2<D, B>,
{
    value: TensorOps<D, B>,
}

impl<const D: usize, B> Tensor2<D, B>
where
    B: Backend2 + TensorType2<D, B>,
    TensorOps<D, B>: TensorTrait<B::Elem, D>,
{
    pub fn new(tensor: TensorOps<D, B>) -> Self {
        Self { value: tensor }
    }
}

impl<const D: usize, B> Tensor2<D, B>
where
    B: Backend2 + TensorType2<D, B>,
    TensorOps<D, B>: TensorTrait<B::Elem, D>,
{
    pub fn reshape<const D2: usize>(&self, shape: Shape<D2>) -> Tensor2<D2, B>
    where
        B: TensorType2<D2, B>,
        TensorOps<D, B>: TensorOpsReshape<B::Elem, D, D2, TensorOps<D2, B>>,
    {
        Tensor2::new(self.value.reshape(shape))
    }

    pub fn shape(&self) -> &Shape<D> {
        self.value.shape()
    }

    pub fn into_data(self) -> Data<B::Elem, D> {
        self.value.into_data()
    }

    pub fn to_data(&self) -> Data<B::Elem, D> {
        self.value.to_data()
    }

    pub fn new_like_empty(&self) -> Self
    where
        TensorOps<D, B>: TensorCreationLike<B::Elem, D>,
    {
        Self::new(self.value.new_like_empty())
    }

    pub fn new_like_random(&self, distribution: Distribution<B::Elem>) -> Self
    where
        TensorOps<D, B>: TensorCreationLike<B::Elem, D>,
    {
        Self::new(self.value.new_like_random(distribution))
    }

    pub fn new_like_data(&self, data: Data<B::Elem, D>) -> Self
    where
        TensorOps<D, B>: TensorCreationLike<B::Elem, D>,
    {
        Self::new(self.value.new_like_data(data))
    }

    pub fn new_like_zeros(&self) -> Self
    where
        TensorOps<D, B>: TensorCreationLike<B::Elem, D>,
    {
        Self::new(self.value.new_like_zeros())
    }

    pub fn new_like_ones(&self) -> Self
    where
        TensorOps<D, B>: TensorCreationLike<B::Elem, D>,
    {
        Self::new(self.value.new_like_ones())
    }

    pub fn new_fork_empty<const D2: usize>(&self, shape: Shape<D2>) -> Tensor2<D2, B>
    where
        B: TensorType2<D2, B>,
        TensorOps<D, B>: TensorCreationFork<B::Elem, D, D2, Output = TensorOps<D2, B>>,
    {
        Tensor2::new(self.value.new_fork_empty(shape))
    }

    pub fn new_fork_random<const D2: usize>(
        &self,
        shape: Shape<D2>,
        distribution: Distribution<B::Elem>,
    ) -> Tensor2<D2, B>
    where
        B: TensorType2<D2, B>,
        TensorOps<D, B>: TensorCreationFork<B::Elem, D, D2, Output = TensorOps<D2, B>>,
    {
        Tensor2::new(self.value.new_fork_random(shape, distribution))
    }

    pub fn new_fork_data<const D2: usize>(&self, data: Data<B::Elem, D2>) -> Tensor2<D2, B>
    where
        B: TensorType2<D2, B>,
        TensorOps<D, B>: TensorCreationFork<B::Elem, D, D2, Output = TensorOps<D2, B>>,
    {
        Tensor2::new(self.value.new_fork_data(data))
    }

    pub fn new_fork_zeros<const D2: usize>(&self, shape: Shape<D2>) -> Tensor2<D2, B>
    where
        B: TensorType2<D2, B>,
        TensorOps<D, B>: TensorCreationFork<B::Elem, D, D2, Output = TensorOps<D2, B>>,
    {
        Tensor2::new(self.value.new_fork_zeros(shape))
    }

    pub fn new_fork_ones<const D2: usize>(&self, shape: Shape<D2>) -> Tensor2<D2, B>
    where
        B: TensorType2<D2, B>,
        TensorOps<D, B>: TensorCreationFork<B::Elem, D, D2, Output = TensorOps<D2, B>>,
    {
        Tensor2::new(self.value.new_fork_ones(shape))
    }
    pub fn add(&self, other: &Self) -> Self {
        Self::new(self.value.add(&other.value))
    }

    pub fn add_scalar(&self, other: &B::Elem) -> Self {
        Self::new(self.value.add_scalar(&other))
    }

    pub fn sub(&self, other: &Self) -> Self {
        Self::new(self.value.sub(&other.value))
    }

    pub fn sub_scalar(&self, other: &B::Elem) -> Self {
        Self::new(self.value.sub_scalar(&other))
    }

    pub fn transpose(&self) -> Self {
        Self::new(self.value.transpose())
    }

    pub fn matmul(&self, other: &Self) -> Self {
        Self::new(self.value.matmul(&other.value))
    }

    pub fn neg(&self) -> Self {
        Self::new(self.value.neg())
    }

    pub fn mul(&self, other: &Self) -> Self {
        Self::new(self.value.mul(&other.value))
    }

    pub fn mul_scalar(&self, other: &B::Elem) -> Self {
        Self::new(self.value.mul_scalar(&other))
    }

    pub fn index<const D2: usize>(&self, indexes: [std::ops::Range<usize>; D2]) -> Self
    where
        TensorOps<D, B>: TensorOpsIndex<B::Elem, D, D2>,
    {
        Self::new(self.value.index(indexes))
    }

    pub fn index_assign<const D2: usize>(
        &self,
        indexes: [std::ops::Range<usize>; D2],
        values: &Self,
    ) -> Self
    where
        TensorOps<D, B>: TensorOpsIndex<B::Elem, D, D2>,
    {
        Self::new(self.value.index_assign(indexes, &values.value))
    }

    pub fn track_grad(&self) -> Tensor2<D, ADBackend2<B>> {
        Tensor2::new(ADTensor::from_tensor(self.value.clone()))
    }
}
