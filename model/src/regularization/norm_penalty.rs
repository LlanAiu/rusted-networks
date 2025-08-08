// builtin
use std::{
    cell::{Ref, RefCell, RefMut},
    marker::PhantomData,
    rc::Rc,
};

// external

// internal
use crate::node::NodeRef;
pub mod l2_penalty;

pub type NormPenaltyRef<'a> = Rc<RefCell<dyn NormPenaltyUnit<'a> + 'a>>;

pub struct NormPenaltyConfig<'a, T, C>
where
    C: NormPenaltyUnit<'a>,
    T: NormPenaltyBuilder<'a, C>,
{
    builder: Box<T>,
    _marker: PhantomData<&'a C>,
}

impl<'a, T, C> NormPenaltyConfig<'a, T, C>
where
    C: NormPenaltyUnit<'a>,
    T: NormPenaltyBuilder<'a, C>,
{
    pub fn new(builder: Box<T>) -> NormPenaltyConfig<'a, T, C> {
        NormPenaltyConfig {
            builder,
            _marker: PhantomData,
        }
    }

    pub fn create_first(&self, weights: &NodeRef<'a>) -> NormPenaltyContainer<'a, C> {
        self.builder.create_first(weights)
    }

    pub fn create_new(
        &self,
        prev: &NormPenaltyRef<'a>,
        weights: &NodeRef<'a>,
    ) -> NormPenaltyContainer<'a, C> {
        self.builder.create_new(prev, weights)
    }

    pub fn create_last(
        &self,
        weights: &NodeRef<'a>,
        prev: &NormPenaltyRef<'a>,
        loss_sum_node: &NodeRef<'a>,
    ) -> NormPenaltyContainer<'a, C> {
        self.builder.create_last(weights, prev, loss_sum_node)
    }
}

pub struct NormPenaltyContainer<'a, T>
where
    T: NormPenaltyUnit<'a>,
{
    unit: Rc<RefCell<T>>,
    _marker: PhantomData<&'a T>,
}

impl<'a, T> NormPenaltyContainer<'a, T>
where
    T: NormPenaltyUnit<'a>,
{
    pub fn new(penalty_unit: T) -> NormPenaltyContainer<'a, T> {
        NormPenaltyContainer {
            unit: Rc::new(RefCell::new(penalty_unit)),
            _marker: PhantomData,
        }
    }

    pub fn borrow(&self) -> Ref<T> {
        self.unit.borrow()
    }

    pub fn borrow_mut(&self) -> RefMut<T> {
        self.unit.borrow_mut()
    }

    pub fn get_ref(&self) -> NormPenaltyRef<'a> {
        Rc::clone(&self.unit) as Rc<RefCell<dyn NormPenaltyUnit<'a>>>
    }
}

pub trait NormPenaltyUnit<'a> {
    fn add_penalty_input(&mut self, input: &NormPenaltyRef<'a>);

    fn add_weight_input(&mut self, weight_node: &NodeRef<'a>);

    fn get_output_ref(&self) -> &NodeRef<'a>;
}

pub trait NormPenaltyBuilder<'a, T>
where
    T: NormPenaltyUnit<'a>,
{
    fn create_first(&self, weights: &NodeRef<'a>) -> NormPenaltyContainer<'a, T>;

    fn create_new(
        &self,
        prev: &NormPenaltyRef<'a>,
        weights: &NodeRef<'a>,
    ) -> NormPenaltyContainer<'a, T>;

    fn create_last(
        &self,
        weights: &NodeRef<'a>,
        prev: &NormPenaltyRef<'a>,
        loss_sum_node: &NodeRef<'a>,
    ) -> NormPenaltyContainer<'a, T>;
}
