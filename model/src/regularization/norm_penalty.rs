// builtin
use std::{
    cell::{Ref, RefCell, RefMut},
    rc::Rc,
};

// external
use serde::{Deserialize, Serialize};

// internal
use crate::{node::NodeRef, regularization::norm_penalty::no_penalty::builder::NullBuilder};
pub mod l2_penalty;
pub mod no_penalty;

#[derive(Serialize, Deserialize)]
pub enum NormPenaltyType {
    L2 { alpha: f32 },
    None,
}

pub type NormPenaltyRef<'a> = Rc<RefCell<dyn NormPenaltyUnit<'a> + 'a>>;

pub struct NormPenaltyConfig<'a> {
    builder: Box<dyn NormPenaltyBuilder<'a> + 'a>,
}

impl<'a> NormPenaltyConfig<'a> {
    pub fn new(builder: impl NormPenaltyBuilder<'a> + 'a) -> NormPenaltyConfig<'a> {
        NormPenaltyConfig {
            builder: Box::new(builder),
        }
    }

    pub fn none() -> NormPenaltyConfig<'a> {
        NormPenaltyConfig {
            builder: Box::new(NullBuilder),
        }
    }

    pub fn create_first(&self, parameter_node: &NodeRef<'a>) -> NormPenaltyContainer<'a> {
        self.builder.create_first(parameter_node)
    }

    pub fn create_new(
        &self,
        prev: &NormPenaltyRef<'a>,
        parameter_node: &NodeRef<'a>,
    ) -> NormPenaltyContainer<'a> {
        self.builder.create_new(prev, parameter_node)
    }

    pub fn get_builder(&self) -> &Box<dyn NormPenaltyBuilder<'a> + 'a> {
        &self.builder
    }
}

pub struct NormPenaltyContainer<'a> {
    unit: Rc<RefCell<dyn NormPenaltyUnit<'a> + 'a>>,
}

impl<'a> NormPenaltyContainer<'a> {
    pub fn new(penalty_unit: impl NormPenaltyUnit<'a> + 'a) -> NormPenaltyContainer<'a> {
        NormPenaltyContainer {
            unit: Rc::new(RefCell::new(penalty_unit)),
        }
    }

    pub fn borrow(&self) -> Ref<dyn NormPenaltyUnit<'a> + 'a> {
        self.unit.borrow()
    }

    pub fn borrow_mut(&self) -> RefMut<dyn NormPenaltyUnit<'a> + 'a> {
        self.unit.borrow_mut()
    }

    pub fn get_ref(&self) -> NormPenaltyRef<'a> {
        Rc::clone(&self.unit)
    }
}

pub trait NormPenaltyUnit<'a> {
    fn add_penalty_input(&mut self, input: &NormPenaltyRef<'a>);

    fn add_parameter_input(&mut self, parameter_node: &NodeRef<'a>);

    fn get_output_ref(&self) -> &NodeRef<'a>;

    fn is_null(&self) -> bool;
}

pub trait NormPenaltyBuilder<'a> {
    fn create_first(&self, parameter_node: &NodeRef<'a>) -> NormPenaltyContainer<'a>;

    fn create_new(
        &self,
        prev: &NormPenaltyRef<'a>,
        parameter_node: &NodeRef<'a>,
    ) -> NormPenaltyContainer<'a>;

    fn get_associated_type(&self) -> NormPenaltyType;
}
