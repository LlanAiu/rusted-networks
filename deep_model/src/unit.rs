// builtin
use std::{
    cell::{Ref, RefCell, RefMut},
    marker::PhantomData,
    rc::Rc,
};

// external

// internal
use crate::node::NodeRef;
pub mod types;
pub mod unit_base;

pub type UnitRef<'a> = Rc<RefCell<dyn Unit<'a> + 'a>>;

pub trait Unit<'a> {
    fn add_input(&mut self, this: &UnitRef<'a>, input: &UnitRef<'a>);

    fn add_output(&mut self, output: &UnitRef<'a>);

    fn get_inputs(&self) -> &Vec<UnitRef<'a>>;

    fn get_outputs(&self) -> &Vec<UnitRef<'a>>;

    fn get_output_node(&self) -> &NodeRef<'a>;
}

pub struct UnitContainer<'a, T: Unit<'a> + ?Sized> {
    unit: Rc<RefCell<T>>,
    _marker: PhantomData<&'a T>,
}

impl<'a, T> UnitContainer<'a, T>
where
    T: Unit<'a>,
{
    pub fn new(unit: T) -> UnitContainer<'a, T> {
        UnitContainer {
            unit: Rc::new(RefCell::new(unit)),
            _marker: PhantomData,
        }
    }

    pub fn borrow(&self) -> Ref<'_, T> {
        self.unit.borrow()
    }

    pub fn borrow_mut(&self) -> RefMut<'_, T> {
        self.unit.borrow_mut()
    }

    pub fn add_input<C: Unit<'a>>(&self, input: &UnitContainer<'a, C>) {
        let this_ref = self.get_ref();
        let input_ref = input.get_ref();

        this_ref.borrow_mut().add_input(&this_ref, &input_ref);
    }

    pub fn get_ref(&self) -> UnitRef<'a> {
        Rc::clone(&self.unit) as Rc<RefCell<dyn Unit<'a> + 'a>>
    }
}
