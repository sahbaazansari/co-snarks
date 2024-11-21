use std::marker::PhantomData;

use acvm::brillig_vm::MEMORY_ADDRESSING_BIT_SIZE;
use ark_ff::PrimeField;
use brillig::{BitSize, IntegerBitSize};

use super::{acir_field_utils, BrilligDriver};

macro_rules! wrapping_op {
    ($lhs:expr, $op: tt, $rhs:expr, $bit_size:expr) => {
        ($lhs $op $rhs) % match $bit_size {
                    IntegerBitSize::U1 => 1 << 1,
                    IntegerBitSize::U8 => 1 << 8,
                    IntegerBitSize::U16 => 1 << 16,
                    IntegerBitSize::U32 => 1 << 32,
                    IntegerBitSize::U64 => 1 << 64,
                    IntegerBitSize::U128 => unreachable!("match it"),
                }
    };
}

#[derive(Default)]
pub struct PlainBrilligDriver<F: PrimeField> {
    phantom_data: PhantomData<F>,
}
impl<F: PrimeField> PlainBrilligDriver<F> {
    pub fn new() -> Self {
        Self {
            phantom_data: Default::default(),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum PlainBrilligType<F: PrimeField> {
    Field(F),
    Int(u128, IntegerBitSize),
}

impl<F: PrimeField> PlainBrilligType<F> {
    pub fn into_field(self) -> F {
        match self {
            PlainBrilligType::Field(f) => f,
            PlainBrilligType::Int(i, _) => F::from(i),
        }
    }
}

impl<F: PrimeField> From<F> for PlainBrilligType<F> {
    fn from(value: F) -> Self {
        PlainBrilligType::Field(value)
    }
}

impl<F: PrimeField> Default for PlainBrilligType<F> {
    fn default() -> Self {
        Self::from(F::default())
    }
}

impl<F: PrimeField> BrilligDriver<F> for PlainBrilligDriver<F> {
    type BrilligType = PlainBrilligType<F>;

    fn cast(&self, src: Self::BrilligType, bit_size: BitSize) -> eyre::Result<Self::BrilligType> {
        let casted = match (src, bit_size) {
            // no-op
            (PlainBrilligType::Field(f), BitSize::Field) => PlainBrilligType::Field(f),
            // downcast to int
            (PlainBrilligType::Field(f), BitSize::Integer(target_bit_size)) => {
                let target_bit_size_u32: u32 = target_bit_size.into();
                let mask = (1_u128 << target_bit_size_u32) - 1;
                PlainBrilligType::Int(acir_field_utils::to_u128(f) & mask, target_bit_size)
            }
            // promote to field
            (PlainBrilligType::Int(int, _), BitSize::Field) => {
                PlainBrilligType::Field(F::from(int))
            }
            (PlainBrilligType::Int(int, bit_size), BitSize::Integer(target_bit_size)) => {
                if bit_size <= target_bit_size {
                    // upcast
                    PlainBrilligType::Int(int, target_bit_size)
                } else {
                    // downcast
                    let target_bit_size_u32: u32 = target_bit_size.into();
                    let mask = (1_u128 << target_bit_size_u32) - 1;
                    PlainBrilligType::Int(int & mask, target_bit_size)
                }
            }
        };
        Ok(casted)
    }

    fn try_into_usize(val: Self::BrilligType) -> eyre::Result<usize> {
        match val {
            PlainBrilligType::Field(_) => Err(eyre::eyre!("cannot convert field to usize")),
            PlainBrilligType::Int(int, bit_size) => {
                if bit_size == MEMORY_ADDRESSING_BIT_SIZE {
                    Ok(usize::try_from(int).expect("u32 into usize"))
                } else {
                    Err(eyre::eyre!(
                        "Must be {} for addresses, but is {}",
                        MEMORY_ADDRESSING_BIT_SIZE,
                        bit_size
                    ))
                }
            }
        }
    }

    fn try_into_bool(val: Self::BrilligType) -> eyre::Result<bool> {
        match val {
            PlainBrilligType::Int(val, bit_size) if bit_size == IntegerBitSize::U1 => Ok(val != 0),
            x => eyre::bail!("cannot cast {x:?} to bool"),
        }
    }

    fn constant(val: F, bit_size: BitSize) -> Self::BrilligType {
        match bit_size {
            BitSize::Field => PlainBrilligType::Field(val),
            BitSize::Integer(bit_size) => {
                PlainBrilligType::Int(acir_field_utils::to_u128(val), bit_size)
            }
        }
    }

    fn add_franco(
        &self,
        lhs: Self::BrilligType,
        rhs: Self::BrilligType,
    ) -> eyre::Result<Self::BrilligType> {
        match (lhs, rhs) {
            (PlainBrilligType::Field(lhs), PlainBrilligType::Field(rhs)) => {
                Ok(PlainBrilligType::Field(lhs + rhs))
            }

            (
                PlainBrilligType::Int(lhs, IntegerBitSize::U128),
                PlainBrilligType::Int(rhs, IntegerBitSize::U128),
            ) => {
                let result = lhs.wrapping_add(rhs);
                Ok(PlainBrilligType::Int(result, IntegerBitSize::U1))
            }
            (
                PlainBrilligType::Int(lhs, lhs_bit_size),
                PlainBrilligType::Int(rhs, rhs_bit_size),
            ) if lhs_bit_size == rhs_bit_size => {
                let result = wrapping_op!(lhs, +, rhs, lhs_bit_size);
                Ok(PlainBrilligType::Int(result, lhs_bit_size))
            }
            x => eyre::bail!(
                "type mismatch! Can only to bin ops on same types, but tried with {x:?}"
            ),
        }
    }

    fn lt_franco(
        &self,
        lhs: Self::BrilligType,
        rhs: Self::BrilligType,
    ) -> eyre::Result<Self::BrilligType> {
        match (lhs, rhs) {
            (PlainBrilligType::Field(lhs), PlainBrilligType::Field(rhs)) => {
                let result = u128::from(lhs < rhs);
                Ok(PlainBrilligType::Int(result, IntegerBitSize::U1))
            }
            (
                PlainBrilligType::Int(lhs, lhs_bit_size),
                PlainBrilligType::Int(rhs, rhs_bit_size),
            ) if lhs_bit_size == rhs_bit_size => {
                let result = u128::from(lhs < rhs);
                Ok(PlainBrilligType::Int(result, IntegerBitSize::U1))
            }
            x => eyre::bail!(
                "type mismatch! Can only to bin ops on same types, but tried with {x:?}"
            ),
        }
    }

    fn not_franco(&self, val: Self::BrilligType) -> eyre::Result<Self::BrilligType> {
        if let PlainBrilligType::Int(val, IntegerBitSize::U1) = val {
            Ok(PlainBrilligType::Int(
                u128::from(val == 0),
                IntegerBitSize::U1,
            ))
        } else {
            eyre::bail!("")
        }
    }

    fn gt_franco(
        &self,
        lhs: Self::BrilligType,
        rhs: Self::BrilligType,
    ) -> eyre::Result<Self::BrilligType> {
        match (lhs, rhs) {
            (PlainBrilligType::Field(lhs), PlainBrilligType::Field(rhs)) => {
                let result = u128::from(lhs > rhs);
                Ok(PlainBrilligType::Int(result, IntegerBitSize::U1))
            }
            (
                PlainBrilligType::Int(lhs, lhs_bit_size),
                PlainBrilligType::Int(rhs, rhs_bit_size),
            ) if lhs_bit_size == rhs_bit_size => {
                let result = u128::from(lhs > rhs);
                Ok(PlainBrilligType::Int(result, IntegerBitSize::U1))
            }
            x => eyre::bail!(
                "type mismatch! Can only to bin ops on same types, but tried with {x:?}"
            ),
        }
    }

    fn expect_int_bit_size(
        val: Self::BrilligType,
        should_bit_size: IntegerBitSize,
    ) -> eyre::Result<Self::BrilligType> {
        if let PlainBrilligType::Int(_, is_bit_size) = val {
            if is_bit_size == should_bit_size {
                Ok(val)
            } else {
                eyre::bail!("expected int with size: {should_bit_size}, but got {val:?}")
            }
        } else {
            eyre::bail!("expected int with size: {should_bit_size}, but got {val:?}")
        }
    }
}
