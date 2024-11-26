use super::{BrilligDriver, PlainBrilligDriver};
use ark_ff::PrimeField;
use brillig::{BitSize, IntegerBitSize};
use mpc_core::protocols::shamir::{network::ShamirNetwork, ShamirPrimeFieldShare, ShamirProtocol};
use std::marker::PhantomData;

use super::PlainBrilligType as Public;

/// A driver for the coBrillig-VM that uses Shamir secret sharing.
pub struct ShamirBrilligDriver<F: PrimeField, N: ShamirNetwork> {
    protocol: ShamirProtocol<F, N>,
    plain_driver: PlainBrilligDriver<F>,
    phantom_data: PhantomData<F>,
}

/// The types for the coBrillig
/// Shamir driver. The values
/// can either be shared or public.
#[derive(Clone, Debug, PartialEq)]
pub enum ShamirBrilligType<F: PrimeField> {
    /// A public value
    Public(Public<F>),
    /// A shared value.
    /// TODO for now we only support prime fields
    Shared(ShamirPrimeFieldShare<F>),
}

impl<F: PrimeField> From<F> for ShamirBrilligType<F> {
    fn from(value: F) -> Self {
        ShamirBrilligType::Public(Public::Field(value))
    }
}

impl<F: PrimeField> Default for ShamirBrilligType<F> {
    fn default() -> Self {
        Self::from(F::default())
    }
}

impl<F: PrimeField, N: ShamirNetwork> BrilligDriver<F> for ShamirBrilligDriver<F, N> {
    type BrilligType = ShamirBrilligType<F>;

    fn cast(&self, val: Self::BrilligType, bit_size: BitSize) -> eyre::Result<Self::BrilligType> {
        if let ShamirBrilligType::Public(public) = val {
            let casted = self.plain_driver.cast(public, bit_size)?;
            Ok(ShamirBrilligType::Public(casted))
        } else {
            eyre::bail!("Cannot cast shared value with Shamir")
        }
    }

    fn try_into_usize(val: Self::BrilligType) -> eyre::Result<usize> {
        // for now we only support casting public values to usize
        // we return an error if we call this on a shared value
        if let ShamirBrilligType::Public(public) = val {
            PlainBrilligDriver::try_into_usize(public)
        } else {
            eyre::bail!("cannot convert shared value to usize")
        }
    }

    fn try_into_bool(val: Self::BrilligType) -> eyre::Result<bool> {
        // for now we only support casting public values to bools
        // we return an error if we call this on a shared value
        if let ShamirBrilligType::Public(public) = val {
            PlainBrilligDriver::try_into_bool(public)
        } else {
            eyre::bail!("cannot convert shared value to usize")
        }
    }

    fn public_value(val: F, bit_size: BitSize) -> Self::BrilligType {
        ShamirBrilligType::Public(PlainBrilligDriver::public_value(val, bit_size))
    }

    fn add(
        &self,
        _lhs: Self::BrilligType,
        _rhs: Self::BrilligType,
    ) -> eyre::Result<Self::BrilligType> {
        todo!()
    }

    fn sub(
        &mut self,
        _lhs: Self::BrilligType,
        _rhs: Self::BrilligType,
    ) -> eyre::Result<Self::BrilligType> {
        todo!()
    }

    fn mul(
        &mut self,
        _lhs: Self::BrilligType,
        _rhs: Self::BrilligType,
    ) -> eyre::Result<Self::BrilligType> {
        todo!()
    }

    fn div(
        &mut self,
        _lhs: Self::BrilligType,
        _rhs: Self::BrilligType,
    ) -> eyre::Result<Self::BrilligType> {
        todo!()
    }

    fn int_div(
        &mut self,
        _lhs: Self::BrilligType,
        _rhs: Self::BrilligType,
    ) -> eyre::Result<Self::BrilligType> {
        todo!()
    }

    fn not(&self, _val: Self::BrilligType) -> eyre::Result<Self::BrilligType> {
        todo!()
    }

    fn eq(
        &mut self,
        _lhs: Self::BrilligType,
        _rhs: Self::BrilligType,
    ) -> eyre::Result<Self::BrilligType> {
        todo!()
    }

    fn lt(
        &mut self,
        _lhs: Self::BrilligType,
        _rhs: Self::BrilligType,
    ) -> eyre::Result<Self::BrilligType> {
        todo!()
    }

    fn le(
        &mut self,
        lhs: Self::BrilligType,
        rhs: Self::BrilligType,
    ) -> eyre::Result<Self::BrilligType> {
        let gt = self.gt(lhs, rhs)?;
        self.not(gt)
    }

    fn gt(
        &mut self,
        _lhs: Self::BrilligType,
        _rhs: Self::BrilligType,
    ) -> eyre::Result<Self::BrilligType> {
        todo!()
    }

    fn ge(
        &mut self,
        lhs: Self::BrilligType,
        rhs: Self::BrilligType,
    ) -> eyre::Result<Self::BrilligType> {
        let gt = self.lt(lhs, rhs)?;
        self.not(gt)
    }

    fn to_radix(
        &mut self,
        val: Self::BrilligType,
        radix: Self::BrilligType,
        output_size: usize,
        bits: bool,
    ) -> eyre::Result<Vec<Self::BrilligType>> {
        if let (ShamirBrilligType::Public(val), ShamirBrilligType::Public(radix)) = (val, radix) {
            let result = self.plain_driver.to_radix(val, radix, output_size, bits)?;
            Ok(result
                .into_iter()
                .map(|val| ShamirBrilligType::Public(val))
                .collect())
        } else {
            eyre::bail!("Cannot use to_radix with Shamir shares")
        }
    }

    fn expect_int(
        val: Self::BrilligType,
        bit_size: IntegerBitSize,
    ) -> eyre::Result<Self::BrilligType> {
        if let ShamirBrilligType::Public(public) = val {
            let result = PlainBrilligDriver::expect_int(public, bit_size)?;
            Ok(ShamirBrilligType::Public(result))
        } else {
            eyre::bail!("expected int with bit size {bit_size}, but was something else")
        }
    }

    fn expect_field(val: Self::BrilligType) -> eyre::Result<Self::BrilligType> {
        match &val {
            ShamirBrilligType::Public(Public::Field(_)) | ShamirBrilligType::Shared(_) => Ok(val),
            _ => eyre::bail!("expected field but got int"),
        }
    }
}
