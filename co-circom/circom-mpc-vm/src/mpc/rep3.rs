use ark_ff::PrimeField;
use eyre::bail;
use mpc_core::{
    protocols::{
        rep3::network::Rep3Network,
        rep3new::{
            arithmetic, binary, conversion, network::IoContext, Rep3BigUintShare,
            Rep3PrimeFieldShare,
        },
    },
    traits::SecretShared,
};

use super::{plain::PlainDriver, VmCircomWitnessExtension};

type ArithmeticShare<F> = Rep3PrimeFieldShare<F>;
type BinaryShare<F> = Rep3BigUintShare<F>;

#[derive(Clone)]
pub enum Rep3VmType<F: PrimeField> {
    Public(F),
    Arithmetic(ArithmeticShare<F>),
    Binary(BinaryShare<F>),
}

impl<F: PrimeField> From<F> for Rep3VmType<F> {
    fn from(value: F) -> Self {
        Self::Public(value)
    }
}

impl<F: PrimeField> From<ArithmeticShare<F>> for Rep3VmType<F> {
    fn from(value: ArithmeticShare<F>) -> Self {
        Self::Arithmetic(value)
    }
}

impl<F: PrimeField> From<BinaryShare<F>> for Rep3VmType<F> {
    fn from(value: BinaryShare<F>) -> Self {
        Self::Binary(value)
    }
}

impl<F: PrimeField> Default for Rep3VmType<F> {
    fn default() -> Self {
        Self::Public(F::zero())
    }
}

pub struct Rep3Driver<F: PrimeField, N: Rep3Network> {
    io_context: IoContext<N>,
    plain: PlainDriver<F>,
}

impl<F: PrimeField, N: Rep3Network> Rep3Driver<F, N> {
    pub async fn new(network: N) -> std::io::Result<Self> {
        Ok(Self {
            io_context: IoContext::init(network).await?,
            plain: PlainDriver::default(),
        })
    }
}

impl<F: PrimeField + SecretShared, N: Rep3Network> VmCircomWitnessExtension<F>
    for Rep3Driver<F, N>
{
    type ArithmeticShare = ArithmeticShare<F>;

    type BinaryShare = BinaryShare<F>;

    type VmType = Rep3VmType<F>;

    async fn add(&mut self, a: Self::VmType, b: Self::VmType) -> eyre::Result<Self::VmType> {
        match (a, b) {
            (Rep3VmType::Public(a), Rep3VmType::Public(b)) => {
                Ok(self.plain.add(a, b).await?.into())
            }
            (Rep3VmType::Public(b), Rep3VmType::Arithmetic(a))
            | (Rep3VmType::Arithmetic(a), Rep3VmType::Public(b)) => {
                Ok(arithmetic::add_public(&a, b, self.io_context.id).into())
            }
            (Rep3VmType::Arithmetic(a), Rep3VmType::Arithmetic(b)) => {
                Ok(arithmetic::add(&a, &b).into())
            }
            (Rep3VmType::Public(b), Rep3VmType::Binary(a))
            | (Rep3VmType::Binary(a), Rep3VmType::Public(b)) => {
                let a = conversion::b2a(a, &mut self.io_context).await?;
                Ok(arithmetic::add_public(&a, b, self.io_context.id).into())
            }
            (Rep3VmType::Arithmetic(a), Rep3VmType::Binary(b))
            | (Rep3VmType::Binary(b), Rep3VmType::Arithmetic(a)) => {
                let b = conversion::b2a(b, &mut self.io_context).await?;
                Ok(arithmetic::add(&a, &b).into())
            }
            (Rep3VmType::Binary(a), Rep3VmType::Binary(b)) => {
                let a = conversion::b2a(a, &mut self.io_context).await?;
                let b = conversion::b2a(b, &mut self.io_context).await?;
                Ok(arithmetic::add(&a, &b).into())
            }
        }
    }

    async fn sub(&mut self, a: Self::VmType, b: Self::VmType) -> eyre::Result<Self::VmType> {
        match (a, b) {
            (Rep3VmType::Public(a), Rep3VmType::Public(b)) => {
                Ok(self.plain.sub(a, b).await?.into())
            }
            (Rep3VmType::Arithmetic(a), Rep3VmType::Public(b)) => {
                Ok(arithmetic::sub_public(&a, b, self.io_context.id).into())
            }
            (Rep3VmType::Public(a), Rep3VmType::Arithmetic(b)) => {
                Ok(arithmetic::sub_public(&b, a, self.io_context.id).into())
            }
            (Rep3VmType::Arithmetic(a), Rep3VmType::Arithmetic(b)) => {
                Ok(arithmetic::sub(&a, &b).into())
            }
            (Rep3VmType::Public(a), Rep3VmType::Binary(b)) => {
                let b = conversion::b2a(b, &mut self.io_context).await?;
                Ok(arithmetic::sub_public(&b, a, self.io_context.id).into())
            }
            (Rep3VmType::Binary(a), Rep3VmType::Public(b)) => {
                let a = conversion::b2a(a, &mut self.io_context).await?;
                Ok(arithmetic::sub_public(&a, b, self.io_context.id).into())
            }
            (Rep3VmType::Arithmetic(a), Rep3VmType::Binary(b)) => {
                let b = conversion::b2a(b, &mut self.io_context).await?;
                Ok(arithmetic::sub(&a, &b).into())
            }
            (Rep3VmType::Binary(a), Rep3VmType::Arithmetic(b)) => {
                let a = conversion::b2a(a, &mut self.io_context).await?;
                Ok(arithmetic::sub(&a, &b).into())
            }
            (Rep3VmType::Binary(a), Rep3VmType::Binary(b)) => {
                let a = conversion::b2a(a, &mut self.io_context).await?;
                let b = conversion::b2a(b, &mut self.io_context).await?;
                Ok(arithmetic::sub(&a, &b).into())
            }
        }
    }

    async fn mul(&mut self, a: Self::VmType, b: Self::VmType) -> eyre::Result<Self::VmType> {
        match (a, b) {
            (Rep3VmType::Public(a), Rep3VmType::Public(b)) => {
                Ok(self.plain.mul(a, b).await?.into())
            }
            (Rep3VmType::Public(b), Rep3VmType::Arithmetic(a))
            | (Rep3VmType::Arithmetic(a), Rep3VmType::Public(b)) => {
                Ok(arithmetic::mul_with_public(&a, b).into())
            }
            (Rep3VmType::Arithmetic(a), Rep3VmType::Arithmetic(b)) => {
                Ok(arithmetic::mul(&a, &b, &mut self.io_context).await?.into())
            }
            (Rep3VmType::Public(b), Rep3VmType::Binary(a))
            | (Rep3VmType::Binary(a), Rep3VmType::Public(b)) => {
                let a = conversion::b2a(a, &mut self.io_context).await?;
                Ok(arithmetic::mul_with_public(&a, b).into())
            }
            (Rep3VmType::Arithmetic(a), Rep3VmType::Binary(b))
            | (Rep3VmType::Binary(b), Rep3VmType::Arithmetic(a)) => {
                let b = conversion::b2a(b, &mut self.io_context).await?;
                Ok(arithmetic::mul(&a, &b, &mut self.io_context).await?.into())
            }
            (Rep3VmType::Binary(a), Rep3VmType::Binary(b)) => {
                let a = conversion::b2a(a, &mut self.io_context).await?;
                let b = conversion::b2a(b, &mut self.io_context).await?;
                Ok(arithmetic::mul(&a, &b, &mut self.io_context).await?.into())
            }
        }
    }

    async fn div(&mut self, a: Self::VmType, b: Self::VmType) -> eyre::Result<Self::VmType> {
        match (a, b) {
            (Rep3VmType::Public(a), Rep3VmType::Public(b)) => {
                Ok(self.plain.div(a, b).await?.into())
            }
            (Rep3VmType::Public(a), Rep3VmType::Arithmetic(b)) => {
                let b = arithmetic::inv(&b, &mut self.io_context).await?;
                Ok(arithmetic::mul_with_public(&b, a).into())
            }
            (Rep3VmType::Arithmetic(a), Rep3VmType::Public(b)) => {
                if b.is_zero() {
                    bail!("Cannot invert zero");
                }
                Ok(arithmetic::mul_with_public(&a, b.inverse().unwrap()).into())
            }
            (Rep3VmType::Arithmetic(a), Rep3VmType::Arithmetic(b)) => {
                let b = arithmetic::inv(&b, &mut self.io_context).await?;
                Ok(arithmetic::mul(&a, &b, &mut self.io_context).await?.into())
            }
            (Rep3VmType::Public(a), Rep3VmType::Binary(b)) => {
                let b = conversion::b2a(b, &mut self.io_context).await?;
                let b = arithmetic::inv(&b, &mut self.io_context).await?;
                Ok(arithmetic::mul_with_public(&b, a).into())
            }
            (Rep3VmType::Binary(a), Rep3VmType::Public(b)) => {
                let a = conversion::b2a(a, &mut self.io_context).await?;
                if b.is_zero() {
                    bail!("Cannot invert zero");
                }
                Ok(arithmetic::mul_with_public(&a, b.inverse().unwrap()).into())
            }
            (Rep3VmType::Arithmetic(a), Rep3VmType::Binary(b)) => {
                let b = conversion::b2a(b, &mut self.io_context).await?;
                let b = arithmetic::inv(&b, &mut self.io_context).await?;
                Ok(arithmetic::mul(&a, &b, &mut self.io_context).await?.into())
            }
            (Rep3VmType::Binary(a), Rep3VmType::Arithmetic(b)) => {
                let a = conversion::b2a(a, &mut self.io_context).await?;
                let b = arithmetic::inv(&b, &mut self.io_context).await?;
                Ok(arithmetic::mul(&a, &b, &mut self.io_context).await?.into())
            }
            (Rep3VmType::Binary(a), Rep3VmType::Binary(b)) => {
                let a = conversion::b2a(a, &mut self.io_context).await?;
                let b = conversion::b2a(b, &mut self.io_context).await?;
                let b = arithmetic::inv(&b, &mut self.io_context).await?;
                Ok(arithmetic::mul(&a, &b, &mut self.io_context).await?.into())
            }
        }
    }

    fn int_div(&mut self, a: Self::VmType, b: Self::VmType) -> eyre::Result<Self::VmType> {
        todo!()
    }

    fn pow(&mut self, a: Self::VmType, b: Self::VmType) -> eyre::Result<Self::VmType> {
        todo!()
    }

    fn modulo(&mut self, a: Self::VmType, b: Self::VmType) -> eyre::Result<Self::VmType> {
        todo!()
    }

    fn sqrt(&mut self, a: Self::VmType) -> eyre::Result<Self::VmType> {
        todo!()
    }

    fn neg(&mut self, a: Self::VmType) -> Self::VmType {
        todo!()
    }

    fn lt(&mut self, a: Self::VmType, b: Self::VmType) -> eyre::Result<Self::VmType> {
        todo!()
    }

    fn le(&mut self, a: Self::VmType, b: Self::VmType) -> eyre::Result<Self::VmType> {
        todo!()
    }

    fn gt(&mut self, a: Self::VmType, b: Self::VmType) -> eyre::Result<Self::VmType> {
        todo!()
    }

    fn ge(&mut self, a: Self::VmType, b: Self::VmType) -> eyre::Result<Self::VmType> {
        todo!()
    }

    fn eq(&mut self, a: Self::VmType, b: Self::VmType) -> eyre::Result<Self::VmType> {
        todo!()
    }

    fn neq(&mut self, a: Self::VmType, b: Self::VmType) -> eyre::Result<Self::VmType> {
        todo!()
    }

    fn shift_r(&mut self, a: Self::VmType, b: Self::VmType) -> eyre::Result<Self::VmType> {
        todo!()
    }

    fn shift_l(&mut self, a: Self::VmType, b: Self::VmType) -> eyre::Result<Self::VmType> {
        todo!()
    }

    fn bool_not(&mut self, a: Self::VmType) -> eyre::Result<Self::VmType> {
        todo!()
    }

    fn bool_and(&mut self, a: Self::VmType, b: Self::VmType) -> eyre::Result<Self::VmType> {
        todo!()
    }

    fn bool_or(&mut self, a: Self::VmType, b: Self::VmType) -> eyre::Result<Self::VmType> {
        todo!()
    }

    fn cmux(
        &mut self,
        cond: Self::VmType,
        truthy: Self::VmType,
        falsy: Self::VmType,
    ) -> eyre::Result<Self::VmType> {
        todo!()
    }

    async fn bit_xor(&mut self, a: Self::VmType, b: Self::VmType) -> eyre::Result<Self::VmType> {
        match (a, b) {
            (Rep3VmType::Public(a), Rep3VmType::Public(b)) => {
                Ok(self.plain.bit_xor(a, b).await?.into())
            }
            (Rep3VmType::Public(b), Rep3VmType::Arithmetic(a))
            | (Rep3VmType::Arithmetic(a), Rep3VmType::Public(b)) => {
                let a = conversion::a2b(&a, &mut self.io_context).await?;
                Ok(binary::xor_public(&a, &b.into_bigint().into(), self.io_context.id).into())
            }
            (Rep3VmType::Arithmetic(a), Rep3VmType::Arithmetic(b)) => {
                let a = conversion::a2b(&a, &mut self.io_context).await?;
                let b = conversion::a2b(&b, &mut self.io_context).await?;
                Ok(binary::xor(&a, &b).into())
            }
            (Rep3VmType::Public(b), Rep3VmType::Binary(a))
            | (Rep3VmType::Binary(a), Rep3VmType::Public(b)) => {
                Ok(binary::xor_public(&a, &b.into_bigint().into(), self.io_context.id).into())
            }
            (Rep3VmType::Arithmetic(a), Rep3VmType::Binary(b))
            | (Rep3VmType::Binary(b), Rep3VmType::Arithmetic(a)) => {
                let a = conversion::a2b(&a, &mut self.io_context).await?;
                Ok(binary::xor(&a, &b).into())
            }
            (Rep3VmType::Binary(a), Rep3VmType::Binary(b)) => Ok(binary::xor(&a, &b).into()),
        }
    }

    async fn bit_or(&mut self, a: Self::VmType, b: Self::VmType) -> eyre::Result<Self::VmType> {
        match (a, b) {
            (Rep3VmType::Public(a), Rep3VmType::Public(b)) => {
                Ok(self.plain.bit_or(a, b).await?.into())
            }
            (Rep3VmType::Public(b), Rep3VmType::Arithmetic(a))
            | (Rep3VmType::Arithmetic(a), Rep3VmType::Public(b)) => {
                let a = conversion::a2b(&a, &mut self.io_context).await?;
                self.bit_or(a.into(), b.into()).await
            }
            (Rep3VmType::Arithmetic(a), Rep3VmType::Arithmetic(b)) => {
                let a = conversion::a2b(&a, &mut self.io_context).await?;
                let b = conversion::a2b(&b, &mut self.io_context).await?;
                self.bit_or(a.into(), b.into()).await
            }
            (Rep3VmType::Public(b), Rep3VmType::Binary(a))
            | (Rep3VmType::Binary(a), Rep3VmType::Public(b)) => {
                Ok(binary::or_public(&a, &b.into_bigint().into(), self.io_context.id).into())
            }
            (Rep3VmType::Arithmetic(a), Rep3VmType::Binary(b))
            | (Rep3VmType::Binary(b), Rep3VmType::Arithmetic(a)) => {
                let a = conversion::a2b(&a, &mut self.io_context).await?;
                self.bit_or(a.into(), b.into()).await
            }
            (Rep3VmType::Binary(a), Rep3VmType::Binary(b)) => {
                Ok(binary::or(&a, &b, &mut self.io_context).await?.into())
            }
        }
    }

    async fn bit_and(&mut self, a: Self::VmType, b: Self::VmType) -> eyre::Result<Self::VmType> {
        match (a, b) {
            (Rep3VmType::Public(a), Rep3VmType::Public(b)) => {
                Ok(self.plain.bit_and(a, b).await?.into())
            }
            (Rep3VmType::Public(b), Rep3VmType::Arithmetic(a))
            | (Rep3VmType::Arithmetic(a), Rep3VmType::Public(b)) => {
                let a = conversion::a2b(&a, &mut self.io_context).await?;
                Ok(binary::and_with_public(&a, &b.into_bigint().into()).into())
            }
            (Rep3VmType::Arithmetic(a), Rep3VmType::Arithmetic(b)) => {
                let a = conversion::a2b(&a, &mut self.io_context).await?;
                let b = conversion::a2b(&b, &mut self.io_context).await?;
                Ok(binary::and(&a, &b, &mut self.io_context).await?.into())
            }
            (Rep3VmType::Public(b), Rep3VmType::Binary(a))
            | (Rep3VmType::Binary(a), Rep3VmType::Public(b)) => {
                Ok(binary::and_with_public(&a, &b.into_bigint().into()).into())
            }
            (Rep3VmType::Arithmetic(a), Rep3VmType::Binary(b))
            | (Rep3VmType::Binary(b), Rep3VmType::Arithmetic(a)) => {
                let a = conversion::a2b(&a, &mut self.io_context).await?;
                Ok(binary::and(&a, &b, &mut self.io_context).await?.into())
            }
            (Rep3VmType::Binary(a), Rep3VmType::Binary(b)) => {
                Ok(binary::and(&a, &b, &mut self.io_context).await?.into())
            }
        }
    }

    fn is_zero(&mut self, a: Self::VmType, allow_secret_inputs: bool) -> eyre::Result<bool> {
        todo!()
    }

    fn is_shared(&mut self, a: &Self::VmType) -> eyre::Result<bool> {
        todo!()
    }

    fn to_index(&mut self, a: Self::VmType) -> eyre::Result<usize> {
        todo!()
    }

    fn open(&mut self, a: Self::VmType) -> eyre::Result<F> {
        todo!()
    }

    fn to_share(&self, a: Self::VmType) -> Self::ArithmeticShare {
        todo!()
    }

    fn public_one(&self) -> Self::VmType {
        todo!()
    }

    fn public_zero(&self) -> Self::VmType {
        todo!()
    }
}

impl<F: PrimeField> std::fmt::Debug for Rep3VmType<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Public(field) => f.debug_tuple("Public").field(field).finish(),
            Self::Arithmetic(share) => f.debug_tuple("Arithmetic").field(share).finish(),
            Self::Binary(binary) => f.debug_tuple("Binary").field(binary).finish(),
        }
    }
}

impl<F: PrimeField> std::fmt::Display for Rep3VmType<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Public(field) => f.write_str(&format!("Public ({field})")),
            Self::Arithmetic(arithmetic) => {
                let (a, b) = arithmetic.clone().ab();
                f.write_str(&format!("Arithmetic (a: {}, b: {})", a, b))
            }
            Self::Binary(binary) => {
                let (a, b) = binary.clone().ab();
                f.write_str(&format!("Binary (a: {}, b: {})", a, b))
            }
        }
    }
}
