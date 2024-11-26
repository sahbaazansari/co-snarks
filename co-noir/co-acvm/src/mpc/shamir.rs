use super::{plain::PlainAcvmSolver, NoirWitnessExtensionProtocol};
use ark_ff::PrimeField;
use co_brillig::mpc::{ShamirBrilligDriver, ShamirBrilligType};
use mpc_core::protocols::{
    rep3::{lut::NaiveRep3LookupTable, network::Rep3MpcNet},
    shamir::{arithmetic, network::ShamirNetwork, ShamirPrimeFieldShare, ShamirProtocol},
};
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

pub struct ShamirAcvmSolver<F: PrimeField, N: ShamirNetwork> {
    protocol: ShamirProtocol<F, N>,
    plain_solver: PlainAcvmSolver<F>,
    phantom_data: PhantomData<F>,
}

impl<F: PrimeField, N: ShamirNetwork> ShamirAcvmSolver<F, N> {
    pub fn new(protocol: ShamirProtocol<F, N>) -> Self {
        let plain_solver = PlainAcvmSolver::<F>::default();
        Self {
            protocol,
            plain_solver,
            phantom_data: PhantomData,
        }
    }

    pub fn into_network(self) -> N {
        self.protocol.network
    }
}

#[derive(Clone, Serialize, Deserialize, PartialEq)]
pub enum ShamirAcvmType<F: PrimeField> {
    Public(
        #[serde(
            serialize_with = "mpc_core::ark_se",
            deserialize_with = "mpc_core::ark_de"
        )]
        F,
    ),
    Shared(
        #[serde(
            serialize_with = "mpc_core::ark_se",
            deserialize_with = "mpc_core::ark_de"
        )]
        ShamirPrimeFieldShare<F>,
    ),
}

impl<F: PrimeField> std::fmt::Debug for ShamirAcvmType<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Public(field) => f.debug_tuple("Public").field(field).finish(),
            Self::Shared(share) => f.debug_tuple("Arithmetic").field(share).finish(),
        }
    }
}

impl<F: PrimeField> std::fmt::Display for ShamirAcvmType<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Public(field) => f.write_str(&format!("Public ({field})")),
            Self::Shared(arithmetic) => {
                f.write_str(&format!("Arithmetic ({})", arithmetic.inner()))
            }
        }
    }
}

impl<F: PrimeField> Default for ShamirAcvmType<F> {
    fn default() -> Self {
        Self::Public(F::zero())
    }
}

impl<F: PrimeField> From<F> for ShamirAcvmType<F> {
    fn from(value: F) -> Self {
        Self::Public(value)
    }
}

impl<F: PrimeField> From<ShamirPrimeFieldShare<F>> for ShamirAcvmType<F> {
    fn from(value: ShamirPrimeFieldShare<F>) -> Self {
        Self::Shared(value)
    }
}

impl<F: PrimeField> From<ShamirAcvmType<F>> for ShamirBrilligType<F> {
    fn from(val: ShamirAcvmType<F>) -> Self {
        match val {
            ShamirAcvmType::Public(public) => ShamirBrilligType::from(public),
            ShamirAcvmType::Shared(share) => ShamirBrilligType::Shared(share),
        }
    }
}

impl<F: PrimeField> From<ShamirBrilligType<F>> for ShamirAcvmType<F> {
    fn from(value: ShamirBrilligType<F>) -> Self {
        match value {
            ShamirBrilligType::Public(public) => ShamirAcvmType::Public(public.into_field()),
            ShamirBrilligType::Shared(shared) => ShamirAcvmType::Shared(shared),
        }
    }
}

impl<F: PrimeField, N: ShamirNetwork> NoirWitnessExtensionProtocol<F> for ShamirAcvmSolver<F, N> {
    type Lookup = NaiveRep3LookupTable<Rep3MpcNet>; // This is just a dummy and unused

    type ArithmeticShare = ShamirPrimeFieldShare<F>;

    type AcvmType = ShamirAcvmType<F>;

    type BrilligDriver = ShamirBrilligDriver<F, N>;

    fn init_brillig_driver(&mut self) -> std::io::Result<Self::BrilligDriver> {
        todo!()
    }

    fn public_zero() -> Self::AcvmType {
        Self::AcvmType::default()
    }

    fn from_brillig_result(brillig_result: Vec<ShamirBrilligType<F>>) -> Vec<Self::AcvmType> {
        brillig_result
            .into_iter()
            .map(ShamirAcvmType::from)
            .collect()
    }

    fn is_public_zero(a: &Self::AcvmType) -> bool {
        if let ShamirAcvmType::Public(x) = a {
            x.is_zero()
        } else {
            false
        }
    }

    fn is_public_one(a: &Self::AcvmType) -> bool {
        if let ShamirAcvmType::Public(x) = a {
            x.is_one()
        } else {
            false
        }
    }

    fn acvm_add_assign_with_public(&mut self, public: F, target: &mut Self::AcvmType) {
        let result = match target.to_owned() {
            ShamirAcvmType::Public(secret) => ShamirAcvmType::Public(public + secret),
            ShamirAcvmType::Shared(secret) => {
                ShamirAcvmType::Shared(arithmetic::add_public(secret, public))
            }
        };
        *target = result;
    }

    fn acvm_sub(&mut self, share_1: Self::AcvmType, share_2: Self::AcvmType) -> Self::AcvmType {
        match (share_1, share_2) {
            (ShamirAcvmType::Public(share_1), ShamirAcvmType::Public(share_2)) => {
                ShamirAcvmType::Public(share_1 - share_2)
            }
            (ShamirAcvmType::Public(share_1), ShamirAcvmType::Shared(share_2)) => {
                ShamirAcvmType::Shared(arithmetic::add_public(-share_2, share_1))
            }
            (ShamirAcvmType::Shared(share_1), ShamirAcvmType::Public(share_2)) => {
                ShamirAcvmType::Shared(arithmetic::add_public(share_1, -share_2))
            }
            (ShamirAcvmType::Shared(share_1), ShamirAcvmType::Shared(share_2)) => {
                let result = arithmetic::sub(share_1, share_2);
                ShamirAcvmType::Shared(result)
            }
        }
    }

    fn acvm_mul_with_public(&mut self, public: F, secret: Self::AcvmType) -> Self::AcvmType {
        match secret {
            ShamirAcvmType::Public(secret) => ShamirAcvmType::Public(public * secret),
            ShamirAcvmType::Shared(secret) => {
                ShamirAcvmType::Shared(arithmetic::mul_public(secret, public))
            }
        }
    }

    fn acvm_mul(
        &mut self,
        secret_1: Self::AcvmType,
        secret_2: Self::AcvmType,
    ) -> std::io::Result<Self::AcvmType> {
        match (secret_1, secret_2) {
            (ShamirAcvmType::Public(secret_1), ShamirAcvmType::Public(secret_2)) => {
                Ok(ShamirAcvmType::Public(secret_1 * secret_2))
            }
            (ShamirAcvmType::Public(secret_1), ShamirAcvmType::Shared(secret_2)) => Ok(
                ShamirAcvmType::Shared(arithmetic::mul_public(secret_2, secret_1)),
            ),
            (ShamirAcvmType::Shared(secret_1), ShamirAcvmType::Public(secret_2)) => Ok(
                ShamirAcvmType::Shared(arithmetic::mul_public(secret_1, secret_2)),
            ),
            (ShamirAcvmType::Shared(secret_1), ShamirAcvmType::Shared(secret_2)) => {
                let result = arithmetic::mul(secret_1, secret_2, &mut self.protocol)?;
                Ok(ShamirAcvmType::Shared(result))
            }
        }
    }

    fn acvm_negate_inplace(&mut self, a: &mut Self::AcvmType) {
        match a {
            ShamirAcvmType::Public(public) => {
                public.neg_in_place();
            }
            ShamirAcvmType::Shared(shared) => *shared = arithmetic::neg(*shared),
        }
    }

    fn solve_linear_term(&mut self, q_l: F, w_l: Self::AcvmType, target: &mut Self::AcvmType) {
        let result = match (w_l, target.to_owned()) {
            (ShamirAcvmType::Public(w_l), ShamirAcvmType::Public(result)) => {
                ShamirAcvmType::Public(q_l * w_l + result)
            }
            (ShamirAcvmType::Public(w_l), ShamirAcvmType::Shared(result)) => {
                ShamirAcvmType::Shared(arithmetic::add_public(result, q_l * w_l))
            }
            (ShamirAcvmType::Shared(w_l), ShamirAcvmType::Public(result)) => {
                let mul = arithmetic::mul_public(w_l, q_l);
                ShamirAcvmType::Shared(arithmetic::add_public(mul, result))
            }
            (ShamirAcvmType::Shared(w_l), ShamirAcvmType::Shared(result)) => {
                let mul = arithmetic::mul_public(w_l, q_l);
                ShamirAcvmType::Shared(arithmetic::add(mul, result))
            }
        };
        *target = result;
    }

    fn add_assign(&mut self, target: &mut Self::AcvmType, rhs: Self::AcvmType) {
        let result = match (target.clone(), rhs) {
            (ShamirAcvmType::Public(lhs), ShamirAcvmType::Public(rhs)) => {
                ShamirAcvmType::Public(lhs + rhs)
            }
            (ShamirAcvmType::Public(public), ShamirAcvmType::Shared(shared))
            | (ShamirAcvmType::Shared(shared), ShamirAcvmType::Public(public)) => {
                ShamirAcvmType::Shared(arithmetic::add_public(shared, public))
            }
            (ShamirAcvmType::Shared(lhs), ShamirAcvmType::Shared(rhs)) => {
                ShamirAcvmType::Shared(arithmetic::add(lhs, rhs))
            }
        };
        *target = result;
    }

    fn solve_mul_term(
        &mut self,
        c: F,
        lhs: Self::AcvmType,
        rhs: Self::AcvmType,
    ) -> std::io::Result<Self::AcvmType> {
        let result = match (lhs, rhs) {
            (ShamirAcvmType::Public(lhs), ShamirAcvmType::Public(rhs)) => {
                ShamirAcvmType::Public(lhs * rhs * c)
            }
            (ShamirAcvmType::Public(public), ShamirAcvmType::Shared(shared))
            | (ShamirAcvmType::Shared(shared), ShamirAcvmType::Public(public)) => {
                ShamirAcvmType::Shared(arithmetic::mul_public(shared, public))
            }
            (ShamirAcvmType::Shared(lhs), ShamirAcvmType::Shared(rhs)) => {
                let shared_mul = arithmetic::mul(lhs, rhs, &mut self.protocol)?;
                ShamirAcvmType::Shared(arithmetic::mul_public(shared_mul, c))
            }
        };
        Ok(result)
    }

    fn solve_equation(
        &mut self,
        q_l: Self::AcvmType,
        c: Self::AcvmType,
    ) -> eyre::Result<Self::AcvmType> {
        //-c/q_l
        let result = match (q_l, c) {
            (ShamirAcvmType::Public(q_l), ShamirAcvmType::Public(c)) => {
                ShamirAcvmType::Public(self.plain_solver.solve_equation(q_l, c)?)
            }
            (ShamirAcvmType::Public(q_l), ShamirAcvmType::Shared(c)) => {
                ShamirAcvmType::Shared(arithmetic::div_shared_by_public(arithmetic::neg(c), q_l)?)
            }
            (ShamirAcvmType::Shared(q_l), ShamirAcvmType::Public(c)) => {
                let result = arithmetic::div_public_by_shared(-c, q_l, &mut self.protocol)?;
                ShamirAcvmType::Shared(result)
            }
            (ShamirAcvmType::Shared(q_l), ShamirAcvmType::Shared(c)) => {
                let result = arithmetic::div(arithmetic::neg(c), q_l, &mut self.protocol)?;
                ShamirAcvmType::Shared(result)
            }
        };
        Ok(result)
    }

    fn init_lut_by_acvm_type(
        &mut self,
        _values: Vec<Self::AcvmType>,
    ) -> <Self::Lookup as mpc_core::lut::LookupTableProvider<F>>::SecretSharedMap {
        panic!("init_lut_by_acvm_type: Operation atm not supported")
    }

    fn read_lut_by_acvm_type(
        &mut self,
        _index: &Self::AcvmType,
        _lut: &<Self::Lookup as mpc_core::lut::LookupTableProvider<F>>::SecretSharedMap,
    ) -> std::io::Result<Self::AcvmType> {
        panic!("read_lut_by_acvm_type: Operation atm not supported")
    }

    fn write_lut_by_acvm_type(
        &mut self,
        _index: Self::AcvmType,
        _value: Self::AcvmType,
        _lut: &mut <Self::Lookup as mpc_core::lut::LookupTableProvider<F>>::SecretSharedMap,
    ) -> std::io::Result<()> {
        panic!("write_lut_by_acvm_type: Operation atm not supported")
    }

    fn is_shared(a: &Self::AcvmType) -> bool {
        matches!(a, ShamirAcvmType::Shared(_))
    }

    fn get_shared(a: &Self::AcvmType) -> Option<Self::ArithmeticShare> {
        match a {
            ShamirAcvmType::Shared(shared) => Some(*shared),
            _ => None,
        }
    }

    fn get_public(a: &Self::AcvmType) -> Option<F> {
        match a {
            ShamirAcvmType::Public(public) => Some(*public),
            _ => None,
        }
    }

    fn open_many(&mut self, a: &[Self::ArithmeticShare]) -> std::io::Result<Vec<F>> {
        arithmetic::open_vec(a, &mut self.protocol)
    }

    fn promote_to_trivial_share(&mut self, public_value: F) -> Self::ArithmeticShare {
        arithmetic::promote_to_trivial_share(public_value)
    }

    fn promote_to_trivial_shares(&mut self, public_values: &[F]) -> Vec<Self::ArithmeticShare> {
        arithmetic::promote_to_trivial_shares(public_values)
    }

    fn decompose_arithmetic(
        &mut self,
        _input: Self::ArithmeticShare,
        _total_bit_size_per_field: usize,
        _decompose_bit_size: usize,
    ) -> std::io::Result<Vec<Self::ArithmeticShare>> {
        panic!("functionality decompose_arithmetic not feasible for Shamir")
    }

    fn sort(
        &mut self,
        _inputs: &[Self::ArithmeticShare],
        _bitsize: usize,
    ) -> std::io::Result<Vec<Self::ArithmeticShare>> {
        panic!("functionality sort not feasible for Shamir")
    }
}
