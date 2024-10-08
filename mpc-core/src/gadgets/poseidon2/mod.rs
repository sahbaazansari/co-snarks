pub mod bn254_t2;
pub mod plain;
pub mod shamir;

use ark_ff::PrimeField;
use eyre::Error;
use num_bigint::BigUint;
use num_traits::Num;

pub struct Poseidon2T2D5Params<F: PrimeField> {
    pub(crate) rounds_f_beginning: usize,
    pub(crate) rounds_f_end: usize,
    pub(crate) rounds_p: usize,
    pub(crate) round_constants_external: &'static Vec<[F; 2]>,
    pub(crate) round_constants_internal: &'static Vec<F>,
}

impl<F: PrimeField> Poseidon2T2D5Params<F> {
    pub(crate) fn new(
        rounds_f: usize,
        rounds_p: usize,
        round_constants_external: &'static Vec<[F; 2]>,
        round_constants_internal: &'static Vec<F>,
    ) -> Self {
        assert_eq!(rounds_f % 2, 0);
        assert_eq!(round_constants_external.len(), rounds_f);
        assert_eq!(round_constants_internal.len(), rounds_p);
        let rounds_f_beginning = rounds_f / 2;
        let rounds_f_end = rounds_f / 2;

        Self {
            rounds_f_beginning,
            rounds_f_end,
            rounds_p,
            round_constants_external,
            round_constants_internal,
        }
    }

    pub fn field_from_hex_string(str: &str) -> Result<F, Error> {
        let tmp = match str.strip_prefix("0x") {
            Some(t) => BigUint::from_str_radix(t, 16),
            None => BigUint::from_str_radix(str, 16),
        };

        Ok(tmp?.into())
    }
}

pub struct Poseidon2T2D5<F: PrimeField> {
    pub(crate) params: &'static Poseidon2T2D5Params<F>,
}

impl<F: PrimeField> Poseidon2T2D5<F> {
    pub fn new(params: &'static Poseidon2T2D5Params<F>) -> Self {
        Self { params }
    }
}
