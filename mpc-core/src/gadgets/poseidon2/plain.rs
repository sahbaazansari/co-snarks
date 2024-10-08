use super::Poseidon2T2D5;
use ark_ff::PrimeField;

impl<F: PrimeField> Poseidon2T2D5<F> {
    pub(crate) fn matmul_external_plain(input: &mut [F; 2]) {
        // Matrix circ(2, 1)
        let sum = input[0] + input[1];

        input[0] += sum;
        input[1] += sum;
    }

    pub(crate) fn matmul_internal_plain(input: &mut [F; 2]) {
        // TODO poseidon 2 uses [[2, 1], [1, 3]], even though circ(2, 1) should be secure
        // Self::matmul_external(input, driver);

        // Matrix [[2, 1], [1, 3]]
        let sum = input[0] + input[1];

        input[0] += sum;
        input[1] = sum + input[1] + input[1];
    }

    pub(crate) fn add_rc_external_plain(&self, input: &mut [F; 2], rc_offset: usize) {
        for (s, rc) in input
            .iter_mut()
            .zip(self.params.round_constants_external[rc_offset].iter())
        {
            *s += rc;
        }
    }

    pub(crate) fn add_rc_internal_plain(&self, input: &mut [F; 2], rc_offset: usize) {
        input[0] += &self.params.round_constants_internal[rc_offset];
    }

    fn sbox_plain(input: &mut [F; 2]) {
        input.iter_mut().for_each(Self::single_sbox_plain);
    }

    fn single_sbox_plain(input: &mut F) {
        let input2 = input.square();
        let input4 = input2.square();
        *input *= input4;
    }

    pub fn plain_permutation_in_place(&self, state: &mut [F; 2]) {
        // Linear layer at beginning
        Self::matmul_external_plain(state);

        // First set of external rounds
        for r in 0..self.params.rounds_f_beginning {
            self.add_rc_external_plain(state, r);
            Self::sbox_plain(state);
            Self::matmul_external_plain(state);
        }

        // Internal rounds
        for r in 0..self.params.rounds_p {
            self.add_rc_internal_plain(state, r);
            Self::single_sbox_plain(&mut state[0]);
            Self::matmul_internal_plain(state);
        }

        // Remaining external rounds
        for r in self.params.rounds_f_beginning
            ..self.params.rounds_f_beginning + self.params.rounds_f_end
        {
            self.add_rc_external_plain(state, r);
            Self::sbox_plain(state);
            Self::matmul_external_plain(state);
        }
    }

    pub fn plain_permutation(&self, state: &[F; 2]) -> [F; 2] {
        let mut state = state.to_owned();
        self.plain_permutation_in_place(&mut state);
        state
    }
}

#[cfg(test)]
mod poseidon2_gadget_plain {
    use super::*;
    use crate::gadgets::poseidon2::{bn254_t2::POSEIDON2_BN254_T2_PARAMS, Poseidon2T2D5Params};
    use ark_ff::PrimeField;
    use rand::thread_rng;

    const TESTRUNS: usize = 10;

    fn poseidon2_kat<F: PrimeField>(
        params: &'static Poseidon2T2D5Params<F>,
        input: &[F; 2],
        expected: &[F; 2],
    ) {
        let poseidon2 = Poseidon2T2D5::new(params);
        let result = poseidon2.plain_permutation(input);
        assert_eq!(&result, expected);
    }

    fn poseidon2_consistent_perm<F: PrimeField>(params: &'static Poseidon2T2D5Params<F>) {
        let mut rng = &mut thread_rng();
        let input1: Vec<F> = (0..2).map(|_| F::rand(&mut rng)).collect();
        let mut input2 = input1.clone();
        input2.rotate_right(1);

        let poseidon2 = Poseidon2T2D5::new(params);
        let perm1 = poseidon2.plain_permutation(input1.as_slice().try_into().unwrap());
        let perm2 = poseidon2.plain_permutation(&input1.try_into().unwrap());
        let perm3 = poseidon2.plain_permutation(&input2.try_into().unwrap());

        assert_eq!(perm1, perm2);
        assert_ne!(perm1, perm3);
    }

    #[test]
    fn posedon2_bn254_consistent_perm() {
        for _ in 0..TESTRUNS {
            poseidon2_consistent_perm(&POSEIDON2_BN254_T2_PARAMS);
        }
    }

    #[test]
    fn posedon2_bn254_kat1() {
        let input = [ark_bn254::Fr::from(0u64), ark_bn254::Fr::from(1u64)];
        let expected = [
            Poseidon2T2D5Params::field_from_hex_string(
                "0x1d01e56f49579cec72319e145f06f6177f6c5253206e78c2689781452a31878b",
            )
            .unwrap(),
            Poseidon2T2D5Params::field_from_hex_string(
                "0x0d189ec589c41b8cffa88cfc523618a055abe8192c70f75aa72fc514560f6c61",
            )
            .unwrap(),
        ];

        poseidon2_kat(&POSEIDON2_BN254_T2_PARAMS, &input, &expected);
    }
}
