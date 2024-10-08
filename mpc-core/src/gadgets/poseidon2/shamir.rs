use super::Poseidon2T2D5;
use crate::protocols::shamir::{
    fieldshare::ShamirPrimeFieldShare, network::ShamirNetwork, ShamirProtocol,
};
use ark_ff::PrimeField;

impl<F: PrimeField> Poseidon2T2D5<F> {
    fn convert_shamir_mut<const T: usize>(
        state: &mut [ShamirPrimeFieldShare<F>; T],
    ) -> &mut [F; T] {
        // SAFETY: ShamirPrimeFieldShare has repr(transparent)
        unsafe { &mut *(state.as_mut() as *mut [ShamirPrimeFieldShare<F>] as *mut [F; T]) }
    }

    fn sbox_shamir<N: ShamirNetwork>(
        input: &mut [F; 2],
        driver: &mut ShamirProtocol<F, N>,
    ) -> std::io::Result<()> {
        // Square
        let mut sq = driver
            .degree_reduce_vec(vec![input[0].square(), input[1].square()])?
            .a;

        // Quad
        sq.iter_mut().for_each(|x| {
            x.square_in_place();
        });
        let mut qu = driver.degree_reduce_vec(sq)?.a;

        // Quint
        qu.iter_mut().zip(input.iter()).for_each(|(x, y)| *x *= y);
        let res = driver.degree_reduce_vec(qu)?.a;

        input.clone_from_slice(&res);
        Ok(())
    }

    fn single_sbox_shamir<N: ShamirNetwork>(
        input: &mut F,
        driver: &mut ShamirProtocol<F, N>,
    ) -> std::io::Result<()> {
        // Square
        let mut sq = driver.degree_reduce(input.square())?.a;

        // Quad
        sq.square_in_place();
        let mut qu = driver.degree_reduce(sq)?.a;

        // Quint
        qu *= &*input;
        *input = driver.degree_reduce(qu)?.a;

        Ok(())
    }

    pub fn shamir_permutation_in_place<N: ShamirNetwork>(
        &self,
        state: &mut [ShamirPrimeFieldShare<F>; 2],
        driver: &mut ShamirProtocol<F, N>,
    ) -> std::io::Result<()> {
        let state = Self::convert_shamir_mut(state);

        // Linear layer at beginning
        Self::matmul_external_plain(state);

        // First set of external rounds
        for r in 0..self.params.rounds_f_beginning {
            self.add_rc_external_plain(state, r);
            Self::sbox_shamir(state, driver)?;
            Self::matmul_external_plain(state);
        }

        // Internal rounds
        for r in 0..self.params.rounds_p {
            self.add_rc_internal_plain(state, r);
            Self::single_sbox_shamir(&mut state[0], driver)?;
            Self::matmul_internal_plain(state);
        }

        // Remaining external rounds
        for r in self.params.rounds_f_beginning
            ..self.params.rounds_f_beginning + self.params.rounds_f_end
        {
            self.add_rc_external_plain(state, r);
            Self::sbox_shamir(state, driver)?;
            Self::matmul_external_plain(state);
        }

        Ok(())
    }

    pub fn shamir_permutation<N: ShamirNetwork>(
        &self,
        state: &[ShamirPrimeFieldShare<F>; 2],
        driver: &mut ShamirProtocol<F, N>,
    ) -> std::io::Result<[ShamirPrimeFieldShare<F>; 2]> {
        let mut state = state.to_owned();
        self.shamir_permutation_in_place(&mut state, driver)?;
        Ok(state)
    }
}
