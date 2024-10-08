use super::{Poseidon2T2D5, Precomputations};
use crate::{
    protocols::shamir::{
        fieldshare::ShamirPrimeFieldShare, network::ShamirNetwork, ShamirProtocol,
    },
    traits::PrimeFieldMpcProtocol,
};
use ark_ff::PrimeField;

impl<F: PrimeField> Poseidon2T2D5<F> {
    fn convert_shamir_mut<const T: usize>(
        state: &mut [ShamirPrimeFieldShare<F>; T],
    ) -> &mut [F; T] {
        // SAFETY: ShamirPrimeFieldShare has repr(transparent)
        unsafe { &mut *(state.as_mut() as *mut [ShamirPrimeFieldShare<F>] as *mut [F; T]) }
    }

    fn precompute_shamir<N: ShamirNetwork>(
        &self,
        driver: &mut ShamirProtocol<F, N>,
    ) -> std::io::Result<Precomputations<F>> {
        let num_sbox =
            (self.params.rounds_f_beginning + self.params.rounds_f_end) * 2 + self.params.rounds_p;

        let mut r = Vec::with_capacity(num_sbox);
        for _ in 0..num_sbox {
            r.push(driver.rand()?);
        }
        let r2 = driver.mul_many(&r, &r)?;
        let r4 = driver.mul_many(&r2, &r2)?;

        let mut lhs = Vec::with_capacity(num_sbox * 2);
        let mut rhs = Vec::with_capacity(num_sbox * 2);
        for (r, r2) in r.iter().cloned().zip(r2.iter().cloned()) {
            lhs.push(r);
            rhs.push(r2);
        }
        for (r, r4) in r.iter().cloned().zip(r4.iter().cloned()) {
            lhs.push(r);
            rhs.push(r4);
        }

        let mut r3 = driver.mul_many(&lhs, &rhs)?;
        let r5 = r3.split_off(num_sbox);

        Ok(Precomputations {
            r: ShamirPrimeFieldShare::convert_vec(r),
            r2: ShamirPrimeFieldShare::convert_vec(r2),
            r3: ShamirPrimeFieldShare::convert_vec(r3),
            r4: ShamirPrimeFieldShare::convert_vec(r4),
            r5: ShamirPrimeFieldShare::convert_vec(r5),
            offset: 0,
        })
    }

    fn sbox_shamir_precomp<N: ShamirNetwork>(
        input: &mut [F; 2],
        driver: &mut ShamirProtocol<F, N>,
        precomp: &mut Precomputations<F>,
    ) -> std::io::Result<()> {
        let (r, r2, r3, r4, r5) = precomp.get(precomp.offset);
        let (r_, r2_, r3_, r4_, r5_) = precomp.get(precomp.offset + 1);

        input[0] -= r;
        input[1] -= r_;

        let y = driver.open_many(ShamirPrimeFieldShare::convert_slice_rev(&*input))?;

        input[0] = Self::sbox_shamir_precomp_post(&y[0], r, r2, r3, r4, r5);
        input[1] = Self::sbox_shamir_precomp_post(&y[1], r_, r2_, r3_, r4_, r5_);
        precomp.offset += 2;

        Ok(())
    }

    fn single_sbox_shamir_precomp<N: ShamirNetwork>(
        input: &mut F,
        driver: &mut ShamirProtocol<F, N>,
        precomp: &mut Precomputations<F>,
    ) -> std::io::Result<()> {
        let (r, r2, r3, r4, r5) = precomp.get(precomp.offset);

        *input -= r;
        let y = driver.open(&ShamirPrimeFieldShare::new(*input))?;
        *input = Self::sbox_shamir_precomp_post(&y, r, r2, r3, r4, r5);
        precomp.offset += 1;

        Ok(())
    }

    fn sbox_shamir_precomp_post(y: &F, r: &F, r2: &F, r3: &F, r4: &F, r5: &F) -> F {
        let y2 = y.square();
        let y3 = y2 * y;
        let y4 = y2.square();
        let y5 = y4 * y;
        let five = F::from(5u64);
        let ten = F::from(10u64);

        let mut res = y5;
        res += y4 * r * five;
        res += y3 * r2 * ten;
        res += y2 * r3 * ten;
        res += *y * r4 * five;
        res += r5;
        res
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

    pub fn shamir_permutation_in_place_with_precomputation<N: ShamirNetwork>(
        &self,
        state: &mut [ShamirPrimeFieldShare<F>; 2],
        driver: &mut ShamirProtocol<F, N>,
    ) -> std::io::Result<()> {
        let mut precomp = self.precompute_shamir(driver)?;

        let state = Self::convert_shamir_mut(state);

        // Linear layer at beginning
        Self::matmul_external_plain(state);

        // First set of external rounds
        for r in 0..self.params.rounds_f_beginning {
            self.add_rc_external_plain(state, r);
            Self::sbox_shamir_precomp(state, driver, &mut precomp)?;
            Self::matmul_external_plain(state);
        }

        // Internal rounds
        for r in 0..self.params.rounds_p {
            self.add_rc_internal_plain(state, r);
            Self::single_sbox_shamir_precomp(&mut state[0], driver, &mut precomp)?;
            Self::matmul_internal_plain(state);
        }

        // Remaining external rounds
        for r in self.params.rounds_f_beginning
            ..self.params.rounds_f_beginning + self.params.rounds_f_end
        {
            self.add_rc_external_plain(state, r);
            Self::sbox_shamir_precomp(state, driver, &mut precomp)?;
            Self::matmul_external_plain(state);
        }

        debug_assert_eq!(precomp.offset, precomp.r.len());
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

    pub fn shamir_permutation_with_precomputation<N: ShamirNetwork>(
        &self,
        state: &[ShamirPrimeFieldShare<F>; 2],
        driver: &mut ShamirProtocol<F, N>,
    ) -> std::io::Result<[ShamirPrimeFieldShare<F>; 2]> {
        let mut state = state.to_owned();
        self.shamir_permutation_in_place_with_precomputation(&mut state, driver)?;
        Ok(state)
    }
}
