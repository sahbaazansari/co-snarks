use super::{Poseidon2T2D5, Precomputations};
use crate::{
    protocols::rep3::{id::PartyID, network::Rep3Network, Rep3PrimeFieldShare, Rep3Protocol},
    traits::PrimeFieldMpcProtocol,
};
use ark_ff::PrimeField;

impl<F: PrimeField> Poseidon2T2D5<F> {
    fn precompute_rep3<N: Rep3Network>(
        &self,
        num_poseidon: usize,
        driver: &mut Rep3Protocol<F, N>,
    ) -> std::io::Result<Precomputations<F>> {
        let num_sbox = ((self.params.rounds_f_beginning + self.params.rounds_f_end) * 2
            + self.params.rounds_p)
            * num_poseidon;

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
            r: r.into_iter().map(|x| x.a).collect(),
            r2: r2.into_iter().map(|x| x.a).collect(),
            r3: r3.into_iter().map(|x| x.a).collect(),
            r4: r4.into_iter().map(|x| x.a).collect(),
            r5: r5.into_iter().map(|x| x.a).collect(),
            offset: 0,
        })
    }

    fn matmul_external_rep3(input: &mut [Rep3PrimeFieldShare<F>; 2]) {
        // Matrix circ(2, 1)
        let sum = input[0].to_owned() + &input[1];

        input[0] += &sum;
        input[1] += &sum;
    }

    fn add_rc_external_rep3<N: Rep3Network>(
        &self,
        input: &mut [Rep3PrimeFieldShare<F>; 2],
        rc_offset: usize,
        driver: &mut Rep3Protocol<F, N>,
    ) {
        for (s, rc) in input
            .iter_mut()
            .zip(self.params.round_constants_external[rc_offset].iter())
        {
            *s = driver.add_with_public(rc, s);
        }
    }

    fn reshare_state_rep3<N: Rep3Network>(
        input: &mut [F; 2],
        driver: &mut Rep3Protocol<F, N>,
    ) -> std::io::Result<[Rep3PrimeFieldShare<F>; 2]> {
        // Reshare (with re-randomization):
        let rands = [driver.rand()?, driver.rand()?];
        input.iter_mut().zip(rands.iter()).for_each(|(x, r)| {
            *x += r.a - r.b;
        });
        driver.network.send_next_many(input)?;
        let b = driver.network.recv_prev_many()?;

        let shares = [
            Rep3PrimeFieldShare::new(input[0], b[0]),
            Rep3PrimeFieldShare::new(input[1], b[1]),
        ];

        Ok(shares)
    }

    fn reshare_state_rep3_packed<N: Rep3Network>(
        input: &mut [F],
        driver: &mut Rep3Protocol<F, N>,
    ) -> std::io::Result<Vec<[Rep3PrimeFieldShare<F>; 2]>> {
        for x in input.iter_mut() {
            let r = driver.rand()?;
            *x += r.a - r.b;
        }
        driver.network.send_next_many(input)?;
        let b = driver.network.recv_prev_many()?;

        let mut shares = Vec::with_capacity(input.len() / 2);
        for (x, b) in input.chunks_exact(2).zip(b.chunks_exact(2)) {
            let share = [
                Rep3PrimeFieldShare::new(x[0], b[0]),
                Rep3PrimeFieldShare::new(x[1], b[1]),
            ];
            shares.push(share);
        }

        Ok(shares)
    }

    fn sbox_rep3_precomp<N: Rep3Network>(
        input: &mut [F],
        driver: &mut Rep3Protocol<F, N>,
        precomp: &mut Precomputations<F>,
    ) -> std::io::Result<()> {
        for (i, inp) in input.iter_mut().enumerate() {
            *inp -= precomp.r[precomp.offset + i];
        }

        // Open
        driver.network.send_next_many(input)?;
        driver
            .network
            .send_many(driver.network.get_id().prev_id(), input)?;

        let b = driver.network.recv_prev_many::<F>()?;
        let c = driver
            .network
            .recv_many::<F>(driver.network.get_id().next_id())?;
        let mut y = b;
        for (y, (c, i)) in y.iter_mut().zip(c.into_iter().zip(input.iter())) {
            *y += c + i;
        }

        let id = driver.network.get_id();

        for (i, (inp, y)) in input.iter_mut().zip(y).enumerate() {
            let (r, r2, r3, r4, r5) = precomp.get(precomp.offset + i);

            *inp = Self::sbox_rep3_precomp_post(&y, r, r2, r3, r4, r5, &id);
        }

        precomp.offset += input.len();

        Ok(())
    }

    fn single_sbox_rep3_precomp_packed<N: Rep3Network>(
        input: &mut [F],
        driver: &mut Rep3Protocol<F, N>,
        precomp: &mut Precomputations<F>,
    ) -> std::io::Result<()> {
        let mut vec = input.iter().cloned().step_by(2).collect::<Vec<_>>();
        Self::sbox_rep3_precomp(&mut vec, driver, precomp)?;

        for (inp, r) in input.iter_mut().step_by(2).zip(vec) {
            *inp = r;
        }

        Ok(())
    }

    fn single_sbox_rep3_precomp<N: Rep3Network>(
        input: &mut F,
        driver: &mut Rep3Protocol<F, N>,
        precomp: &mut Precomputations<F>,
    ) -> std::io::Result<()> {
        let (r, r2, r3, r4, r5) = precomp.get(precomp.offset);

        *input -= r;

        // Open
        driver.network.send_next(input.to_owned())?;
        driver
            .network
            .send(driver.network.get_id().prev_id(), input.to_owned())?;

        let b = driver.network.recv_prev::<F>()?;
        let c = driver
            .network
            .recv::<F>(driver.network.get_id().next_id())?;
        let mut y = b;
        y += c + *input;

        *input = Self::sbox_rep3_precomp_post(&y, r, r2, r3, r4, r5, &driver.network.get_id());
        precomp.offset += 1;

        Ok(())
    }

    fn sbox_rep3_precomp_post(y: &F, r: &F, r2: &F, r3: &F, r4: &F, r5: &F, id: &PartyID) -> F {
        let y2 = y.square();
        let y3 = y2 * y;
        let y4 = y2.square();
        let five = F::from(5u64);
        let ten = F::from(10u64);

        let mut res = *r5;
        res += *y * r4 * five;
        res += y2 * r3 * ten;
        res += y3 * r2 * ten;
        res += y4 * r * five;

        if id == &PartyID::ID0 {
            let y5 = y4 * y;
            res += y5;
        }
        res
    }

    fn sbox_rep3<N: Rep3Network>(
        input: &mut [F; 2],
        driver: &mut Rep3Protocol<F, N>,
    ) -> std::io::Result<()> {
        let shares = Self::reshare_state_rep3(input, driver)?;
        *input = Self::sbox_rep3_first(&shares, driver)?;

        Ok(())
    }

    fn sbox_rep3_first<N: Rep3Network>(
        input: &[Rep3PrimeFieldShare<F>; 2],
        driver: &mut Rep3Protocol<F, N>,
    ) -> std::io::Result<[F; 2]> {
        // Square
        let sq: Vec<Rep3PrimeFieldShare<F>> = driver.mul_many(input, input)?;

        // Quad
        let qu = driver.mul_many(&sq, &sq)?;

        // Quint
        let res = [qu[0].to_owned() * &input[0], qu[1].to_owned() * &input[1]];

        Ok(res)
    }

    fn single_sbox_rep3<N: Rep3Network>(
        input: &mut F,
        driver: &mut Rep3Protocol<F, N>,
    ) -> std::io::Result<()> {
        // Reshare (with re-randomization):
        let rand = driver.rand()?;
        let input_a = input.to_owned() + rand.a - rand.b;
        driver.network.send_next(input_a.to_owned())?;
        let input_b = driver.network.recv_prev()?;
        let share = Rep3PrimeFieldShare::new(input_a, input_b);

        // Square
        let sq = driver.mul(&share, &share)?;

        // Quad
        let qu = driver.mul(&sq, &sq)?;

        // Quint
        *input = qu * share;

        Ok(())
    }

    pub fn rep3_permutation_in_place_with_precomputation_packed<N: Rep3Network>(
        &self,
        state_: &mut Vec<[Rep3PrimeFieldShare<F>; 2]>,
        driver: &mut Rep3Protocol<F, N>,
    ) -> std::io::Result<()> {
        let id = driver.network.get_id();
        let num_inputs = state_.len();

        // Just use a
        let mut state = Vec::with_capacity(num_inputs * 2);
        let mut precomp = self.precompute_rep3(num_inputs, driver)?;

        // Linear layer at beginning
        for s_ in state_.iter() {
            let mut s = [s_[0].a, s_[1].a];
            Self::matmul_external_plain(&mut s);
            state.push(s[0]);
            state.push(s[1]);
        }

        // First set of external rounds
        for r in 0..self.params.rounds_f_beginning {
            if id == PartyID::ID0 {
                for s in state.chunks_exact_mut(2) {
                    self.add_rc_external_plain(s.try_into().unwrap(), r);
                }
            }
            Self::sbox_rep3_precomp(&mut state, driver, &mut precomp)?;
            for s in state.chunks_exact_mut(2) {
                Self::matmul_external_plain(s.try_into().unwrap());
            }
        }

        // Internal rounds
        for r in 0..self.params.rounds_p {
            if id == PartyID::ID0 {
                for s in state.chunks_exact_mut(2) {
                    self.add_rc_internal_plain(s.try_into().unwrap(), r);
                }
            }
            Self::single_sbox_rep3_precomp_packed(&mut state, driver, &mut precomp)?;
            for s in state.chunks_exact_mut(2) {
                Self::matmul_internal_plain(s.try_into().unwrap());
            }
        }

        // Remaining external rounds
        for r in self.params.rounds_f_beginning
            ..self.params.rounds_f_beginning + self.params.rounds_f_end
        {
            if id == PartyID::ID0 {
                for s in state.chunks_exact_mut(2) {
                    self.add_rc_external_plain(s.try_into().unwrap(), r);
                }
            }
            Self::sbox_rep3_precomp(&mut state, driver, &mut precomp)?;
            for s in state.chunks_exact_mut(2) {
                Self::matmul_external_plain(s.try_into().unwrap());
            }
        }

        *state_ = Self::reshare_state_rep3_packed(&mut state, driver)?;

        debug_assert_eq!(precomp.offset, precomp.r.len());
        Ok(())
    }

    pub fn rep3_permutation_in_place_with_precomputation<N: Rep3Network>(
        &self,
        state_: &mut [Rep3PrimeFieldShare<F>; 2],
        driver: &mut Rep3Protocol<F, N>,
    ) -> std::io::Result<()> {
        let id = driver.network.get_id();

        // Just use a
        let mut state = [state_[0].a, state_[1].a];
        let mut precomp = self.precompute_rep3(1, driver)?;

        // Linear layer at beginning
        Self::matmul_external_plain(&mut state);

        // First set of external rounds
        for r in 0..self.params.rounds_f_beginning {
            if id == PartyID::ID0 {
                self.add_rc_external_plain(&mut state, r);
            }
            Self::sbox_rep3_precomp(&mut state, driver, &mut precomp)?;
            Self::matmul_external_plain(&mut state);
        }

        // Internal rounds
        for r in 0..self.params.rounds_p {
            if id == PartyID::ID0 {
                self.add_rc_internal_plain(&mut state, r);
            }
            Self::single_sbox_rep3_precomp(&mut state[0], driver, &mut precomp)?;
            Self::matmul_internal_plain(&mut state);
        }

        // Remaining external rounds
        for r in self.params.rounds_f_beginning
            ..self.params.rounds_f_beginning + self.params.rounds_f_end
        {
            if id == PartyID::ID0 {
                self.add_rc_external_plain(&mut state, r);
            }
            Self::sbox_rep3_precomp(&mut state, driver, &mut precomp)?;
            Self::matmul_external_plain(&mut state);
        }

        *state_ = Self::reshare_state_rep3(&mut state, driver)?;

        debug_assert_eq!(precomp.offset, precomp.r.len());
        Ok(())
    }

    pub fn rep3_permutation_in_place<N: Rep3Network>(
        &self,
        state_: &mut [Rep3PrimeFieldShare<F>; 2],
        driver: &mut Rep3Protocol<F, N>,
    ) -> std::io::Result<()> {
        let id = driver.network.get_id();

        // Linear layer at beginning
        Self::matmul_external_rep3(state_);

        // First round:
        self.add_rc_external_rep3(state_, 0, driver);
        let mut state = Self::sbox_rep3_first(state_, driver)?;
        Self::matmul_external_plain(&mut state);

        // First set of external rounds
        for r in 1..self.params.rounds_f_beginning {
            if id == PartyID::ID0 {
                self.add_rc_external_plain(&mut state, r);
            }
            Self::sbox_rep3(&mut state, driver)?;
            Self::matmul_external_plain(&mut state);
        }

        // Internal rounds
        for r in 0..self.params.rounds_p {
            if id == PartyID::ID0 {
                self.add_rc_internal_plain(&mut state, r);
            }
            Self::single_sbox_rep3(&mut state[0], driver)?;
            Self::matmul_internal_plain(&mut state);
        }

        // Remaining external rounds
        for r in self.params.rounds_f_beginning
            ..self.params.rounds_f_beginning + self.params.rounds_f_end
        {
            if id == PartyID::ID0 {
                self.add_rc_external_plain(&mut state, r);
            }
            Self::sbox_rep3(&mut state, driver)?;
            Self::matmul_external_plain(&mut state);
        }

        *state_ = Self::reshare_state_rep3(&mut state, driver)?;

        Ok(())
    }

    pub fn rep3_permutation<N: Rep3Network>(
        &self,
        state: &[Rep3PrimeFieldShare<F>; 2],
        driver: &mut Rep3Protocol<F, N>,
    ) -> std::io::Result<[Rep3PrimeFieldShare<F>; 2]> {
        let mut state = state.to_owned();
        self.rep3_permutation_in_place(&mut state, driver)?;
        Ok(state)
    }

    pub fn rep3_permutation_with_precomputation<N: Rep3Network>(
        &self,
        state: &[Rep3PrimeFieldShare<F>; 2],
        driver: &mut Rep3Protocol<F, N>,
    ) -> std::io::Result<[Rep3PrimeFieldShare<F>; 2]> {
        let mut state = state.to_owned();
        self.rep3_permutation_in_place_with_precomputation(&mut state, driver)?;
        Ok(state)
    }
}
