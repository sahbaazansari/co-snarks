use super::Poseidon2T2D5;
use crate::{
    protocols::rep3::{id::PartyID, network::Rep3Network, Rep3PrimeFieldShare, Rep3Protocol},
    traits::PrimeFieldMpcProtocol,
};
use ark_ff::PrimeField;

impl<F: PrimeField> Poseidon2T2D5<F> {
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
}
