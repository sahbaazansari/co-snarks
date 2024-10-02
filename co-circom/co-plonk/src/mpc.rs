use std::future::Future;

use ark_ec::pairing::Pairing;
use ark_poly::EvaluationDomain;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

pub(crate) mod plain;
pub(crate) mod rep3;
pub(crate) mod shamir;

pub use plain::PlainPlonkDriver;
pub use rep3::Rep3PlonkDriver;

type IoResult<T> = std::io::Result<T>;

pub trait CircomPlonkProver<P: Pairing> {
    type ArithmeticShare: CanonicalSerialize + CanonicalDeserialize + Copy + Clone + Default + Send;
    type PointShareG1: Send;
    type PointShareG2: Send;

    type PartyID: Send + Sync + Copy + std::fmt::Display;

    type IoContext;

    async fn close_network(self) -> IoResult<()>;

    fn debug_print(a: Self::ArithmeticShare);

    fn rand(&mut self) -> IoResult<Self::ArithmeticShare>;

    fn get_party_id(&self) -> Self::PartyID;

    /// Subtract the share b from the share a: \[c\] = \[a\] - \[b\]
    fn add(a: Self::ArithmeticShare, b: Self::ArithmeticShare) -> Self::ArithmeticShare;

    /// Add a public value a to the share b: \[c\] = a + \[b\]
    fn add_with_public(
        party_id: Self::PartyID,
        shared: Self::ArithmeticShare,
        public: P::ScalarField,
    ) -> Self::ArithmeticShare;

    /// Subtract the share b from the share a: \[c\] = \[a\] - \[b\]
    fn sub(a: Self::ArithmeticShare, b: Self::ArithmeticShare) -> Self::ArithmeticShare;

    /// Negates a vector of shared values: \[b\] = -\[a\] for every element in place.
    fn neg_vec_in_place(&mut self, a: &mut [Self::ArithmeticShare]);

    /// Multiply a share b by a public value a: c = a * \[b\].
    fn mul_with_public(
        shared: Self::ArithmeticShare,
        public: P::ScalarField,
    ) -> Self::ArithmeticShare;

    fn local_mul_vec(
        &mut self,
        a: &[Self::ArithmeticShare],
        b: &[Self::ArithmeticShare],
    ) -> Vec<P::ScalarField>;

    async fn io_round_mul_vec(
        &mut self,
        a: Vec<P::ScalarField>,
    ) -> IoResult<Vec<Self::ArithmeticShare>>;

    fn mul_vec(
        &mut self,
        a: &[Self::ArithmeticShare],
        b: &[Self::ArithmeticShare],
    ) -> impl Future<Output = IoResult<Vec<Self::ArithmeticShare>>>;

    fn mul_vecs(
        &mut self,
        a: &[Self::ArithmeticShare],
        b: &[Self::ArithmeticShare],
        c: &[Self::ArithmeticShare],
    ) -> impl Future<Output = IoResult<Vec<Self::ArithmeticShare>>>;

    /// Convenience method for \[a\] + \[b\] * \[c\]
    fn add_mul_vec(
        &mut self,
        a: &[Self::ArithmeticShare],
        b: &[Self::ArithmeticShare],
        c: &[Self::ArithmeticShare],
    ) -> impl Future<Output = IoResult<Vec<Self::ArithmeticShare>>>;

    /// Convenience method for \[a\] + \[b\] * c
    fn add_mul_public(
        &mut self,
        a: Self::ArithmeticShare,
        b: Self::ArithmeticShare,
        c: P::ScalarField,
    ) -> Self::ArithmeticShare {
        Self::add(a, Self::mul_with_public(b, c))
    }

    /// This function performs a multiplication directly followed by an opening. This safes one round of communication in some MPC protocols compared to calling `mul` and `open` separately.
    fn mul_open_vec(
        &mut self,
        a: &[Self::ArithmeticShare],
        b: &[Self::ArithmeticShare],
    ) -> impl Future<Output = IoResult<Vec<P::ScalarField>>>;

    /// Reconstructs many shared values: a = Open(\[a\]).
    fn open_vec(
        &mut self,
        a: &[Self::ArithmeticShare],
    ) -> impl Future<Output = IoResult<Vec<P::ScalarField>>>;

    /// Computes the inverse of many shared values: \[b\] = \[a\] ^ -1. Requires network communication.
    fn inv_vec(
        &mut self,
        a: &[Self::ArithmeticShare],
    ) -> impl Future<Output = IoResult<Vec<Self::ArithmeticShare>>>;

    /// Transforms a public value into a shared value: \[a\] = a.
    fn promote_to_trivial_share(
        party_id: Self::PartyID,
        public_value: P::ScalarField,
    ) -> Self::ArithmeticShare;

    /// Computes the FFT of a vector of shared field elements.
    fn fft<D: EvaluationDomain<P::ScalarField>>(
        data: &[Self::ArithmeticShare],
        domain: &D,
    ) -> Vec<Self::ArithmeticShare>;

    /// Computes the inverse FFT of a vector of shared field elements.
    fn ifft<D: EvaluationDomain<P::ScalarField>>(
        data: &[Self::ArithmeticShare],
        domain: &D,
    ) -> Vec<Self::ArithmeticShare>;

    /// Reconstructs many shared points: A = Open(\[A\]).
    fn open_point_g1(&mut self, a: Self::PointShareG1) -> impl Future<Output = IoResult<P::G1>>;
    fn open_point_vec_g1(
        &mut self,
        a: &[Self::PointShareG1],
    ) -> impl Future<Output = IoResult<Vec<P::G1>>>;

    // WE NEED THIS ALSO FOR GROTH16
    fn msm_public_points_g1(
        points: &[P::G1Affine],
        scalars: &[Self::ArithmeticShare],
    ) -> Self::PointShareG1;

    fn evaluate_poly_public(
        poly: Vec<Self::ArithmeticShare>,
        point: P::ScalarField,
    ) -> (Self::ArithmeticShare, Vec<Self::ArithmeticShare>);

    async fn array_prod_mul(
        io_context: &mut Self::IoContext,
        inv: bool,
        arr1: &[Self::ArithmeticShare],
        arr2: &[Self::ArithmeticShare],
        arr3: &[Self::ArithmeticShare],
    ) -> IoResult<Vec<Self::ArithmeticShare>>;

    async fn array_prod_mul2(
        &mut self,
        n1: &[Self::ArithmeticShare],
        n2: &[Self::ArithmeticShare],
        n3: &[Self::ArithmeticShare],
        d1: &[Self::ArithmeticShare],
        d2: &[Self::ArithmeticShare],
        d3: &[Self::ArithmeticShare],
    ) -> IoResult<(Vec<Self::ArithmeticShare>, Vec<Self::ArithmeticShare>)>;
}
