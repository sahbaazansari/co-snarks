pub mod file_utils;
use acir::{
    acir_field::GenericFieldElement,
    native_types::{WitnessMap, WitnessStack},
};
use ark_ec::pairing::Pairing;
use ark_ff::PrimeField;
use clap::{Args, ValueEnum};
use co_acvm::{
    solver::{partial_abi::PublicMarker, Rep3CoSolver},
    Rep3AcvmType, ShamirAcvmType,
};
use color_eyre::eyre::{eyre, Context};
use figment::{
    providers::{Env, Format, Serialized, Toml},
    Figment,
};
use mpc_core::protocols::{
    rep3::{
        self,
        network::{Rep3MpcNet, Rep3Network},
    },
    shamir,
};
use mpc_net::config::NetworkConfigFile;
use noirc_abi::Abi;
use rand::{CryptoRng, Rng};
use serde::{Deserialize, Serialize};
use std::{array, collections::BTreeMap, fs::File, io::Write, path::PathBuf};

#[derive(Clone, Debug)]
pub enum PubShared<F: Clone> {
    Public(F),
    Shared(F),
}

impl<F: Clone> PubShared<F> {
    pub fn from_shared(f: F) -> Self {
        Self::Shared(f)
    }

    pub fn set_public(&mut self) {
        if let Self::Shared(ref mut f) = self {
            *self = Self::Public(f.clone());
        }
    }
}

/// An enum representing the MPC protocol to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, ValueEnum)]
#[clap(rename_all = "UPPER")]
pub enum MPCProtocol {
    /// A protocol based on the Replicated Secret Sharing Scheme for 3 parties.
    /// For more information see <https://eprint.iacr.org/2018/403.pdf>.
    REP3,
    /// A protocol based on Shamir Secret Sharing Scheme for n parties.
    /// For more information see <https://iacr.org/archive/crypto2007/46220565/46220565.pdf>.
    SHAMIR,
}
/// An enum representing the transcript hasher to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, ValueEnum)]
#[clap(rename_all = "UPPER")]
pub enum TranscriptHash {
    /// The Poseidon2 sponge hash function
    POSEIDON,
    // The Keccak256 hash function
    KECCAK,
}

impl std::fmt::Display for MPCProtocol {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MPCProtocol::REP3 => write!(f, "REP3"),
            MPCProtocol::SHAMIR => write!(f, "SHAMIR"),
        }
    }
}
impl std::fmt::Display for TranscriptHash {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TranscriptHash::POSEIDON => write!(f, "POSEIDON"),
            TranscriptHash::KECCAK => write!(f, "KECCAK"),
        }
    }
}

/// Cli arguments for `split_witness`
#[derive(Debug, Default, Serialize, Args)]
pub struct SplitWitnessCli {
    /// The path to the config file
    #[arg(long)]
    #[serde(skip_serializing_if = "::std::option::Option::is_none")]
    pub config: Option<PathBuf>,
    /// The path to the input witness file generated by Noir
    #[arg(long)]
    #[serde(skip_serializing_if = "::std::option::Option::is_none")]
    pub witness: Option<PathBuf>,
    /// The path to the circuit file, generated by Noir
    #[arg(long)]
    #[serde(skip_serializing_if = "::std::option::Option::is_none")]
    pub circuit: Option<PathBuf>,
    /// The MPC protocol to be used
    #[arg(long, value_enum)]
    #[serde(skip_serializing_if = "::std::option::Option::is_none")]
    pub protocol: Option<MPCProtocol>,
    /// The path to the (existing) output directory
    #[arg(long)]
    #[serde(skip_serializing_if = "::std::option::Option::is_none")]
    pub out_dir: Option<PathBuf>,
    /// The threshold of tolerated colluding parties
    #[arg(short, long, default_value_t = 1)]
    pub threshold: usize,
    /// The number of parties
    #[arg(short, long, default_value_t = 3)]
    pub num_parties: usize,
}

/// Config for `split_witness`
#[derive(Debug, Deserialize)]
pub struct SplitWitnessConfig {
    /// The path to the input witness file generated by Circom
    pub witness: PathBuf,
    /// The path to the circuit file, generated by Noir
    pub circuit: PathBuf,
    /// The MPC protocol to be used
    pub protocol: MPCProtocol,
    /// The path to the (existing) output directory
    pub out_dir: PathBuf,
    /// The threshold of tolerated colluding parties
    pub threshold: usize,
    /// The number of parties
    pub num_parties: usize,
}

/// Cli arguments for `split_input`
#[derive(Debug, Default, Clone, Serialize, Args)]
pub struct SplitInputCli {
    /// The path to the config file
    #[arg(long)]
    #[serde(skip_serializing_if = "::std::option::Option::is_none")]
    pub config: Option<PathBuf>,
    /// The path to the input JSON file
    #[arg(long)]
    #[serde(skip_serializing_if = "::std::option::Option::is_none")]
    pub input: Option<PathBuf>,
    /// The path to the circuit file
    #[arg(long)]
    #[serde(skip_serializing_if = "::std::option::Option::is_none")]
    pub circuit: Option<String>,
    /// The MPC protocol to be used
    #[arg(long, value_enum)]
    #[serde(skip_serializing_if = "::std::option::Option::is_none")]
    pub protocol: Option<MPCProtocol>,
    /// The path to the (existing) output directory
    #[arg(long)]
    #[serde(skip_serializing_if = "::std::option::Option::is_none")]
    pub out_dir: Option<PathBuf>,
}

/// Config for `split_input`
#[derive(Debug, Clone, Deserialize)]
pub struct SplitInputConfig {
    /// The path to the input JSON file
    pub input: PathBuf,
    /// The path to the circuit file
    pub circuit: String,
    /// The MPC protocol to be used
    pub protocol: MPCProtocol,
    /// The path to the (existing) output directory
    pub out_dir: PathBuf,
}

/// Cli arguments for `split_proving_key`
#[derive(Debug, Default, Serialize, Args)]
pub struct SplitProvingKeyCli {
    /// The path to the config file
    #[arg(long)]
    #[serde(skip_serializing_if = "::std::option::Option::is_none")]
    pub config: Option<PathBuf>,
    /// The path to the input witness file generated by Noir
    #[arg(long)]
    #[serde(skip_serializing_if = "::std::option::Option::is_none")]
    pub witness: Option<PathBuf>,
    /// The path to the circuit file, generated by Noir
    #[arg(long)]
    #[serde(skip_serializing_if = "::std::option::Option::is_none")]
    pub circuit: Option<PathBuf>,
    /// The path to the prover crs file
    #[arg(long)]
    #[serde(skip_serializing_if = "::std::option::Option::is_none")]
    pub crs: Option<PathBuf>,
    /// The MPC protocol to be used
    #[arg(long, value_enum)]
    #[serde(skip_serializing_if = "::std::option::Option::is_none")]
    pub protocol: Option<MPCProtocol>,
    /// The path to the (existing) output directory
    #[arg(long)]
    #[serde(skip_serializing_if = "::std::option::Option::is_none")]
    pub out_dir: Option<PathBuf>,
    /// The threshold of tolerated colluding parties
    #[arg(short, long, default_value_t = 1)]
    pub threshold: usize,
    /// The number of parties
    #[arg(short, long, default_value_t = 3)]
    pub num_parties: usize,
    /// Generate a recursive friendly proof
    #[arg(long)]
    pub recursive: bool,
}

/// Config for `split_proving_key`
#[derive(Debug, Deserialize)]
pub struct SplitProvingKeyConfig {
    /// The path to the input witness file generated by Circom
    pub witness: PathBuf,
    /// The path to the circuit file, generated by Noir
    pub circuit: PathBuf,
    /// The path to the prover crs file
    pub crs: PathBuf,
    /// The MPC protocol to be used
    pub protocol: MPCProtocol,
    /// The path to the (existing) output directory
    pub out_dir: PathBuf,
    /// The threshold of tolerated colluding parties
    pub threshold: usize,
    /// The number of parties
    pub num_parties: usize,
    /// Whether to generate a recursive friendly proof
    pub recursive: bool,
}

/// Cli arguments for `merge_input_shares`
#[derive(Debug, Default, Serialize, Args)]
pub struct MergeInputSharesCli {
    /// The path to the config file
    #[arg(long)]
    #[serde(skip_serializing_if = "::std::option::Option::is_none")]
    pub config: Option<PathBuf>,
    /// The path to the input JSON file
    #[arg(long)]
    pub inputs: Vec<PathBuf>,
    /// The MPC protocol to be used
    #[arg(long, value_enum)]
    #[serde(skip_serializing_if = "::std::option::Option::is_none")]
    pub protocol: Option<MPCProtocol>,
    /// The output file where the merged input share is written to
    #[arg(long)]
    #[serde(skip_serializing_if = "::std::option::Option::is_none")]
    pub out: Option<PathBuf>,
}

/// Config for `merge_input_shares`
#[derive(Debug, Deserialize)]
pub struct MergeInputSharesConfig {
    /// The path to the input JSON file
    pub inputs: Vec<PathBuf>,
    /// The MPC protocol to be used
    pub protocol: MPCProtocol,
    /// The output file where the merged input share is written to
    pub out: PathBuf,
}

/// Cli arguments for `generate_witness`
#[derive(Debug, Default, Serialize, Args)]
pub struct GenerateWitnessCli {
    /// The path to the config file
    #[arg(long)]
    #[serde(skip_serializing_if = "::std::option::Option::is_none")]
    pub config: Option<PathBuf>,
    /// The path to the input share file
    #[arg(long)]
    #[serde(skip_serializing_if = "::std::option::Option::is_none")]
    pub input: Option<PathBuf>,
    /// The path to the circuit file
    #[arg(long)]
    #[serde(skip_serializing_if = "::std::option::Option::is_none")]
    pub circuit: Option<String>,
    /// The MPC protocol to be used
    #[arg(long, value_enum)]
    #[serde(skip_serializing_if = "::std::option::Option::is_none")]
    pub protocol: Option<MPCProtocol>,
    /// The output file where the final witness share is written to
    #[arg(long)]
    #[serde(skip_serializing_if = "::std::option::Option::is_none")]
    pub out: Option<PathBuf>,
}

/// Config for `generate_witness`
#[derive(Debug, Deserialize)]
pub struct GenerateWitnessConfig {
    /// The path to the input share file
    pub input: PathBuf,
    /// The path to the circuit file
    pub circuit: String,
    /// The MPC protocol to be used
    pub protocol: MPCProtocol,
    /// The output file where the final witness share is written to
    pub out: PathBuf,
    /// Network config
    pub network: NetworkConfigFile,
}

/// Cli arguments for `translate_witness`
#[derive(Debug, Serialize, Args)]
pub struct TranslateWitnessCli {
    /// The path to the config file
    #[arg(long)]
    #[serde(skip_serializing_if = "::std::option::Option::is_none")]
    pub config: Option<PathBuf>,
    /// The path to the witness share file
    #[arg(long)]
    #[serde(skip_serializing_if = "::std::option::Option::is_none")]
    pub witness: Option<PathBuf>,
    /// The MPC protocol that was used for the witness generation
    #[arg(long, value_enum)]
    #[serde(skip_serializing_if = "::std::option::Option::is_none")]
    pub src_protocol: Option<MPCProtocol>,
    /// The MPC protocol to be used for the proof generation
    #[arg(long, value_enum)]
    #[serde(skip_serializing_if = "::std::option::Option::is_none")]
    pub target_protocol: Option<MPCProtocol>,
    /// The output file where the final witness share is written to
    #[arg(long)]
    #[serde(skip_serializing_if = "::std::option::Option::is_none")]
    pub out: Option<PathBuf>,
}

/// Config for `translate_witness`
#[derive(Debug, Deserialize)]
pub struct TranslateWitnessConfig {
    /// The path to the witness share file
    pub witness: PathBuf,
    /// The MPC protocol that was used for the witness generation
    pub src_protocol: MPCProtocol,
    /// The MPC protocol to be used for the proof generation
    pub target_protocol: MPCProtocol,
    /// The output file where the final witness share is written to
    pub out: PathBuf,
    /// Network config
    pub network: NetworkConfigFile,
}

/// Cli arguments for `translate_witness`
#[derive(Debug, Serialize, Args)]
pub struct TranslateProvingKeyCli {
    /// The path to the config file
    #[arg(long)]
    #[serde(skip_serializing_if = "::std::option::Option::is_none")]
    pub config: Option<PathBuf>,
    /// The path to the proving_key share file
    #[arg(long)]
    #[serde(skip_serializing_if = "::std::option::Option::is_none")]
    pub proving_key: Option<PathBuf>,
    /// The MPC protocol that was used for the witness generation
    #[arg(long, value_enum)]
    #[serde(skip_serializing_if = "::std::option::Option::is_none")]
    pub src_protocol: Option<MPCProtocol>,
    /// The MPC protocol to be used for the proof generation
    #[arg(long, value_enum)]
    #[serde(skip_serializing_if = "::std::option::Option::is_none")]
    pub target_protocol: Option<MPCProtocol>,
    /// The output file where the final witness share is written to
    #[arg(long)]
    #[serde(skip_serializing_if = "::std::option::Option::is_none")]
    pub out: Option<PathBuf>,
}

/// Config for `translate_witness`
#[derive(Debug, Deserialize)]
pub struct TranslateProvingKeyConfig {
    /// The path to the proving_key share file
    pub proving_key: PathBuf,
    /// The MPC protocol that was used for the witness generation
    pub src_protocol: MPCProtocol,
    /// The MPC protocol to be used for the proof generation
    pub target_protocol: MPCProtocol,
    /// The output file where the final witness share is written to
    pub out: PathBuf,
    /// Network config
    pub network: NetworkConfigFile,
}

/// Cli arguments for `build_proving_key`
#[derive(Debug, Default, Serialize, Args)]
pub struct BuildProvingKeyCLi {
    /// The path to the config file
    #[arg(long)]
    #[serde(skip_serializing_if = "::std::option::Option::is_none")]
    pub config: Option<PathBuf>,
    /// The path to the witness share file
    #[arg(long)]
    #[serde(skip_serializing_if = "::std::option::Option::is_none")]
    pub witness: Option<PathBuf>,
    /// The path to the circuit file, generated by Noir
    #[arg(long)]
    #[serde(skip_serializing_if = "::std::option::Option::is_none")]
    pub circuit: Option<PathBuf>,
    /// The path to the prover crs file
    #[arg(long)]
    #[serde(skip_serializing_if = "::std::option::Option::is_none")]
    pub crs: Option<PathBuf>,
    /// The MPC protocol to be used
    #[arg(long, value_enum)]
    #[serde(skip_serializing_if = "::std::option::Option::is_none")]
    pub protocol: Option<MPCProtocol>,
    /// The output file where the final proving key is written to.
    #[arg(long)]
    #[serde(skip_serializing_if = "::std::option::Option::is_none")]
    pub out: Option<PathBuf>,
    /// The threshold of tolerated colluding parties
    #[arg(short, long, default_value_t = 1)]
    pub threshold: usize,
    /// Generate a recursive friendly proof
    #[arg(long)]
    pub recursive: bool,
}

/// Config for `build_proving_key`
#[derive(Debug, Deserialize)]
pub struct BuildProvingKeyConfig {
    /// The path to the witness share file
    pub witness: PathBuf,
    /// The path to the circuit file, generated by Noir
    pub circuit: PathBuf,
    /// The path to the prover crs file
    pub crs: PathBuf,
    /// The MPC protocol to be used
    pub protocol: MPCProtocol,
    /// The output file where the final proving key is written to.
    pub out: PathBuf,
    /// The threshold of tolerated colluding parties
    pub threshold: usize,
    /// Network config
    pub network: NetworkConfigFile,
    /// Whether to generate a recursive friendly proof
    pub recursive: bool,
}

/// Cli arguments for `generate_proof`
#[derive(Debug, Serialize, Args)]
pub struct GenerateProofCli {
    /// The path to the config file
    #[arg(long)]
    #[serde(skip_serializing_if = "::std::option::Option::is_none")]
    pub config: Option<PathBuf>,
    /// The path to the shared proving_key file
    #[arg(long)]
    #[serde(skip_serializing_if = "::std::option::Option::is_none")]
    pub proving_key: Option<PathBuf>,
    /// The MPC protocol to be used
    #[arg(long, value_enum)]
    #[serde(skip_serializing_if = "::std::option::Option::is_none")]
    pub protocol: Option<MPCProtocol>,
    /// The transcript hasher to be used
    #[arg(long, value_enum)]
    #[serde(skip_serializing_if = "::std::option::Option::is_none")]
    pub hasher: Option<TranscriptHash>,
    /// The output file where the final proof is written to. If not passed, this party will not write the proof to a file.
    #[arg(long)]
    #[serde(skip_serializing_if = "::std::option::Option::is_none")]
    pub out: Option<PathBuf>,
    /// The output JSON file where the public inputs are written to. If not passed, this party will not write the public inputs to a file.
    #[arg(long)]
    #[serde(skip_serializing_if = "::std::option::Option::is_none")]
    pub public_input: Option<PathBuf>,
    /// The threshold of tolerated colluding parties
    #[arg(short, long, default_value_t = 1)]
    pub threshold: usize,
}

/// Config for `generate_proof`
#[derive(Debug, Deserialize)]
pub struct GenerateProofConfig {
    /// The path to the witness share file
    pub proving_key: PathBuf,
    /// The MPC protocol to be used
    pub protocol: MPCProtocol,
    /// The transcript hasher to be used
    pub hasher: TranscriptHash,
    /// The output file where the final proof is written to. If not passed, this party will not write the proof to a file.
    pub out: Option<PathBuf>,
    /// The output JSON file where the public inputs are written to. If not passed, this party will not write the public inputs to a file.
    pub public_input: Option<PathBuf>,
    /// The threshold of tolerated colluding parties
    pub threshold: usize,
    /// Network config
    pub network: NetworkConfigFile,
}

/// Cli arguments for `build_and_generate_proof`
#[derive(Debug, Serialize, Args)]
pub struct BuildAndGenerateProofCli {
    /// The path to the config file
    #[arg(long)]
    #[serde(skip_serializing_if = "::std::option::Option::is_none")]
    pub config: Option<PathBuf>,
    /// The path to the witness share file
    #[arg(long)]
    #[serde(skip_serializing_if = "::std::option::Option::is_none")]
    pub witness: Option<PathBuf>,
    /// The path to the circuit file, generated by Noir
    #[arg(long)]
    #[serde(skip_serializing_if = "::std::option::Option::is_none")]
    pub circuit: Option<PathBuf>,
    /// The path to the prover crs file
    #[arg(long)]
    #[serde(skip_serializing_if = "::std::option::Option::is_none")]
    pub crs: Option<PathBuf>,
    /// The MPC protocol to be used
    #[arg(long, value_enum)]
    #[serde(skip_serializing_if = "::std::option::Option::is_none")]
    pub protocol: Option<MPCProtocol>,
    /// The transcript hasher to be used
    #[arg(long, value_enum)]
    #[serde(skip_serializing_if = "::std::option::Option::is_none")]
    pub hasher: Option<TranscriptHash>,
    /// The output file where the final proof is written to. If not passed, this party will not write the proof to a file.
    #[arg(long)]
    #[serde(skip_serializing_if = "::std::option::Option::is_none")]
    pub out: Option<PathBuf>,
    /// The output JSON file where the public inputs are written to. If not passed, this party will not write the public inputs to a file.
    #[arg(long)]
    #[serde(skip_serializing_if = "::std::option::Option::is_none")]
    pub public_input: Option<PathBuf>,
    /// The threshold of tolerated colluding parties
    #[arg(short, long, default_value_t = 1)]
    pub threshold: usize,
    /// Generate a recursive friendly proof
    #[arg(long)]
    pub recursive: bool,
}

/// Config for `build_and_generate_proof`
#[derive(Debug, Deserialize)]
pub struct BuildAndGenerateProofConfig {
    /// The path to the witness share file
    pub witness: PathBuf,
    /// The path to the circuit file, generated by Noir
    pub circuit: PathBuf,
    /// The path to the prover crs file
    pub crs: PathBuf,
    /// The MPC protocol to be used
    pub protocol: MPCProtocol,
    /// The transcript hasher to be used
    pub hasher: TranscriptHash,
    /// The output file where the final proof is written to. If not passed, this party will not write the proof to a file.
    pub out: Option<PathBuf>,
    /// The output JSON file where the public inputs are written to. If not passed, this party will not write the public inputs to a file.
    pub public_input: Option<PathBuf>,
    /// The threshold of tolerated colluding parties
    pub threshold: usize,
    /// Network config
    pub network: NetworkConfigFile,
    /// Whether to generate a recursive friendly proof
    pub recursive: bool,
}

/// Cli arguments for `creating_vk`
#[derive(Debug, Serialize, Args)]
pub struct CreateVKCli {
    /// The path to the config file
    #[arg(long)]
    #[serde(skip_serializing_if = "::std::option::Option::is_none")]
    pub config: Option<PathBuf>,
    /// The transcript hasher used for the proof
    #[arg(long, value_enum)]
    #[serde(skip_serializing_if = "::std::option::Option::is_none")]
    pub hasher: Option<TranscriptHash>,
    /// The path to the circuit file, generated by Noir
    #[arg(long)]
    #[serde(skip_serializing_if = "::std::option::Option::is_none")]
    pub circuit: Option<PathBuf>,
    /// The path to the prover crs file
    #[arg(long)]
    #[serde(skip_serializing_if = "::std::option::Option::is_none")]
    pub crs: Option<PathBuf>,
    /// The output path to the verification key file
    #[arg(long)]
    #[serde(skip_serializing_if = "::std::option::Option::is_none")]
    pub vk: Option<PathBuf>,
    /// Generate a recursive friendly vk
    #[arg(long)]
    pub recursive: bool,
}

/// Config for `creating_vk`
#[derive(Debug, Deserialize)]
pub struct CreateVKConfig {
    /// The transcript hasher used for the proof, bb uses a different vk for keccak, which misses some fields compared to the non-keccak vk
    pub hasher: TranscriptHash,
    /// The path to the circuit file, generated by Noir
    pub circuit: PathBuf,
    /// The path to the prover crs file
    pub crs: PathBuf,
    /// The path to the verification key file
    pub vk: PathBuf,
    /// Whether to generate a recursive friendly vk
    pub recursive: bool,
}

/// Cli arguments for `verify`
#[derive(Debug, Serialize, Args)]
pub struct VerifyCli {
    /// The transcript hasher used for the proof
    #[arg(long, value_enum)]
    #[serde(skip_serializing_if = "::std::option::Option::is_none")]
    pub hasher: Option<TranscriptHash>,
    /// The path to the config file
    #[arg(long)]
    #[serde(skip_serializing_if = "::std::option::Option::is_none")]
    pub config: Option<PathBuf>,
    /// The path to the proof file
    #[arg(long)]
    #[serde(skip_serializing_if = "::std::option::Option::is_none")]
    pub proof: Option<PathBuf>,
    /// The path to the verification key file
    #[arg(long)]
    #[serde(skip_serializing_if = "::std::option::Option::is_none")]
    pub vk: Option<PathBuf>,
    /// The path to the verifier crs file
    #[arg(long)]
    #[serde(skip_serializing_if = "::std::option::Option::is_none")]
    pub crs: Option<PathBuf>,
}

/// Config for `verify`
#[derive(Debug, Deserialize)]
pub struct VerifyConfig {
    /// The transcript hasher used for the proof
    pub hasher: TranscriptHash,
    /// The path to the proof file
    pub proof: PathBuf,
    /// The path to the verification key file
    pub vk: PathBuf,
    /// The path to the verifier crs file
    pub crs: PathBuf,
}

/// Cli arguments for `verify`
#[derive(Debug, Serialize, Args)]
pub struct DownloadCrsCLi {
    /// The path to the config file
    #[arg(long)]
    #[serde(skip_serializing_if = "::std::option::Option::is_none")]
    pub config: Option<PathBuf>,
    /// The path to the prover crs file
    #[arg(long)]
    #[serde(skip_serializing_if = "::std::option::Option::is_none")]
    pub crs: Option<PathBuf>,
    /// The number of points to download
    #[arg(short, long, default_value_t = 1)]
    pub num_points: usize,
}

/// Config for `verify`
#[derive(Debug, Deserialize)]
pub struct DownloadCrsConfig {
    /// The path to the prover crs file
    pub crs: PathBuf,
    /// The number of points to download
    pub num_points: usize,
}

/// Prefix for config env variables
pub const CONFIG_ENV_PREFIX: &str = "CONOIR_";

/// Error type for config parsing and merging
#[derive(thiserror::Error, Debug)]
#[error(transparent)]
pub struct ConfigError(#[from] figment::error::Error);

macro_rules! impl_config {
    ($cli: ty, $config: ty) => {
        impl $config {
            /// Parse config from file, env, cli
            pub fn parse(cli: $cli) -> Result<Self, ConfigError> {
                if let Some(path) = &cli.config {
                    Ok(Figment::new()
                        .merge(Toml::file(path))
                        .merge(Env::prefixed(CONFIG_ENV_PREFIX))
                        .merge(Serialized::defaults(cli))
                        .extract()?)
                } else {
                    Ok(Figment::new()
                        .merge(Env::prefixed(CONFIG_ENV_PREFIX))
                        .merge(Serialized::defaults(cli))
                        .extract()?)
                }
            }
        }
    };
}

impl_config!(SplitInputCli, SplitInputConfig);
impl_config!(SplitWitnessCli, SplitWitnessConfig);
impl_config!(SplitProvingKeyCli, SplitProvingKeyConfig);
impl_config!(MergeInputSharesCli, MergeInputSharesConfig);
impl_config!(GenerateWitnessCli, GenerateWitnessConfig);
impl_config!(TranslateWitnessCli, TranslateWitnessConfig);
impl_config!(TranslateProvingKeyCli, TranslateProvingKeyConfig);
impl_config!(BuildProvingKeyCLi, BuildProvingKeyConfig);
impl_config!(GenerateProofCli, GenerateProofConfig);
impl_config!(BuildAndGenerateProofCli, BuildAndGenerateProofConfig);
impl_config!(CreateVKCli, CreateVKConfig);
impl_config!(VerifyCli, VerifyConfig);
impl_config!(DownloadCrsCLi, DownloadCrsConfig);

pub fn share_rep3<F: PrimeField, R: Rng + CryptoRng>(
    witness: Vec<PubShared<F>>,
    rng: &mut R,
) -> [Vec<Rep3AcvmType<F>>; 3] {
    let mut res = array::from_fn(|_| Vec::with_capacity(witness.len()));

    for witness in witness {
        match witness {
            PubShared::Public(f) => {
                for r in res.iter_mut() {
                    r.push(Rep3AcvmType::from(f));
                }
            }
            PubShared::Shared(f) => {
                let shares = rep3::share_field_element(f, rng);
                for (r, share) in res.iter_mut().zip(shares) {
                    r.push(Rep3AcvmType::from(share));
                }
            }
        }
    }
    res
}

pub fn share_shamir<F: PrimeField, R: Rng + CryptoRng>(
    witness: Vec<PubShared<F>>,
    degree: usize,
    num_parties: usize,
    rng: &mut R,
) -> Vec<Vec<ShamirAcvmType<F>>> {
    let mut res = (0..num_parties)
        .map(|_| Vec::with_capacity(witness.len()))
        .collect::<Vec<_>>();

    for witness in witness {
        match witness {
            PubShared::Public(f) => {
                for r in res.iter_mut() {
                    r.push(ShamirAcvmType::from(f));
                }
            }
            PubShared::Shared(f) => {
                let shares = shamir::share_field_element(f, degree, num_parties, rng);
                for (r, share) in res.iter_mut().zip(shares) {
                    r.push(ShamirAcvmType::from(share));
                }
            }
        }
    }
    res
}

pub fn share_input_rep3<P: Pairing, N: Rep3Network, R: Rng + CryptoRng>(
    initial_witness: BTreeMap<String, PublicMarker<GenericFieldElement<P::ScalarField>>>,
    rng: &mut R,
) -> [BTreeMap<String, Rep3AcvmType<P::ScalarField>>; 3] {
    let mut witnesses = array::from_fn(|_| BTreeMap::default());
    for (witness, v) in initial_witness.into_iter() {
        match v {
            PublicMarker::Public(v) => {
                for w in witnesses.iter_mut() {
                    w.insert(witness.to_owned(), Rep3AcvmType::Public(v.into_repr()));
                }
            }
            PublicMarker::Private(v) => {
                let shares = rep3::share_field_element(v.into_repr(), rng);
                for (w, share) in witnesses.iter_mut().zip(shares) {
                    w.insert(witness.clone(), Rep3AcvmType::Shared(share));
                }
            }
        }
    }

    witnesses
}

pub fn translate_witness_share_rep3(
    witness: BTreeMap<String, Rep3AcvmType<ark_bn254::Fr>>,
    abi: &Abi,
) -> color_eyre::Result<WitnessMap<Rep3AcvmType<ark_bn254::Fr>>> {
    Rep3CoSolver::<ark_bn254::Fr, Rep3MpcNet>::witness_map_from_string_map(witness, abi)
}

pub fn convert_witness_to_vec_rep3<F: PrimeField>(
    mut witness_stack: WitnessStack<Rep3AcvmType<F>>,
) -> Vec<Rep3AcvmType<F>> {
    let witness_map = witness_stack
        .pop()
        .expect("Witness should be present")
        .witness;

    let mut wv = Vec::new();
    let mut index = 0;
    for (w, f) in witness_map.into_iter() {
        // ACIR uses a sparse format for WitnessMap where unused witness indices may be left unassigned.
        // To ensure that witnesses sit at the correct indices in the `WitnessVector`, we fill any indices
        // which do not exist within the `WitnessMap` with the dummy value of zero.
        while index < w.0 {
            wv.push(Rep3AcvmType::from(F::zero()));
            index += 1;
        }
        wv.push(f);
        index += 1;
    }
    wv
}

// This function is basically copied from Barretenberg
/// Downloads the CRS with num_points points to the crs_path.
pub fn download_g1_crs(num_points: usize, crs_path: &PathBuf) -> color_eyre::Result<()> {
    tracing::info!("Downloading CRS with {} points", num_points);
    let g1_end = num_points * 64 - 1;

    let url = "https://aztec-ignition.s3.amazonaws.com/MAIN%20IGNITION/flat/g1.dat";
    let command = format!("curl -s -H \"Range: bytes=0-{}\" '{}'", g1_end, url);
    let output = std::process::Command::new("sh")
        .arg("-c")
        .arg(&command)
        .output()
        .wrap_err("Failed to execute curl command")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(eyre!("Could not download CRS: {}", stderr));
    }

    let data = output.stdout;
    let mut file = File::create(crs_path).wrap_err("Failed to create CRS file")?;
    file.write_all(&data)
        .wrap_err("Failed to write data to CRS file")?;

    if data.len() < (g1_end + 1) {
        return Err(eyre!(
            "Downloaded CRS is incomplete: expected {} bytes, got {} bytes",
            g1_end + 1,
            data.len()
        ));
    }

    Ok(())
}
