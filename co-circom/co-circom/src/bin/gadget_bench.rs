use ark_ff::{PrimeField, UniformRand};
use clap::Parser;
use co_circom::{ConfigError, CONFIG_ENV_PREFIX};
use color_eyre::eyre::Context;
use figment::{
    providers::{Env, Format, Serialized, Toml},
    Figment,
};
use mpc_core::{
    gadgets::poseidon2::{bn254_t2::POSEIDON2_BN254_T2_PARAMS, Poseidon2T2D5},
    protocols::{
        rep3::{
            self,
            network::{Rep3MpcNet, Rep3Network},
            Rep3PrimeFieldShare, Rep3Protocol,
        },
        shamir::{
            self,
            fieldshare::ShamirPrimeFieldShare,
            network::{ShamirMpcNet, ShamirNetwork},
            ShamirProtocol,
        },
    },
};
use mpc_net::config::NetworkConfig;
use rand::{CryptoRng, Rng};
use serde::{Deserialize, Serialize};
use std::{
    path::PathBuf,
    process::ExitCode,
    thread::sleep,
    time::{Duration, Instant},
};

fn install_tracing() {
    use tracing_subscriber::prelude::*;
    use tracing_subscriber::{fmt, EnvFilter};

    let fmt_layer = fmt::layer().with_target(true).with_line_number(true);
    let filter_layer = EnvFilter::try_from_default_env()
        .or_else(|_| EnvFilter::try_new("info"))
        .unwrap();

    tracing_subscriber::registry()
        .with(filter_layer)
        .with(fmt_layer)
        .init();
}

/// Cli arguments
#[derive(Debug, Serialize, Parser)]
pub struct Cli {
    /// The path to the config file
    #[arg(long)]
    #[serde(skip_serializing_if = "::std::option::Option::is_none")]
    pub config: Option<PathBuf>,

    /// The number of testruns
    #[arg(short, long, default_value_t = 10)]
    pub runs: usize,

    /// The threshold of tolerated colluding parties
    #[arg(short, long, default_value_t = 1)]
    pub threshold: usize,
}

/// Config
#[derive(Debug, Deserialize)]
pub struct Config {
    /// The number of testruns
    pub runs: usize,
    /// The threshold of tolerated colluding parties
    pub threshold: usize,
    /// Network config
    pub network: NetworkConfig,
}

impl Config {
    /// Parse config from file, env, cli
    pub fn parse(cli: Cli) -> Result<Self, ConfigError> {
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

fn main() -> color_eyre::Result<ExitCode> {
    install_tracing();
    let cli = Cli::parse();
    let config = Config::parse(cli).context("while parsing config")?;

    poseidon2_plain(&config)?;

    poseidon2_rep3(&config)?;
    poseidon2_rep3_with_precomp(&config)?;

    poseidon2_shamir(&config)?;
    poseidon2_shamir_with_precomp(&config)?;

    Ok(ExitCode::SUCCESS)
}

fn print_runtimes(times: Vec<f64>, id: usize, s: &str) {
    let mut min = f64::INFINITY;
    let mut max = 0f64;
    let mut avg = 0f64;

    let len = times.len();
    for runtime in times {
        avg += runtime;
        min = min.min(runtime);
        max = max.max(runtime);
    }
    avg /= len as f64;

    tracing::info!("{}: Party {}, {} runs", s, id, len);
    tracing::info!("\tavg: {:.2}µs", avg);
    tracing::info!("\tmin: {:.2}µs", min);
    tracing::info!("\tmax: {:.2}µs", max);
}

fn poseidon2_plain(config: &Config) -> color_eyre::Result<ExitCode> {
    let mut rng = rand::thread_rng();

    let mut times = Vec::with_capacity(config.runs);

    for _ in 0..config.runs {
        let mut input: Vec<ark_bn254::Fr> = (0..2).map(|_| ark_bn254::Fr::rand(&mut rng)).collect();

        let poseidon2 = Poseidon2T2D5::new(&POSEIDON2_BN254_T2_PARAMS);

        let start = Instant::now();
        poseidon2.plain_permutation_in_place(input.as_mut_slice().try_into().unwrap());
        let duration = start.elapsed().as_micros() as f64;
        times.push(duration);
    }

    print_runtimes(times, 0, "Poseidon2 plain");

    Ok(ExitCode::SUCCESS)
}

fn share_random_input_rep3<F: PrimeField, R: Rng + CryptoRng>(
    net: &mut Rep3MpcNet,
    rng: &mut R,
) -> color_eyre::Result<Vec<Rep3PrimeFieldShare<F>>> {
    let share = match net.get_id() {
        rep3::id::PartyID::ID0 => {
            let input: Vec<F> = (0..2).map(|_| F::rand(rng)).collect();
            let shares = rep3::utils::share_field_elements(&input, rng);
            let shares = shares
                .into_iter()
                .map(|s| s.into_iter().collect::<Vec<_>>())
                .collect::<Vec<_>>();

            net.send_next_many(&shares[1])?;
            net.send_many(net.get_id().prev_id(), &shares[2])?;
            shares[0].clone()
        }
        rep3::id::PartyID::ID1 => net.recv_prev_many()?,
        rep3::id::PartyID::ID2 => net.recv_many(net.get_id().next_id())?,
    };

    Ok(share)
}

fn poseidon2_rep3(config: &Config) -> color_eyre::Result<ExitCode> {
    if config.threshold != 1 {
        return Err(color_eyre::Report::msg("Threshold must be 1 for rep3"));
    }

    let mut rng = rand::thread_rng();

    let mut times = Vec::with_capacity(config.runs);
    let mut id = 0;

    for _ in 0..config.runs {
        // connect to network
        let mut net = Rep3MpcNet::new(config.network.to_owned())?;
        id = usize::from(net.get_id());

        let mut share = share_random_input_rep3::<ark_bn254::Fr, _>(&mut net, &mut rng)?;

        // init MPC protocol
        let mut protocol = Rep3Protocol::new(net)?;

        let poseidon2 = Poseidon2T2D5::new(&POSEIDON2_BN254_T2_PARAMS);

        let start = Instant::now();
        poseidon2
            .rep3_permutation_in_place(share.as_mut_slice().try_into().unwrap(), &mut protocol)?;
        let duration = start.elapsed().as_micros() as f64;
        times.push(duration);

        sleep(Duration::from_millis(100));
    }

    print_runtimes(times, id, "Poseidon2 rep3");

    Ok(ExitCode::SUCCESS)
}

fn poseidon2_rep3_with_precomp(config: &Config) -> color_eyre::Result<ExitCode> {
    if config.threshold != 1 {
        return Err(color_eyre::Report::msg("Threshold must be 1 for rep3"));
    }

    let mut rng = rand::thread_rng();

    let mut times = Vec::with_capacity(config.runs);
    let mut id = 0;

    for _ in 0..config.runs {
        // connect to network
        let mut net = Rep3MpcNet::new(config.network.to_owned())?;
        id = usize::from(net.get_id());

        let mut share = share_random_input_rep3::<ark_bn254::Fr, _>(&mut net, &mut rng)?;

        // init MPC protocol
        let mut protocol = Rep3Protocol::new(net)?;

        let poseidon2 = Poseidon2T2D5::new(&POSEIDON2_BN254_T2_PARAMS);

        let start = Instant::now();
        poseidon2.rep3_permutation_in_place_with_precomputation(
            share.as_mut_slice().try_into().unwrap(),
            &mut protocol,
        )?;
        let duration = start.elapsed().as_micros() as f64;
        times.push(duration);

        sleep(Duration::from_millis(100));
    }

    print_runtimes(times, id, "Poseidon2 rep3 with precomp");

    Ok(ExitCode::SUCCESS)
}

fn share_random_input_shamir<F: PrimeField, R: Rng + CryptoRng>(
    net: &mut ShamirMpcNet,
    threshold: usize,
    rng: &mut R,
) -> color_eyre::Result<Vec<ShamirPrimeFieldShare<F>>> {
    let share = if net.get_id() == 0 {
        let input: Vec<F> = (0..2).map(|_| F::rand(rng)).collect();
        let shares =
            shamir::utils::share_field_elements(&input, threshold, net.get_num_parties(), rng);
        let myshare = shares[0].clone();
        for (i, val) in shares.into_iter().enumerate().skip(1) {
            net.send_many(i, &val.get_inner())?;
        }
        myshare.get_inner()
    } else {
        net.recv_many(0)?
    };

    let share = ShamirPrimeFieldShare::convert_vec_rev(share);

    Ok(share)
}

fn poseidon2_shamir(config: &Config) -> color_eyre::Result<ExitCode> {
    let mut rng = rand::thread_rng();

    let mut times = Vec::with_capacity(config.runs);
    let mut id = 0;

    for _ in 0..config.runs {
        // connect to network
        let mut net = ShamirMpcNet::new(config.network.to_owned())?;
        id = net.get_id();

        let mut share =
            share_random_input_shamir::<ark_bn254::Fr, _>(&mut net, config.threshold, &mut rng)?;

        // init MPC protocol
        let mut protocol = ShamirProtocol::new(config.threshold, net)?;

        let poseidon2 = Poseidon2T2D5::new(&POSEIDON2_BN254_T2_PARAMS);

        let start = Instant::now();
        poseidon2
            .shamir_permutation_in_place(share.as_mut_slice().try_into().unwrap(), &mut protocol)?;
        let duration = start.elapsed().as_micros() as f64;
        times.push(duration);

        sleep(Duration::from_millis(100));
    }

    print_runtimes(times, id, "Poseidon2 shamir");

    Ok(ExitCode::SUCCESS)
}

fn poseidon2_shamir_with_precomp(config: &Config) -> color_eyre::Result<ExitCode> {
    let mut rng = rand::thread_rng();

    let mut times = Vec::with_capacity(config.runs);
    let mut id = 0;

    for _ in 0..config.runs {
        // connect to network
        let mut net = ShamirMpcNet::new(config.network.to_owned())?;
        id = net.get_id();

        let mut share =
            share_random_input_shamir::<ark_bn254::Fr, _>(&mut net, config.threshold, &mut rng)?;

        // init MPC protocol
        let mut protocol = ShamirProtocol::new(config.threshold, net)?;

        let poseidon2 = Poseidon2T2D5::new(&POSEIDON2_BN254_T2_PARAMS);

        let start = Instant::now();
        poseidon2.shamir_permutation_in_place_with_precomputation(
            share.as_mut_slice().try_into().unwrap(),
            &mut protocol,
        )?;
        let duration = start.elapsed().as_micros() as f64;
        times.push(duration);

        sleep(Duration::from_millis(100));
    }

    print_runtimes(times, id, "Poseidon2 shamir with precomp");

    Ok(ExitCode::SUCCESS)
}
