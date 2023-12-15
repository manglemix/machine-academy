use burn::{
    config::Config,
    module::Module,
    nn::{GroupNormConfig, LinearConfig},
    tensor::{backend::Backend, Tensor},
};

use crate::Model;

use super::{
    find_nth_factor,
    linear::{LinearNetwork, LinearNetworkConfig},
    Activation, Normalize,
};

#[derive(Module, Debug)]
pub struct LinearAutoEncoder<B: Backend> {
    encoder: LinearNetwork<B>,
    decoder: LinearNetwork<B>,
}

#[derive(Config)]
pub struct LinearAutoEncoderConfig {
    pub latent_dim: usize,
    pub input_size: usize,
    pub half_depth: usize,
    pub activation: Activation,
    #[config(default = true)]
    pub normalize: bool,
    #[config(default = 1)]
    pub normalize_group_idx: usize,
    #[config(default = 1e-5)]
    pub normalize_eps: f64,
}

impl<B: Backend> Model<B> for LinearAutoEncoder<B> {
    type Input = Tensor<B, 2>;
    type Output = Tensor<B, 2>;
    type Config = LinearAutoEncoderConfig;

    fn from_config(config: Self::Config) -> Self {
        let mut linears = Vec::with_capacity(config.half_depth);
        let step = (config.input_size - config.latent_dim) as f64 / config.half_depth as f64;

        for i in 0..config.half_depth {
            let layer_input_size = (config.input_size as f64 - step * i as f64).round() as usize;
            let layer_output_size =
                (config.input_size as f64 - step * (i + 1) as f64).round() as usize;

            let norm = config.normalize.then(|| {
                GroupNormConfig::new(
                    find_nth_factor(layer_output_size, config.normalize_group_idx)
                        .unwrap_or(layer_output_size),
                    layer_output_size,
                )
                .with_epsilon(config.normalize_eps)
            });
            linears.push((
                LinearConfig::new(layer_input_size, layer_output_size),
                norm,
                config.activation,
            ));
        }
        let encoder = Model::from_config(LinearNetworkConfig { linears });
        linears = Vec::with_capacity(config.half_depth);

        for i in 0..config.half_depth {
            let layer_input_size = (config.latent_dim as f64 + step * i as f64).round() as usize;
            let layer_output_size =
                (config.latent_dim as f64 + step * (i + 1) as f64).round() as usize;

            let norm = config.normalize.then(|| {
                GroupNormConfig::new(
                    find_nth_factor(layer_output_size, config.normalize_group_idx)
                        .unwrap_or(layer_output_size),
                    layer_output_size,
                )
                .with_epsilon(config.normalize_eps)
            });
            linears.push((
                LinearConfig::new(layer_input_size, layer_output_size),
                norm,
                config.activation,
            ));
        }
        let decoder = Model::from_config(LinearNetworkConfig { linears });
        Self { encoder, decoder }
    }

    fn forward(&self, input: Self::Input) -> Self::Output {
        self.decoder.forward(self.encoder.forward(input))
    }
}

#[derive(Config)]
pub struct LinearAutoEncoderSuperConfig {
    #[config(default = 1)]
    pub min_latent_dim: usize,
    pub max_latent_dim: usize,
    pub input_size: usize,
    #[config(default = 1)]
    pub min_half_depth: usize,
    pub max_half_depth: usize,
    pub activations: Vec<Activation>,
    pub normalize: Normalize,
    #[config(default = 0)]
    pub min_normalize_group_idx: usize,
    pub max_normalize_group_idx: usize,
    #[config(default = -5)]
    pub min_norm_eps_pow: isize,
    #[config(default = -1)]
    pub max_norm_eps_pow: isize,
}
