use burn::{tensor::{backend::Backend, Tensor}, module::Module, nn::{Linear, GroupNorm, LinearConfig, GroupNormConfig}, config::Config};

use crate::Model;

use super::{ThreeTuple, Activation};

#[derive(Module, Debug)]
pub struct LinearNetwork<B: Backend> {
    linears: Vec<ThreeTuple<Linear<B>, Option<GroupNorm<B>>, Activation>>,
}

#[derive(Config)]
pub struct LinearNetworkConfig {
    pub linears: Vec<(LinearConfig, Option<GroupNormConfig>, Activation)>,
}


impl<B: Backend> Model<B> for LinearNetwork<B> {
    type Input = Tensor<B, 2>;
    type Output = Tensor<B, 2>;
    type Config = LinearNetworkConfig;

    fn from_config(config: Self::Config) -> Self {
        Self {
            linears: config.linears.into_iter().map(|(linear_config, norm, activation)| ThreeTuple(linear_config.init(), norm.map(|x| x.init()), activation)).collect()
        }
    }

    fn forward(&self, mut x: Self::Input) -> Self::Output {
        for layer in &self.linears {
            x = layer.0.forward(x);
            if let Some(norm) = &layer.1 {
                x = norm.forward(x);
            }
            x = layer.2.forward(x);
        }
        x
    }
}