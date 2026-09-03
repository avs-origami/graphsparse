//! This module contains all model options and hyperparameters in once place.

use clap::ValueEnum;
use lazy_static::lazy_static;
use serde::Serialize;

lazy_static! {
    static ref DEF: GlobalOpts = GlobalOpts::default();
}

pub use clap::Parser;

use sim::Clutter;

#[derive(Clone, Copy, Debug, ValueEnum, Serialize)]
pub enum OnOff {
    On,
    Off
}

#[derive(Clone, Copy, Debug, ValueEnum, Serialize)]
pub enum Activation {
    #[value(name = "tanh")]
    Tanh,
    #[value(name = "relu")]
    ReLU
}


/// This struct contains all options and hyperparameters for model training and
/// inference.
/// 
/// There are two ways to construct this. By parsing command line arguments:
/// 
/// ```rust
/// let opts = GlobalOpts::parse();
/// ```
/// 
/// Or using default:
/// 
/// ```rust
/// let opts = GlobalOpts {
///     episodes: 1000,
///     ..Default::default()
/// };
/// ```
#[derive(Clone, Debug, Parser, Serialize)]
#[command(about = "Program used to train the RL model", long_about = None)]
pub struct GlobalOpts {
    // =========================================
    // =======   Training Loop Options   =======
    // =========================================
    
    /// Number of training episodes.
    #[arg(long, default_value_t = DEF.episodes)]
    pub episodes: usize,
    /// Number of timesteps in each episode.
    #[arg(long, default_value_t = DEF.num_steps)]
    pub num_steps: usize,
    /// Number of robot moves before resetting environment.
    #[arg(long, default_value_t = DEF.num_moves)]
    pub num_moves: usize,
    /// Number of parallel environments.
    #[arg(long, default_value_t = DEF.num_envs)]
    pub num_envs: usize,
    /// Number of epochs to train on each batch of data.
    #[arg(long, default_value_t = DEF.ppo_epochs)]
    pub ppo_epochs: usize,
    /// Number of minibathces to train on.
    #[arg(long, default_value_t = DEF.num_minis)]
    pub num_minis: usize,
    /// How many episodes to complete before training the actor.
    #[arg(long, default_value_t = DEF.warmup)]
    pub warmup: usize,
    /// Number of nodes to prune per timestep.
    #[arg(long, default_value_t = DEF.num_prune)]
    pub num_prune: usize,
    /// Total fraction of nodes to prune between robot moves.
    #[arg(long, default_value_t = DEF.prune_frac)]
    pub prune_frac: f32,
    /// Number of evals per episode.
    #[arg(long, default_value_t = DEF.num_eval)]
    pub num_eval: usize,

    // =========================================
    // =======   Model Hyperparameters   =======
    // =========================================
    
    /// Number of neurons in the hidden layers of the linear heads.
    #[arg(long, default_value_t = DEF.hidden_size)]
    pub hidden_size: usize,
    /// Number of hidden layers in the linear heads.
    #[arg(long, default_value_t = DEF.num_hidden)]
    pub num_hidden: usize,
    /// Number of actions.
    #[arg(long, default_value_t = DEF.num_actions)]
    pub num_actions: usize,

    /// Size of GTrXL input / output.
    #[arg(long, default_value_t = DEF.d_model)]
    pub d_model: usize,
    /// Number of encoder layers for the GTrXL.
    #[arg(long, default_value_t = DEF.gtrxl_layers)]
    pub gtrxl_layers: usize,
    /// Number of attention heads for the GTrXL.
    #[arg(long, default_value_t = DEF.num_heads)]
    pub num_heads: usize,
    /// Size of attention heads for GTrXL.
    #[arg(long, default_value_t = DEF.d_head_inner)]
    pub d_head_inner: usize,
    /// Size of FF layers for GTrXL.
    #[arg(long, default_value_t = DEF.d_ff_inner)]
    pub d_ff_inner: usize,
    /// Compressed size of each token.
    #[arg(long, default_value_t = DEF.d_comp)]
    pub d_comp: usize,
    /// Number of hidden states to store in GTrXL memory.
    #[arg(long, default_value_t = DEF.gtrxl_mem_len)]
    pub gtrxl_mem_len: usize,
    /// Image dimensions (assumes square images).
    #[arg(long, default_value_t = DEF.img_size)]
    pub img_size: usize,
    /// Patch size for GTrXL input.
    #[arg(long, default_value_t = DEF.patch_size)]
    pub patch_size: usize,
    /// Image channels. Options are 3 or 1.
    #[arg(long, default_value_t = DEF.img_chan)]
    pub img_chan: usize,
    /// Number of gaussian components to predict.
    #[arg(long, default_value_t = DEF.num_gauss)]
    pub num_gauss: usize,
    /// Activation function for linear heads.
    #[arg(value_enum, long, default_value_t = DEF.activation)]
    pub activation: Activation,
    
    // =========================================
    // ========   PPO Hyperparameters   ========
    // =========================================

    /// Whether to anneal the learning rate.
    #[arg(long, default_value_t = DEF.anneal_lr)]
    pub anneal_lr: bool,
    /// Wehther to normalize advantages.
    #[arg(long, default_value_t = DEF.norm_adv)]
    pub norm_adv: bool,
    /// Whether to clip the value loss.
    #[arg(long, default_value_t = DEF.clip_vloss)]
    pub clip_vloss: bool,
    /// Whether to check for KL divergence.
    #[arg(long, default_value_t = DEF.early_stop)]
    pub early_stop: bool,
    /// Whether to sample GMM parameters stochastically
    /// (train time) or not (eval time).
    #[arg(value_enum, long, default_value_t = DEF.stochastic)]
    pub stochastic: OnOff,

    /// Learning rate.
    #[arg(long, default_value_t = DEF.lr)]
    pub lr: f32,
    /// Discount factor.
    #[arg(long, default_value_t = DEF.gamma)]
    pub gamma: f32,
    /// Lambda for GAE estimation.
    #[arg(long, default_value_t = DEF.gae_lambda)]
    pub gae_lambda: f32,
    /// PPO clip parameter.
    #[arg(long, default_value_t = DEF.clip_coef)]
    pub clip_coef: f32,
    /// Value function coefficient in the loss.
    #[arg(long, default_value_t = DEF.vf_coef)]
    pub vf_coef: f32,
    /// Entropy coefficient in the loss.
    #[arg(long, default_value_t = DEF.ent_coef)]
    pub ent_coef: f32,
    /// Maximum gradient norm for clipping.
    #[arg(long, default_value_t = DEF.max_grad_norm)]
    pub max_grad_norm: f32,
    /// Target KL divergence threshold.
    #[arg(long, default_value_t = DEF.target_kl)]
    pub target_kl: f32,

    // =========================================
    // =======   Miscellaneous Options   =======
    // =========================================

    /// Clutter level for the environment.
    #[serde(skip)]
    #[arg(value_enum, long, default_value_t = DEF.clutter)]
    pub clutter: Clutter,
    /// Whether to render the simulation.
    #[arg(long, default_value_t = DEF.render)]
    pub render: bool,
    /// Whether to hide the tree. Only takes effect with --render.
    #[arg(long, default_value_t = DEF.htree)]
    pub htree: bool,
    /// Whether to show debugging info.
    #[arg(long, default_value_t = DEF.dbg)]
    pub dbg: bool,
    /// Don't train the model, just test.
    #[arg(long, default_value_t = DEF.test)]
    pub test: bool,
    /// Disable pruning. Only takes effect when used with --test.
    #[arg(long, default_value_t = DEF.no_prune)]
    pub no_prune: bool,
    /// Random pruning.
    #[arg(long, default_value_t = DEF.random)]
    pub random: bool,
    /// Seed for the RNG.
    #[arg(long)]
    pub seed: Option<u64>,
    /// Load model files. Format: "run_id, checkpoint". If checkpoint specified,
    /// ppo[checkpoint].pt and [checkpoint].pt are tried; otherwise,
    /// use the highest-numbered ppoXX.pt checkpoint.
    #[arg(long, value_delimiter = ',')]
    pub load: Option<Vec<String>>,
    /// Directory to save model files.
    #[arg(long, default_value_t = DEF.save_dir.clone())]
    pub save_dir: String,
    /// Whether to show even more debugging info than --dbg.
    #[arg(long, default_value_t = DEF.verbose)]
    pub verbose: bool,
}

impl Default for GlobalOpts {
    fn default() -> Self {
        GlobalOpts {
            episodes: 500,
            num_steps: 2048,
            num_moves: 50,
            num_envs: 1,
            ppo_epochs: 4,
            num_minis: 16,
            warmup: 15,
            num_prune: 25,
            prune_frac: 0.96,
            num_eval: 3,
            hidden_size: 512,
            num_hidden: 3,
            num_actions: 2,
            d_model: 1024,
            gtrxl_layers: 3,
            num_heads: 8,
            d_head_inner: 512,
            d_ff_inner: 512,
            d_comp: 64,
            gtrxl_mem_len: 100,
            img_size: 250,
            patch_size: 25,
            img_chan: 1,
            num_gauss: 32,
            activation: Activation::Tanh,
            anneal_lr: true,
            norm_adv: true,
            clip_vloss: true,
            early_stop: true,
            stochastic: OnOff::On,
            lr: 1e-4,
            gamma: 0.99,
            gae_lambda: 0.95,
            clip_coef: 0.1,
            ent_coef: 5e-4,
            vf_coef: 0.5,
            max_grad_norm: 0.5,
            target_kl: 0.08,
            clutter: Clutter::Spicy,
            render: false,
            htree: false,
            dbg: false,
            test: false,
            no_prune: false,
            random: false,
            seed: None,
            load: None,
            save_dir: "models".into(),
            verbose: false,
        }
    }
}