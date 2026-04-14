import wandb
import subprocess
import sys

# Grid search configuration for PPO
grid_search_config = {
    'method': 'grid',
    'metric': {
        'name': 'episode_reward_mean',  # Common PPO metric
        'goal': 'maximize'
    },
    'parameters': {
        # PPO Core Hyperparameters
        'lr': {
            'values': [1e-4, 3e-4, 1e-3]
        },
        'gamma': {
            'values': [0.95, 0.99, 0.995]
        },
        'gae_lambda': {
            'values': [0.9, 0.95, 0.98]
        },
        'clip_coef': {
            'values': [0.1, 0.2, 0.3]
        },
        'ent_coef': {
            'values': [0.001, 0.01, 0.1]
        },
        'vf_coef': {
            'values': [0.25, 0.5, 1.0]
        },
        
        # Training Loop Parameters
        'num_steps': {
            'values': [64, 128, 256]
        },
        'ppo_epochs': {
            'values': [3, 4, 8]
        },
        'num_minis': {
            'values': [16, 32, 64]
        },
        'num_envs': {
            'values': [1, 4, 8]
        },
        
        # Model Architecture
        'hidden_size': {
            'values': [128, 256, 512]
        },
        'num_hidden': {
            'values': [3, 5, 7]
        },
        'gtrxl_layers': {
            'values': [2, 3, 4]
        },
        'num_heads': {
            'values': [4, 8, 16]
        },
        
        # Fixed parameters
        'anneal_lr': {'value': True},
        'norm_adv': {'value': True},
        'clip_vloss': {'value': True},
        'episodes': {'value': 750},
        'num_moves': {'value': 50},
        'warmup': {'value': 5},
        'num_prune': {'value': 25},
        'prune_frac': {'value': 0.5},
        'num_actions': {'value': 2},
        'd_model': {'value': 128},
        'd_head_inner': {'value': 64},
        'd_ff_inner': {'value': 128},
        'd_comp': {'value': 64},
        'gtrxl_mem_len': {'value': 800},
        'img_size': {'value': 250},
        'patch_size': {'value': 25},
        'img_chan': {'value': 3},
        'num_gauss': {'value': 8},
        'max_grad_norm': {'value': 0.5},
        'target_kl': {'value': 0.03},
        'early_stop': {'value': True},
        'stochastic': {'value': 'On'},
        'render': {'value': False},
        'dbg': {'value': False},
        'test': {'value': False},
        'save_dir': {'value': 'models'}
    }
}

# Bayesian search configuration for PPO
bayesian_search_config = {
    'method': 'bayes',
    'metric': {
        'name': 'episode_reward_mean',
        'goal': 'maximize'
    },
    'parameters': {
        # PPO Core Hyperparameters (continuous ranges)
        'lr': {
            'distribution': 'log_uniform_values',
            'min': 1e-5,
            'max': 1e-2
        },
        'gamma': {
            'distribution': 'uniform',
            'min': 0.9,
            'max': 0.999
        },
        'gae_lambda': {
            'distribution': 'uniform',
            'min': 0.8,
            'max': 0.99
        },
        'clip_coef': {
            'distribution': 'uniform',
            'min': 0.05,
            'max': 0.4
        },
        'ent_coef': {
            'distribution': 'log_uniform_values',
            'min': 1e-4,
            'max': 1e-1
        },
        'vf_coef': {
            'distribution': 'uniform',
            'min': 0.1,
            'max': 2.0
        },
        'max_grad_norm': {
            'distribution': 'uniform',
            'min': 0.1,
            'max': 2.0
        },
        'target_kl': {
            'distribution': 'uniform',
            'min': 0.01,
            'max': 0.1
        },
        
        # Training Loop Parameters
        'num_steps': {
            'distribution': 'int_uniform',
            'min': 32,
            'max': 512
        },
        'ppo_epochs': {
            'distribution': 'int_uniform',
            'min': 1,
            'max': 10
        },
        'num_minis': {
            'distribution': 'int_uniform',
            'min': 8,
            'max': 128
        },
        'num_envs': {
            'values': [1, 2, 4, 8, 16]  # Discrete values for env count
        },
        
        # Model Architecture
        'hidden_size': {
            'distribution': 'int_uniform',
            'min': 64,
            'max': 1024
        },
        'num_hidden': {
            'distribution': 'int_uniform',
            'min': 2,
            'max': 8
        },
        'd_model': {
            'distribution': 'int_uniform',
            'min': 64,
            'max': 512
        },
        'gtrxl_layers': {
            'distribution': 'int_uniform',
            'min': 1,
            'max': 6
        },
        'num_heads': {
            'values': [2, 4, 8, 16, 32]  # Powers of 2 for attention heads
        },
        'd_head_inner': {
            'distribution': 'int_uniform',
            'min': 32,
            'max': 128
        },
        'd_ff_inner': {
            'distribution': 'int_uniform',
            'min': 64,
            'max': 512
        },
        'd_comp': {
            'distribution': 'int_uniform',
            'min': 32,
            'max': 256
        },
        'gtrxl_mem_len': {
            'distribution': 'int_uniform',
            'min': 200,
            'max': 2000
        },
        
        # Fixed parameters
        'anneal_lr': {'value': True},
        'norm_adv': {'value': True},
        'clip_vloss': {'value': True},
        'episodes': {'value': 750},
        'num_moves': {'value': 50},
        'warmup': {'value': 5},
        'num_prune': {'value': 25},
        'prune_frac': {'value': 0.5},
        'num_actions': {'value': 2},
        'img_size': {'value': 250},
        'patch_size': {'value': 25},
        'img_chan': {'value': 3},
        'num_gauss': {'value': 8},
        'early_stop': {'value': True},
        'stochastic': {'value': 'On'},
        'render': {'value': False},
        'dbg': {'value': False},
        'test': {'value': False},
        'save_dir': {'value': 'models'}
    }
}

def train_model_cli():
    """Wrapper function that calls external PPO training program with hyperparameters"""
    wandb.init()
    config = wandb.config
    
    # Build CLI arguments from config
    cmd = ["./your_ppo_program"]  # Replace with your actual executable
    
    # Add all parameters as CLI arguments
    for param_name, param_value in config.items():
        if param_name == 'stochastic':
            # Handle the OnOff enum
            cmd.extend([f"--{param_name}", param_value.lower()])
        elif isinstance(param_value, bool):
            # Handle boolean flags
            if param_value:
                cmd.append(f"--{param_name}")
        elif isinstance(param_value, (int, float, str)):
            # Handle numeric and string parameters
            cmd.extend([f"--{param_name}", str(param_value)])
    
    # Add wandb run name for tracking
    cmd.extend(["--save_dir", f"models/{wandb.run.name}"])
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"PPO training completed: {result.stdout}")
        
        # Parse any metrics from stdout if your program outputs them
        # You might need to modify this based on your program's output format
        
    except subprocess.CalledProcessError as e:
        print(f"PPO training failed: {e.stderr}")
        wandb.log({"training_failed": True})
        sys.exit(1)
    
    wandb.finish()

def run_sweep(config, sweep_name, count=None):
    """Run a hyperparameter sweep"""
    sweep_id = wandb.sweep(config, project="hyperparameter-sweep", entity=None)
    print(f"Starting {sweep_name} with sweep ID: {sweep_id}")
    print(f"View sweep at: https://wandb.ai/[your-username]/hyperparameter-sweep/sweeps/{sweep_id}")
    
    if count:
        wandb.agent(sweep_id, train_model_cli, count=count)  # Changed this line
    else:
        wandb.agent(sweep_id, train_model_cli)  # Changed this line