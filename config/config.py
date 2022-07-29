from yacs.config import CfgNode as CN

cfg = CN()

cfg.env = CN()
cfg.env.max_time_steps = 10_000_000

cfg.ppo = CN()
cfg.ppo.lr = 2.5e-4
cfg.ppo.horizon = 128
cfg.ppo.num_epochs = 3
cfg.ppo.discount = 0.99
cfg.ppo.gae_parameter = 0.95
cfg.ppo.num_actors = 8
cfg.ppo.clipping_parameter = 0.1
cfg.ppo.vf_coeff = 1.0
cfg.ppo.entropy_coeff = 0.01
cfg.ppo.num_minibatch = 4
cfg.ppo.annealing = True

cfg.model = CN()
cfg.model.share_params = False
cfg.model.use_rnn = False
cfg.model.rnn_units = 64
cfg.model.memory_size = 16

def get_cfg_defaults():
    """
    Returns a clone of the yacs CfgNode object with default values for the project, so that the original defaults 
    will not be altered.
    """
    return cfg.clone()

def save_cfg_default():
    """Save in a YAML file the default version of the configuration file, in order to provide a template to be modified."""
    with open('config/default.yaml', 'w') as f:
        f.write(cfg.dump())
        f.flush()
        f.close()

if __name__ == '__main__':
    save_cfg_default()