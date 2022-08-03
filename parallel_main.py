from main import main
import os
import argparse
import tf_agents
import tensorflow as tf
import numpy as np
import random

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('game_name', help='Name of the Atari game')
    parser.add_argument('cfg_paths', help='List of YAML files containing arguments and hyperparameters', nargs='+', default=[])
    parser.add_argument('--seed', help='Random seed, for reproducibility', default=42)
    parser.add_argument('--debug', help='If set, run in debug mode (no logs)', action='store_true')
    parser.add_argument('--render', help='If set, render experiences every 100 collections', action='store_true')

    args = parser.parse_args()

    assert 'SLURM_PROCID' in os.environ
    rank = int(os.environ['SLURM_PROCID'])

    cfg_paths = args.cfg_paths
    assert len(cfg_paths) > rank

    args.cfg_path = cfg_paths[rank]

    if args.debug:
        os.environ["WANDB_MODE"] = "offline"

    random.seed(args.seed)
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)
    
    tf.keras.utils.set_random_seed(args.seed)
    tf.config.experimental.enable_op_determinism()
    tf_agents.system.multiprocessing.handle_main(main, [args])