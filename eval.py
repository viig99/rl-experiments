import random
import torch
import gym
from easydict import EasyDict

from ding.config import compile_config
from ding.worker import (
    BaseLearner,
    SampleSerialCollector,
    InteractionSerialEvaluator,
    AdvancedReplayBuffer,
)
from ding.envs import BaseEnvManager, DingEnvWrapper
from ding.policy import PPOPolicy
from ding.model import VAC
from ding.utils import set_pkg_seed
from dizoo.box2d.lunarlander.config.lunarlander_ppo_config import (
    main_config,
    create_config,
)


def register_custom_env():
    from gym.envs.registration import register

    register(
        id="LunarLander-v3",
        entry_point="lunar_lander_other:LunarLander",
        max_episode_steps=1000,
        reward_threshold=200,
    )


def windy_lunarlander_env():
    env = gym.make(
        "LunarLander-v3",
        continuous=False,
        gravity=-10.0,
        enable_wind=True,
        wind_power=15.0,
        turbulence_power=1.5,
    )
    return env


def wrapped_lunarlander_env():
    return DingEnvWrapper(
        windy_lunarlander_env(),
        EasyDict(env_wrapper="default"),
    )


def main(main_cfg, create_cfg):
    seed = random.randint(0, 2**16 - 1)
    main_cfg["exp_name"] = "lunarlander_ppo_eval"
    main_config["env"]["evaluator_env_num"] = 3
    main_config["env"]["n_evaluator_episode"] = 3
    cfg = compile_config(
        main_cfg,
        BaseEnvManager,
        PPOPolicy,
        BaseLearner,
        SampleSerialCollector,
        InteractionSerialEvaluator,
        AdvancedReplayBuffer,
        create_cfg=create_cfg,
        save_cfg=False,
    )

    num_evs = 1

    # Create main components: env, policy
    evaluator_env = BaseEnvManager(
        env_fn=[wrapped_lunarlander_env for _ in range(num_evs)],
        cfg=cfg.env.manager,
    )

    evaluator_env.enable_save_replay("video")  # switch save replay interface

    # Set random seed for all package and instance
    evaluator_env.seed(seed, dynamic_seed=True)
    set_pkg_seed(seed, use_cuda=cfg.policy.cuda)

    # Set up RL Policy
    model = VAC(**cfg.policy.model)
    policy = PPOPolicy(cfg.policy, model=model)
    policy.eval_mode.load_state_dict(
        torch.load("lunarlander_ppo_seed0/ckpt/final.pth.tar", map_location="cpu")
    )

    evaluator = InteractionSerialEvaluator(
        cfg.policy.eval.evaluator,
        evaluator_env,
        policy.eval_mode,
        exp_name=cfg.exp_name,
    )
    evaluator.eval()


if __name__ == "__main__":
    register_custom_env()
    main(main_config, create_config)
