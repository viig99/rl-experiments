import gym
from ditk import logging
from ding.model import VAC
from ding.policy import PPOPolicy
from ding.envs import DingEnvWrapper, BaseEnvManagerV2
from ding.config import compile_config
from ding.framework import task, ding_init
from ding.framework.context import OnlineRLContext
from ding.framework.middleware import (
    multistep_trainer,
    StepCollector,
    interaction_evaluator,
    CkptSaver,
    gae_estimator,
    online_logger,
    termination_checker,
)
from ding.utils import set_pkg_seed
from dizoo.box2d.lunarlander.config.lunarlander_ppo_config import (
    main_config,
    create_config,
)


def register_custom_env():
    from gym.envs.registration import register

    register(
        id="LunarLander-v3",
        entry_point="lunar_lander_custom:LunarLander",
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


def main():
    logging.getLogger().setLevel(logging.INFO)
    cfg = compile_config(main_config, create_cfg=create_config, auto=True)
    ding_init(cfg)
    with task.start(async_mode=False, ctx=OnlineRLContext()):
        collector_env = BaseEnvManagerV2(
            env_fn=[
                lambda: DingEnvWrapper(windy_lunarlander_env())
                for _ in range(cfg.env.collector_env_num)
            ],
            cfg=cfg.env.manager,
        )
        evaluator_env = BaseEnvManagerV2(
            env_fn=[
                lambda: DingEnvWrapper(windy_lunarlander_env())
                for _ in range(cfg.env.evaluator_env_num)
            ],
            cfg=cfg.env.manager,
        )

        set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)

        model = VAC(**cfg.policy.model)
        policy = PPOPolicy(cfg.policy, model=model)

        task.use(interaction_evaluator(cfg, policy.eval_mode, evaluator_env))
        task.use(StepCollector(cfg, policy.collect_mode, collector_env))
        task.use(gae_estimator(cfg, policy.collect_mode))
        task.use(multistep_trainer(policy.learn_mode, log_freq=50))
        task.use(CkptSaver(policy, cfg.exp_name, train_freq=100))
        task.use(online_logger(train_show_freq=3))
        task.use(termination_checker(1e6))
        task.run()


if __name__ == "__main__":
    register_custom_env()
    main()
