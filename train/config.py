from dataclasses import dataclass
from typing import Optional

from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from train import common, env_spec, learner, pantheon_env


@dataclass
class ConfigTrain:
    # Base directory for logging (will be set at runtime).
    base_logdir: Optional[str] = None
    # Whether to also test the model after training it.
    test_after_train: bool = True


@dataclass
class MergedConfig(
    ConfigTrain,
    common.ConfigCommon,
    learner.ConfigLearner,
    pantheon_env.ConfigEnv,
):
    """
    This is the class that will hold the final config.

    It merges fields from all sub-configs.
    """


def init_config():
    # Register the config.
    cs = ConfigStore.instance()

    # Each sub-config is converted to a dictionary so that it can be merged.
    # See https://github.com/omry/omegaconf/issues/721#issuecomment-846794662
    configs_as_dicts = [
        OmegaConf.to_container(OmegaConf.structured(c))
        for c in [
            ConfigTrain,
            common.ConfigCommon,
            learner.ConfigLearner,
            pantheon_env.ConfigEnv,
        ]
    ]

    cs.store(group="jobs", name="base_jobs", node=env_spec.ConfigJobs)

    cs.store(
        name="base_config",
        # We merge all configs from multiple sources.
        node=OmegaConf.merge(MergedConfig, *configs_as_dicts),
    )
