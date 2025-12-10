import numpy as np
import argparse
import torch
import habitat
import habitat.config
import habitat.config.default
from habitat.datasets import make_dataset
from habitat import Env

from ovon.dataset import OVONDatasetV1
# from ovon.task.simulator import OVONHabitatConfig
from omegaconf import DictConfig, OmegaConf
from ovon.config import OVONDistanceToGoalConfig
from habitat import Env


from agent.citywalker_agent import evaluate_agent_ovon





def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--exp-config",
        type=str,
        required=True,
        help="path to config yaml containing info about experiment",
    )
    parser.add_argument(
        "--split-num",
        type=int,
        required=True,
        help="chunks of evluation"
    )
    
    parser.add_argument(
        "--split-id",
        type=int,
        required=True,
        help="chunks ID of evluation"

    )

    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="location of model weights"

    )

    parser.add_argument(
        "--result-path",
        type=str,
        required=True,
        help="location to save results"

    )

    parser.add_argument(
        "--use-unified-prompt",
        type=int,
        required=True,
        help="location to save results"
    )

    args = parser.parse_args()
    run_exp(**vars(args))


def run_exp(exp_config: str, split_num: str, split_id: str, model_path: str, result_path: str, opts=None, use_unified_prompt:int = 0) -> None:
    """Runs experiment given mode and config

    Args:
        exp_config: path to config file.
        run_type: "train" or "eval.
        opts: list of strings of additional config options.
    """

    print("========================================")
    print("model_path:\t", model_path)
    print("result_path:\t", result_path)
    print("========================================")
    
    config_all = habitat.get_config(exp_config)
    config = config_all.habitat
    # OmegaConf.set_readonly(config_all, False)
    dataset = habitat.make_dataset(config.dataset.type, config=config.dataset)
    # env = Env(config, dataset)
    # obs = env.reset()

    dataset.episodes.sort(key=lambda ep: ep.episode_id)

    OmegaConf.set_readonly(config, False)
    config.task.measurements.distance_to_goal = OVONDistanceToGoalConfig()
    # debug select a episode
    # dataset.episodes = [item for item in dataset.episodes if item.episode_id == '381' ]


    print("filter_existing_episodes.......")
    print("before filtering: len(episodes):", len(dataset.episodes))
    # import pdb;pdb.set_trace()
    try:
        dataset.filter_existing_episodes(result_path = result_path)
    except Exception as e:
        print(e)
        print('filtering failed......')
    print("after filtering: len(episodes):", len(dataset.episodes))
    print('\n\n\n\n\n')





    
    np.random.seed(42)
    dataset_split = dataset.get_splits(split_num)[split_id]
    with torch.no_grad():
        evaluate_agent_ovon(config, split_id, dataset_split, model_path, result_path, use_unified_prompt = use_unified_prompt)


    # # check if splits is non-overlaped
    # test_cur_splits = [int(item.episode_id) for item in dataset_split.episodes]
    # with open(f"test_cur_splits_{split_id}.txt", "w") as f:
    #     for item in test_cur_splits:
    #         f.write(str(item) + "\n")
  



if __name__ == "__main__":

    main()
