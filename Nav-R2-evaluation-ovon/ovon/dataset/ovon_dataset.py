#!/usr/bin/env python3

import json
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

import attr
from habitat.core.registry import registry
from habitat.core.simulator import AgentState, ShortestPathPoint
from habitat.core.utils import DatasetFloatJSONEncoder
from habitat.datasets.pointnav.pointnav_dataset import (
    CONTENT_SCENES_PATH_FIELD,
    DEFAULT_SCENE_PATH_PREFIX,
    PointNavDatasetV1,
)
from habitat.tasks.nav.object_nav_task import (
    ObjectGoal,
    ObjectGoalNavEpisode,
    ObjectViewLocation,
)

if TYPE_CHECKING:
    from omegaconf import DictConfig


@attr.s(auto_attribs=True)
class OVONObjectViewLocation(ObjectViewLocation):
    r"""OVONObjectViewLocation

    Args:
        raidus: radius of the circle
    """

    radius: Optional[float] = None


@attr.s(auto_attribs=True, kw_only=True)
class OVONEpisode(ObjectGoalNavEpisode):
    r"""OVON Episode

    :param children_object_categories: Category of the object
    """

    children_object_categories: Optional[List[str]] = []


@registry.register_dataset(name="OVON-v1")
class OVONDatasetV1(PointNavDatasetV1):
    r"""
    Class inherited from PointNavDataset that loads Open-Vocab
    Object Navigation dataset.
    """

    episodes: List[OVONEpisode] = []  # type: ignore
    content_scenes_path: str = "{data_path}/content/{scene}.json.gz"
    goals_by_category: Dict[str, Sequence[ObjectGoal]]

    @staticmethod
    def dedup_goals(dataset: Dict[str, Any]) -> Dict[str, Any]:
        if len(dataset["episodes"]) == 0:
            return dataset

        goals_by_category = {}
        for i, ep in enumerate(dataset["episodes"]):
            # Get the category from the first goal
            dataset["episodes"][i]["object_category"] = ep["goals"][0][
                "object_category"
            ]
            ep = OVONEpisode(**ep)

            # Store unique goals under their key
            goals_key = ep.goals_key
            if goals_key not in goals_by_category:
                goals_by_category[goals_key] = ep.goals

            # Store a reference to the shared goals
            dataset["episodes"][i]["goals"] = []

        dataset["goals_by_category"] = goals_by_category

        return dataset

    def to_json(self) -> str:
        for i in range(len(self.episodes)):
            self.episodes[i].goals = []

        result = DatasetFloatJSONEncoder().encode(self)

        for i in range(len(self.episodes)):
            goals = self.goals_by_category[self.episodes[i].goals_key]
            if not isinstance(goals, list):
                goals = list(goals)
            self.episodes[i].goals = goals

        return result

    def __init__(self, config: Optional["DictConfig"] = None) -> None:
        self.goals_by_category = {}
        super().__init__(config)
        self.episodes = list(self.episodes)

    @staticmethod
    def __deserialize_goal(serialized_goal: Dict[str, Any]) -> ObjectGoal:
        g = ObjectGoal(**serialized_goal)
        g.object_id = int(g.object_id.split("_")[-1])

        for vidx, view in enumerate(g.view_points):
            view_location = OVONObjectViewLocation(**view)  # type: ignore
            view_location.agent_state = AgentState(
                **view_location.agent_state  # type: ignore
            )
            g.view_points[vidx] = view_location

        return g

    def from_json(self, json_str: str, scenes_dir: Optional[str] = None) -> None:

        # gif_id_to_eval = [
        #     "4ok3usBNeis_1129",
        #     "4ok3usBNeis_1136",
        #     "4ok3usBNeis_1684",
        #     "4ok3usBNeis_1849",
        #     "4ok3usBNeis_1908",
        #     "4ok3usBNeis_1959",
        #     "4ok3usBNeis_3630",
        #     "4ok3usBNeis_4010",
        #     "4ok3usBNeis_4177",
        #     "4ok3usBNeis_4699",
        #     "4ok3usBNeis_6622",
        #     "4ok3usBNeis_1849",
        #     "4ok3usBNeis_3574",
        #     "4ok3usBNeis_3825",
        #     "4ok3usBNeis_5220",
        #     "4ok3usBNeis_5285",
        #     "4ok3usBNeis_5315",
        #     "4ok3usBNeis_6145",
        #     "4ok3usBNeis_6193",
        #     "4ok3usBNeis_6927"
        # ]




        deserialized = json.loads(json_str)
        if CONTENT_SCENES_PATH_FIELD in deserialized:
            self.content_scenes_path = deserialized[CONTENT_SCENES_PATH_FIELD]

        if len(deserialized["episodes"]) == 0:
            return

        if "goals_by_category" not in deserialized:
            deserialized = self.dedup_goals(deserialized)

        for k, v in deserialized["goals_by_category"].items():
            self.goals_by_category[k] = [self.__deserialize_goal(g) for g in v]

        # import pdb;pdb.set_trace()
        for i, episode in enumerate(deserialized["episodes"]):
            episode = OVONEpisode(**episode)
            episode.goals = self.goals_by_category[episode.goals_key]  # noqa

            if scenes_dir is not None:
                if episode.scene_id.startswith(DEFAULT_SCENE_PATH_PREFIX):
                    episode.scene_id = episode.scene_id[
                        len(DEFAULT_SCENE_PATH_PREFIX) :
                    ]

                episode.scene_id = os.path.join(scenes_dir, episode.scene_id)

            if episode.shortest_paths is not None:
                for path in episode.shortest_paths:
                    for p_index, point in enumerate(path):
                        if point is None or isinstance(point, (int, str)):
                            point = {
                                "action": point,
                                "rotation": None,
                                "position": None,
                            }

                        path[p_index] = ShortestPathPoint(**point)




            # scene_name = os.path.basename(episode.scene_id).split('.')[0]
            # scene_name_and_episode_id = f'{scene_name}_{episode.episode_id}'
            # if scene_name_and_episode_id not in gif_id_to_eval:
            #     continue




            self.episodes.append(episode)  # type: ignore [attr-defined]
        print()
    





    
    def filter_existing_episodes(self, result_path):
        # import pdb;pdb.set_trace()
        dp1 = os.path.dirname(result_path)
        dp2 = os.path.dirname(dp1)
        dp3 = os.path.dirname(dp2)
        dp4 = os.path.dirname(dp3)
        log_map_render_video_folder_path = result_path + os.path.dirname(result_path[len(dp4):])
        # log_map_render_video_folder_path = result_path + os.path.dirname(result_path[len(dp3):])
        log_folder_path = log_map_render_video_folder_path + "/log"
        print("log_folder_path:\t", log_folder_path)
        new_episode_list = []

        # exist_filename_list = [item for item in os.listdir(log_folder_path) if item.endswith(".json")]
        all_filename_list = []
        for episode in self.episodes:
            scene_name = os.path.basename(episode.scene_id).split('.')[0]
            scene_name_and_episode_id = f'{scene_name}_{episode.episode_id}'
            # if scene_name_and_episode_id
            all_filename_list.append("stats_" + scene_name_and_episode_id + ".json")
            if os.path.exists(log_folder_path + "/stats_" + scene_name_and_episode_id + ".json"):
                continue
            else:
                new_episode_list.append(episode)
        # import pdb;pdb.set_trace()
        # len(set(all_filename_list) - set(exist_filename_list))
        import copy
        self.episodes = copy.deepcopy(new_episode_list)

