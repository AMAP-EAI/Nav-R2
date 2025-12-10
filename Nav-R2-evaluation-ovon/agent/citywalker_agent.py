import json
import random
import numpy as np
import time
try:
    from habitat import Env
    from habitat.core.agent import Agent
    import imageio
    from habitat.utils.visualizations import maps
except:
    pass
from tqdm import trange
import os
import re
import torch
import cv2
from PIL import Image
import time
from scipy.spatial.transform import Rotation as R
from safetensors.torch import load_file
import random
import oss2
import io
from datetime import datetime
from PIL import Image
from copy import deepcopy

from transformers import AutoProcessor, AutoTokenizer, AutoConfig, Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration, AutoModelForImageTextToText
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration__Customed_by_XWT
from qwen_vl_utils import process_vision_info



SAVE_RENDER_IMG = False
# SAVE_RENDER_IMG = True
PREDICT_SCALE=0.3
MAX_HISTORY_FRAMES=20

INPUT_IMG_SIZE = (640, 520)
HISTORY_RESIZE_RATIO= 1/4

MODEL_TYPE = 'ActionTrunkV2'
NUM_ACTION_TRUNK = 5
NUM_EXCUTE_ACTION_IN_TRUNK = 1
# NUM_EXCUTE_ACTION_IN_TRUNK = 5

obj_goal_template = ["Move through the environment to discover a {}. Your task is complete when you're directly facing it."] 

IS_FRONT_VIEW_ONLY = False
IS_FRONT_VIEW_ONLY = True
if IS_FRONT_VIEW_ONLY:
    NUM_CURRENT_IMAGE = 1

def get_model_name_from_path(model_path):
    return '/'.join(model_path.split('/')[-3:])

def evaluate_agent_ovon(config, split_id, dataset, model_path, result_path, use_unified_prompt:int = 0) -> None:

    env = Env(config, dataset)
    model_name = get_model_name_from_path(model_path)
    result_path = os.path.join(result_path, model_name)


    agent = Citywalker_Agent(model_path, result_path, use_unified_prompt=use_unified_prompt)

    num_episodes = len(env.episodes)
    
    EARLY_STOP_ROTATION = config.EVAL.EARLY_STOP_ROTATION
    EARLY_STOP_STEPS = config.EVAL.EARLY_STOP_STEPS

    
    target_key = {"distance_to_goal", "success", "spl"}

    count = 0
    is_collision = False

    for _ in trange(num_episodes, desc=config.EVAL.IDENTIFICATION+"-{}".format(split_id)):
        try:
            obs = env.reset()
            scene_name = os.path.basename(env.current_episode.scene_id).split('.')[0]
            env.current_episode.episode_id = f'{scene_name}_{env.current_episode.episode_id}'
            iter_step = 0

            agent.reset()

            target_object = env.current_episode.object_category
            instruction = random.choice(obj_goal_template).format(target_object)


            print(env.current_episode.episode_id)
            if os.path.exists(
                os.path.join(os.path.join(result_path, "log"),"stats_{}.json".format(env.current_episode.episode_id))
            ):
                print("evaled already...", os.path.join(os.path.join(result_path, "log"),"stats_{}.json".format(env.current_episode.episode_id)))
                continue


            continuse_rotation_count = 0
            continuse_collision_count = 0
            last_dtg = 999        
            action_list = []
            action_id_2_action_text_map = {
                0: "stop",
                1: "forward",
                2: "left",
                3: "right",
                "STOP": "stop"
            }
            action_text_list = []
            instruction_text = instruction
            per_action_seconds_cost_list = []
            while not env.episode_over:
                info = env.get_metrics()

                if info['collisions'] is not None and info['collisions']['is_collision']:
                    is_collision = True
                    # break
                
                if info["distance_to_goal"] != last_dtg:
                    last_dtg = info["distance_to_goal"]
                    continuse_rotation_count=0
                    continuse_collision_count = 0
                else :
                    continuse_rotation_count +=1 
                
                obs['pose'] = {'position':env._sim.get_agent_state().position.tolist(),
                                'rotation':[env._sim.get_agent_state().rotation.w,
                                            env._sim.get_agent_state().rotation.x,
                                            env._sim.get_agent_state().rotation.y,
                                            env._sim.get_agent_state().rotation.z]}
                obs["instruction"] = {"text":instruction}
                wp_pred, arrive_pred = None, None
                with torch.no_grad():
                    time_before_act = time.time()
                    action = agent.act(obs, info, env.current_episode.episode_id)
                    time_after_act = time.time()
                    time_act_cost = time_after_act - time_before_act
                    per_action_seconds_cost_list.append(time_act_cost)
                    wp_pred, arrive_pred = action['wp_pred'] if "wp_pred" in action else wp_pred, action['arrive_pred'] if "arrive_pred" in action else arrive_pred
                if MODEL_TYPE == 'ActionTrunkV2':
                    if continuse_rotation_count > 0 and action['action'] == 1:
                        continuse_collision_count += 1
                    else:
                        continuse_collision_count = 0
                    
                    if continuse_collision_count > 5:
                        #随机2，3一次
                        action = {"action": random.randint(2, 3)}
                        print(f'episode: {env.current_episode.episode_id} because of continuse_rotation_count={continuse_rotation_count} and continuse_collision_count={continuse_collision_count}, use random action: {action}')
                        continuse_collision_count = 0
                    pass
                    
                else:
                    if action['arrive_pred'] >=-1 or np.max(np.linalg.norm(action['action'],axis=1)) < 0.2:
                        action = {"action": 0}
                    else:
                        select_way_point_idx = 1
                        way_point_loc = action['action'][select_way_point_idx,:]

                        distance = np.linalg.norm(way_point_loc)
                        theta = np.arctan2(-way_point_loc[0], way_point_loc[1])
                        action = {"action": "GO_TOWARD_POINT", "action_args": {"theta": theta, "r": distance}}
                    print(f'step: {iter_step}, action: {action}')

                if continuse_rotation_count > EARLY_STOP_ROTATION or iter_step>EARLY_STOP_STEPS:
                    action = {"action": 0}     
                obs = env.step(action)


                action_id_2_text_map = {
                    0: "stop",
                    1: "forward",
                    2: "left", 
                    3: "right"
                }
                action['action_text'] = action_id_2_text_map[action['action']]
                action_text_list.append(action['action_text'])
                # print(action)
                iter_step+=1

            info = env.get_metrics()
            result_dict = dict()
            result_dict = {k: info[k] for k in target_key if k in info}
            result_dict["is_collision"] = is_collision
            result_dict["id"] = env.current_episode.episode_id
            result_dict["instruction_text"] = instruction_text
            
            count+=1

            result_dict['action_text_list'] = action_text_list
            
            print("action_text_list:\t", action_text_list)
            print("per action avg seconds: {}s".format(sum(per_action_seconds_cost_list) / len(per_action_seconds_cost_list)))
            with open(os.path.join(os.path.join(result_path, "log"),"stats_{}.json".format(env.current_episode.episode_id)), "w") as f:
                json.dump(result_dict, f, indent=4)
        except Exception as e:
            print("error....")
            print(e)
            print(scene_name)
            print("error....") 
            print()

class QwenActionModel():
    def __init__(self, model_path, use_unified_prompt = False):
        self.use_unified_prompt = use_unified_prompt
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.processor = AutoProcessor.from_pretrained(model_path)

        self.nav_version = 'special_token'
    
        config = AutoConfig.from_pretrained(model_path)
        if config.model_type == 'qwen2_vl':
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype="auto",
                device_map="auto",
                attn_implementation="flash_attention_2"
            )
        elif config.model_type == 'qwen2_5_vl':
            # self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model = Qwen2_5_VLForConditionalGeneration__Customed_by_XWT.from_pretrained(
                model_path,
                torch_dtype="auto",
                device_map="auto",
                attn_implementation="flash_attention_2"
            )
            self.model.set_tokenizer(self.tokenizer)
        else:
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_path,
                torch_dtype="auto",
                device_map="auto",
                attn_implementation="flash_attention_2"
            )
    @staticmethod
    def qwen_data_pack(images, user_content):
        content = []
        for idx,image in enumerate(images):
            if idx >= len(images) - NUM_CURRENT_IMAGE:
                # 不需要resize的当前画面，可能是左中右三个视角，可能是中视角
                cur_json = {
                    "type": "image",
                    "image": image,
                    "resized_height": INPUT_IMG_SIZE[1],
                    "resized_width": INPUT_IMG_SIZE[0],
                }
            else:
                # 需要resize的历史帧
                cur_json = {
                    "type": "image",
                    "image": image,
                    "resized_height": INPUT_IMG_SIZE[1]*HISTORY_RESIZE_RATIO,
                    "resized_width": INPUT_IMG_SIZE[0]*HISTORY_RESIZE_RATIO,
                }
            content.append(cur_json)
        content.append({
            "type": "text",
            "text": user_content,
        })
        messages = [
            {
                "role": "user",
                "content": content
            }
        ]
        return messages
    
    def qwen_infer(self, messages, max_new_tokens = 5):
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) # 把图片和文本拼接成一个完整的prompt，并且每张图片都用<|vision_start|><|vision_end|>包裹，然后prompt会包含很多个子部分的文本，每个子部分都用<|im_strat|><|im_end|>包裹
        # 但这里其实并没有在history view和front view的位置增加image token，而是在下面的if语句里才通过replace函数新增的

        # 计时
        if self.nav_version == 'special_token':
            text = text.replace('<|vision_start|><|image_pad|><|vision_end|>','')
            num_image = len(messages[0]['content']) - 1
            num_current_image = NUM_CURRENT_IMAGE
            num_history_image = num_image - num_current_image

            HISTORICAL_MEMORY_STRING_____BEFORE_IMAGE_SPECIAL_TOKENS = "Your historical front-view image(s) is(are): "
            if num_history_image > 0:
                historical_image_token_special_tokens_string = ''.join(['<|vision_start|><|image_pad|><|vision_end|>'] * num_history_image)
                historical_image_token_special_tokens_string_start_index = text.rfind(HISTORICAL_MEMORY_STRING_____BEFORE_IMAGE_SPECIAL_TOKENS) + len(HISTORICAL_MEMORY_STRING_____BEFORE_IMAGE_SPECIAL_TOKENS)
                text = text[:historical_image_token_special_tokens_string_start_index] + historical_image_token_special_tokens_string + text[historical_image_token_special_tokens_string_start_index:]
            else:
                history_img_str = "Your do not have any historical pictures right now."
                history_str_pos_start = text.rfind(HISTORICAL_MEMORY_STRING_____BEFORE_IMAGE_SPECIAL_TOKENS)
                history_str_pos_end = text.rfind(HISTORICAL_MEMORY_STRING_____BEFORE_IMAGE_SPECIAL_TOKENS) + len(HISTORICAL_MEMORY_STRING_____BEFORE_IMAGE_SPECIAL_TOKENS)
                text = text[: history_str_pos_start] + history_img_str + text[history_str_pos_end: ]
            if not IS_FRONT_VIEW_ONLY:
                assert False, 'only support front-view, please check IS_FRONT_VIEW_ONLY:\t' + str(IS_FRONT_VIEW_ONLY)
            else:
                FRONT_VIEW_STRING = "Your current front-view image is:"
                text = text.replace(FRONT_VIEW_STRING, FRONT_VIEW_STRING + " " + '<|vision_start|><|image_pad|><|vision_end|>')

        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(text=text, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt") # 会进入transformers/models/qwen2_5_vl/processing_qwen2_5_vl.py(170)
        inputs = inputs.to("cuda")
        generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens) 
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        return output_text[0]


class QwenModel():
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

class Citywalker_Agent():
    def __init__(self, model_path, result_path, require_map=True, use_unified_prompt:int = 0):
        self.use_unified_prompt = use_unified_prompt if type(use_unified_prompt) == bool else (False if use_unified_prompt == 0 else True)
        print("Initialize Qwen")
        
        self.result_path = result_path
        self.require_map = require_map

        if not self.result_path is None:
            os.makedirs(self.result_path, exist_ok=True)
            os.makedirs(os.path.join(self.result_path, "log"), exist_ok=True)
            os.makedirs(os.path.join(self.result_path, "video"), exist_ok=True)
            os.makedirs(os.path.join(self.result_path, "map_vis"), exist_ok=True)
            os.makedirs(os.path.join(self.result_path, "render_img"), exist_ok=True)

        assert self.use_unified_prompt is True, ''
        if self.use_unified_prompt:
            if MODEL_TYPE == 'ActionTrunkV2':
                self.model = QwenActionModel(model_path, use_unified_prompt = self.use_unified_prompt)

                OUTPUT_THINK_INSTRUCTION_CUE = "Please output your thinking process."
                DO_NOT_OUTPUT_THINK_INSTRUCTION_CUE = "Please DO NOT output your thinking process."

                self.prompt_template = """As an autonomous navigation robot, use the provided historical and current front-view images to complete your task. You should either follow the given instructions and stop when the specified condition is met, or explore the unknown environment to locate the designated object.\nBased on this information, determine your next {num_action_trunck} actions using any combination of <|left|>, <|right|>, <|forward|>, and <|stop|>. Output <|stop|> when the mission is complete. Example outputs (separated by commas) include: <|left|><|forward|><|right|><|forward|><|stop|>, <|forward|><|forward|><|forward|><|left|><|forward|>, or simply <|stop|>. \nYour historical front-view image(s) is(are): {history_img_string}.\nYour current front-view image is: {current_img_string}.\n<|I_AM_MISSON_START_TOKEN|>Your mission is: {instruction}<|NAV|>\n\n"""
                self.prompt_template = DO_NOT_OUTPUT_THINK_INSTRUCTION_CUE + "\n" + self.prompt_template
                # self.prompt_template = OUTPUT_THINK_INSTRUCTION_CUE + "\n" + self.prompt_template

                I_AM_A_SPECIAL_TOKEN_INDICATING_OVON_TASK = "<|I_AM_A_SPECIAL_TOKEN_INDICATING_OVON_TASK|>"
                self.prompt_template = self.prompt_template + I_AM_A_SPECIAL_TOKEN_INDICATING_OVON_TASK

            else:
                assert False, 'not implemented right now, please check the value of MODEL_TYPE parameter.'
            pass

        print("Initialization Complete")

        
        self.last_infer_result = ''
        self.history_rgb_tensor = None
        
        self.rgb_list = []
        self.pose_list = []
        self.topdown_map_list = []
        self.count_id = 0
        self.reset()


    def process_images(self, rgb_list):
        
        start_img_index = 0
        
        if self.history_rgb_tensor is not None:
            start_img_index = self.history_rgb_tensor.shape[0]
        
        batch_image = np.asarray(rgb_list[start_img_index:])
        video = self.image_processor.preprocess(batch_image, return_tensors='pt')['pixel_values'].half().cuda()

        if self.history_rgb_tensor is None:
            self.history_rgb_tensor = video
        else:
            self.history_rgb_tensor = torch.cat((self.history_rgb_tensor, video), dim = 0)
        

        return [self.history_rgb_tensor]

    def action_to_waypoint(self, action_index,action_scalar): # 包含起点0,0
        turnning_fake_len = 0.01

        if isinstance(action_index,list):
            arrive_pred = 0
            wp_pred_ = [np.array([0.,0.])]
            cur_heading = 0
            for idx,action in enumerate(action_index):
                if action == 0:
                    if idx == 0:
                        arrive_pred = 1
                    break
                elif action == 1:
                    wp_pred_.append(wp_pred_[-1] + 0.25 * np.array([-np.sin(cur_heading),np.cos(cur_heading)]))
                elif action == 2:
                    cur_heading += 30/180*np.pi
                    wp_pred_.append(wp_pred_[-1] + turnning_fake_len * np.array([-np.sin(cur_heading),np.cos(cur_heading)]))
                elif action == 3:
                    cur_heading -= 30/180*np.pi
                    wp_pred_.append(wp_pred_[-1] + turnning_fake_len * np.array([-np.sin(cur_heading),np.cos(cur_heading)]))
        else:
            if action_index is None:
                print("action_index is None")
                arrive_pred =1
                wp_pred_ = np.array([0,0])

            elif action_index == 0:
                arrive_pred = 1
                wp_pred_ = np.array([0,0])
            elif action_index == 1:
                arrive_pred = 0
                wp_pred_ = np.array([0,action_scalar])
            elif action_index == 2:
                arrive_pred = 0
                wp_pred_ = np.array([-np.sin(action_scalar/180*np.pi) * turnning_fake_len, np.cos(action_scalar/180*np.pi) * turnning_fake_len])
            elif action_index == 3:
                arrive_pred = 0
                wp_pred_ = np.array([np.sin(action_scalar/180*np.pi) * turnning_fake_len, np.cos(action_scalar/180*np.pi) * turnning_fake_len])
            wp_pred_ = np.insert(wp_pred_,0,[0,0],axis=0)
            

        wp_pred = np.array(wp_pred_).reshape(-1,2)

        return wp_pred, arrive_pred

    def extract_result(self,output):
        # id: 0-stop, 1 move forward, 2 turn left, 3 turn right
        self.last_infer_result = output    
        if MODEL_TYPE == 'ActionTrunkV2':
            try:
                output = output.replace('<|im_end|>', '')
                pattern = r'<\|([^|]+)\|>'
                actions_text = re.findall(pattern, output)
                actions = []
                for action_text in actions_text:
                    if "forward" in action_text:
                        actions.append(1)
                    elif "left" in action_text:
                        actions.append(2)
                    elif "right" in action_text:
                        actions.append(3)
                    elif "stop" in action_text:
                        actions.append(0)
                return actions, None
            except:
                print(f"Error extracting actions: {output}")
                return None, None
        else:
            assert False, ''

    def addtext(self, image, instuction, navigation):
        h, w = image.shape[:2]
        new_height = h + 150
        new_image = np.zeros((new_height, w, 3), np.uint8)
        new_image.fill(255)  
        new_image[:h, :w] = image

        font = cv2.FONT_HERSHEY_SIMPLEX
        textsize = cv2.getTextSize(instuction, font, 0.5, 2)[0]
        textY = h + (50 + textsize[1]) // 2

        y_line = textY + 0 * textsize[1]

        words = instuction.split(' ')
        max_width = new_image.shape[1]
        x = 10
        line = ""

        for word in words:

            test_line = line + ' ' + word if line else word
            test_line_size, _ = cv2.getTextSize(test_line, font, 0.5, 2)

            if test_line_size[0] > image.shape[1] - x:
                new_image = cv2.putText(new_image, line, (x, y_line ), font, 0.5, (0, 0, 0), 2)
                line = word
                y_line += textsize[1]+5
            else:
                line = test_line

        if line:
            new_image = cv2.putText(new_image, line, (x, y_line), font, 0.5, (0, 0, 0), 2)

        y_line = y_line + 1 * textsize[1] + 10

        words = navigation.split(' ')
        max_width = new_image.shape[1]
        x = 10
        line = ""

        for word in words:

            test_line = line + ' ' + word if line else word
            test_line_size, _ = cv2.getTextSize(test_line, font, 0.5, 2)

            if test_line_size[0] > image.shape[1] - x:
                cv2.putText(new_image, line, (x, y_line ), font, 0.5, (0, 0, 0), 2)
                line = word
                y_line += textsize[1]+5
            else:
                line = test_line

        if line:
            new_image = cv2.putText(new_image, line, (x, y_line), font, 0.5, (0, 0, 0), 2)

        return new_image

    def reset(self):
                
        if self.require_map:
            if len(self.topdown_map_list)!=0:
                output_video_path = os.path.join(self.result_path, "video","{}.gif".format(self.episode_id))

                imageio.mimsave(output_video_path, self.topdown_map_list)

        self.history_rgb_tensor = None
        self.transformation_list = []
        self.rgb_list = []
        self.pose_list = []
        self.image_indices = []
        self.topdown_map_list = []
        self.total_frame_count = 0
        self.last_action = None
        self.count_id += 1
        self.count_stop = 0
        self.pending_action_list = []

        self.model.model.clear_memory_frame_embeds_list()

        self.first_forward = False
    
    def generate_infer_prompt_front_view_only(self,instruction):
        cur_prompt = deepcopy(self.prompt_template)

        images = self.rgb_list

        if not self.use_unified_prompt:
            assert False, 'not supported! self.use_unified_prompt should be True'
        else:
            history_img_string = ''
            current_img_string = ''
            cur_prompt = cur_prompt.format(
                num_action_trunck=NUM_ACTION_TRUNK, 
                history_img_string=history_img_string, 
                current_img_string=current_img_string, 
                instruction=instruction, 
            )
        # 此时cur_prompt不含有<image>特殊token
        return self.model.qwen_data_pack(images, cur_prompt)

    def add_frame_front_view_only(self, rgbs, pose):
        rgbs_new = []
        for rgb in rgbs:
            if isinstance(rgb, np.ndarray):
                rgb = Image.fromarray(rgb).resize((INPUT_IMG_SIZE[0], INPUT_IMG_SIZE[1]))
            else:
                rgb = rgb.resize((INPUT_IMG_SIZE[0], INPUT_IMG_SIZE[1]))
            rgbs_new.append(rgb)
        
        
        self.rgb_list.extend(rgbs_new)
        self.pose_list.extend([pose] * len(rgbs_new))
        self.image_indices.extend([self.total_frame_count] * len(rgbs_new))
        self.total_frame_count += 1
        if len(self.rgb_list) > NUM_CURRENT_IMAGE: 
            self.rgb_list[-1 - NUM_CURRENT_IMAGE] = self.rgb_list[-1 - NUM_CURRENT_IMAGE].resize((int(INPUT_IMG_SIZE[0]*HISTORY_RESIZE_RATIO), int(INPUT_IMG_SIZE[1]*HISTORY_RESIZE_RATIO)))
            
        if len(self.rgb_list) > MAX_HISTORY_FRAMES + NUM_CURRENT_IMAGE:
            min_interval_idx = np.argmin(np.diff(self.image_indices[:-NUM_CURRENT_IMAGE]))
            self.rgb_list.pop(min_interval_idx+1)
            self.pose_list.pop(min_interval_idx+1)
            self.image_indices.pop(min_interval_idx+1)

    def save_rgb(self, path):
        output_dir = os.path.join(path, str(self.count_id), str(self.total_frame_count))
        os.makedirs(output_dir, exist_ok=True)
        for idx, rgb in enumerate(self.rgb_list):
            output_img_path = os.path.join(output_dir, "{}.png".format(idx))
            rgb.save(output_img_path)
        
        with open(os.path.join(output_dir, "last_infer_result.txt"), "w") as f:
            f.write(self.last_infer_result)

    def act(self, observations, info, episode_id):

        self.episode_id = episode_id
        cur_episode_folder = os.path.join(self.result_path, "render_img", str(episode_id))
        os.makedirs(cur_episode_folder, exist_ok=True)

        cur_episode_vis_folder = os.path.join(self.result_path, "map_vis", str(episode_id))
        os.makedirs(cur_episode_vis_folder, exist_ok=True)

        if self.model.nav_version == 'special_token':
            rgb = observations["front"]
            pose = observations["pose"]
        else:
            rgb = observations["rgb"]
            pose = observations["pose"]
        
        # 保存当前图像（使用即将分配的原始帧索引）
        current_frame_index = self.total_frame_count
        output_img_path = os.path.join(cur_episode_folder, "{}.png".format(current_frame_index))
        if SAVE_RENDER_IMG:
            cv2.imwrite(output_img_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        
        if self.model.nav_version == 'special_token':
            if IS_FRONT_VIEW_ONLY:
                self.add_frame_front_view_only([rgb], pose)
            else:
                assert False, ""
        else:
            assert False, ""

        if self.require_map:
            if 'top_down_map_vlnce' in info:
                top_down_map = maps.colorize_draw_agent_and_fit_to_height(info["top_down_map_vlnce"], rgb.shape[0])
            else:
                top_down_map = maps.colorize_draw_agent_and_fit_to_height(info["top_down_map"], rgb.shape[0])
            output_im = np.concatenate((rgb, top_down_map), axis=1)



        navigation_qs = ''
        if IS_FRONT_VIEW_ONLY:
            navigation_qs = self.generate_infer_prompt_front_view_only(observations["instruction"]["text"])
            pass
        else:
            assert False, "front-view supported only, check variable IS_FRONT_VIEW_ONLY"

        if len(self.pending_action_list) != 0 :
            temp_action = self.pending_action_list.pop(0)
            if self.require_map:   
                img = self.addtext(output_im, observations["instruction"]["text"], "Pending action: {}".format(temp_action))
                self.topdown_map_list.append(img)
                if SAVE_RENDER_IMG:
                    cv2.imwrite(os.path.join(cur_episode_vis_folder, "{}.png".format(current_frame_index)), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            self.model.qwen_infer(navigation_qs, max_new_tokens = 1)
            
            return {"action": temp_action}


        if MODEL_TYPE == 'ActionTrunkV2':
            result_text = self.model.qwen_infer(navigation_qs, max_new_tokens = 2)

            action_index, _ = self.extract_result(result_text)
            wp_pred, arrive_pred = None, None
            try:
                wp_pred, arrive_pred = self.action_to_waypoint(action_index=action_index, action_scalar=None)
            except Exception as e:
                print(e)

            if self.require_map:
                img = self.addtext(output_im, observations["instruction"]["text"], result_text)
                self.topdown_map_list.append(img)
                if SAVE_RENDER_IMG:
                    cv2.imwrite(os.path.join(cur_episode_vis_folder, "{}.png".format(current_frame_index)), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            if action_index is None or \
                len(action_index) == 0: 
                print('===========\nrandom action\n======')
                self.pending_action_list.append(random.randint(1, 3))
                print(self.pending_action_list)
                print("result_text:\t", result_text)
            else:
                for idx in range(len(action_index)):
                    self.pending_action_list.append(action_index[idx])
                    if idx == NUM_EXCUTE_ACTION_IN_TRUNK-1:
                        break
            return {"action": self.pending_action_list.pop(0), "arrive_pred": arrive_pred, "wp_pred": wp_pred}
        else:
            assert False, "not supported, only ActionTrunkV2 is supported but got " + MODEL_TYPE

