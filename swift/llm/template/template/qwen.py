# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Tuple

import torch
import torch.nn.functional as F

from swift.llm import to_device, to_float_dtype
from swift.utils import get_env_args, is_deepspeed_enabled
from ..base import Template
from ..constant import LLMTemplateType, MLLMTemplateType
from ..register import register_template
from ..template_inputs import StdTemplateInputs
from ..template_meta import TemplateMeta
from ..utils import Context, Word, findall
from ..vision_utils import load_audio, load_batch, load_video_ovis2
from .llama import Llama3TemplateMeta
from .utils import DEFAULT_SYSTEM, ChatmlTemplateMeta


@dataclass
class QwenTemplateMeta(ChatmlTemplateMeta):
    default_system: Optional[str] = DEFAULT_SYSTEM
    auto_add_bos: bool = False
    stop_words: List[Word] = field(default_factory=lambda: ['<|endoftext|>'])
    agent_template: str = 'hermes'


@dataclass
class Qwen2_5TemplateMeta(QwenTemplateMeta):
    default_system: Optional[str] = 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.'


@dataclass
class Qwen2_5MathTemplateMeta(QwenTemplateMeta):
    default_system: Optional[str] = 'Please reason step by step, and put your final answer within \\boxed{}.'


qwq_preview_system = ('You are a helpful and harmless assistant. You are Qwen developed by Alibaba. '
                      'You should think step-by-step.')

register_template(QwenTemplateMeta(LLMTemplateType.qwen))
register_template(Qwen2_5TemplateMeta(LLMTemplateType.qwen2_5))
register_template(QwenTemplateMeta(LLMTemplateType.qwq_preview, default_system=qwq_preview_system))


class ThinkingTemplate(Template):

    def _swift_prepare_messages(self, messages):
        super()._swift_prepare_messages(messages)
        for i, message in enumerate(messages):
            if message['role'] == 'assistant' and isinstance(message['content'], str) and i != len(messages) - 1:
                message['content'] = message['content'].split('</think>')[-1].strip()


register_template(
    QwenTemplateMeta(
        LLMTemplateType.qwq, default_system=None, response_prefix='<think>\n', template_cls=ThinkingTemplate))

# '<think>\n\n</think>\n\n'
register_template(QwenTemplateMeta(LLMTemplateType.qwen3, default_system=None, template_cls=ThinkingTemplate))


class Qwen3RerankerTemplate(Template):
    instruction = 'Given a web search query, retrieve relevant passages that answer the query'

    def _preprocess_inputs(self, inputs: StdTemplateInputs) -> None:
        super()._preprocess_inputs(inputs)
        query = inputs.messages[-2]['content']
        doc = inputs.messages[-1]['content']
        user_message = '<Instruct>: ' + self.instruction + '\n' + '<Query>: ' + query + '\n' + '<Document>: ' + doc
        inputs.messages[-2]['content'] = user_message
        inputs.messages.pop(-1)


qwen3_reranker_system = (
    'Judge whether the Document meets the requirements based on the Query and the Instruct provided. '
    'Note that the answer can only be \"yes\" or \"no\".')

register_template(
    QwenTemplateMeta(
        LLMTemplateType.qwen3_reranker,
        default_system=qwen3_reranker_system,
        response_prefix='<think>\n\n</think>\n\n',
        template_cls=Qwen3RerankerTemplate))

register_template(Qwen2_5MathTemplateMeta(LLMTemplateType.qwen2_5_math))


class QwenPRMTemplate(Template):
    cot_process_placeholder = '<extra_0>'

    def _preprocess_inputs(
        self,
        inputs: StdTemplateInputs,
    ) -> None:
        super()._preprocess_inputs(inputs)
        total_content = '\n'.join([message['content'] or '' for message in inputs.messages])
        if self.cot_process_placeholder not in total_content:
            inputs.messages[-1]['content'] = inputs.messages[-1]['content'] + self.cot_process_placeholder

    @staticmethod
    def make_step_rewards(logits, token_masks):
        probabilities = F.softmax(logits, dim=-1)
        probabilities = probabilities * token_masks.unsqueeze(-1)  # bs, seq_len, num_labels

        all_scores_res = []
        for i in range(probabilities.size(0)):
            sample = probabilities[i]  # seq_len, num_labels
            positive_probs = sample[sample != 0].view(-1, 2)[:, 1]  # valid_tokens, num_labels
            non_zero_elements_list = positive_probs.cpu().tolist()
            all_scores_res.append(non_zero_elements_list)
        return all_scores_res

    def decode_prm(self, input_ids: torch.Tensor, logits: torch.Tensor) -> Any:
        step_sep_id = self.tokenizer.encode(self.cot_process_placeholder)[0]
        token_masks = (input_ids == step_sep_id)
        return self.make_step_rewards(logits, token_masks)


register_template(Qwen2_5MathTemplateMeta(LLMTemplateType.qwen2_5_math_prm, template_cls=QwenPRMTemplate))


class QwenVLTemplate(Template):
    load_images = False

    @staticmethod
    def _load_image(image, load_images: bool):
        if not load_images and isinstance(image, str) and (image.startswith('data:') or len(image) > 200):
            load_images = True
        return Template._load_image(image, load_images)

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        assert media_type == 'image'
        if self.mode == 'lmdeploy':
            return [f'Picture {index + 1}: ', [-100], '\n']
        else:
            image = inputs.images[index]
            if self.mode == 'vllm':
                return [f'Picture {index + 1}: <img></img>\n']
            else:
                assert isinstance(image, str)
                return [f'Picture {index + 1}: <img>{image}</img>\n']

    def replace_ref(self, ref: str, index: int, inputs: StdTemplateInputs) -> List[Context]:
        return [f'<ref>{ref}</ref>']

    def replace_bbox(self, bbox: List[int], index: int, inputs: StdTemplateInputs) -> List[Context]:
        return [f'<box>{self._get_bbox_str(bbox)}</box>']


register_template(QwenTemplateMeta(MLLMTemplateType.qwen_vl, template_cls=QwenVLTemplate))


class QwenAudioTemplate(Template):

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        assert media_type == 'audio'
        audios = inputs.audios
        audio = audios[index]
        assert isinstance(audio, str)
        return [f'Audio {index + 1}:<audio>{audio}</audio>\n']

    def _tokenize(self, context, **tokenizer_kwargs):
        audio_info = self.processor.process_audio(context)
        return super()._tokenize(context, audio_info=audio_info)

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        text = ''.join([f'<audio>{audio}</audio>' for audio in inputs.audios])
        audio_info = self.processor.process_audio(text)
        if audio_info:
            tokenizer_kwargs = {'audio_info': audio_info}
            encoded.update(tokenizer_kwargs)
            encoded['tokenizer_kwargs'] = tokenizer_kwargs
        return encoded

    def _data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        res = super()._data_collator(batch, padding_to=padding_to)
        if batch[0].get('audio_info') is not None:
            res['audio_info'] = [b['audio_info'] for b in batch]
        return res


register_template(QwenTemplateMeta(MLLMTemplateType.qwen_audio, template_cls=QwenAudioTemplate))


class Qwen2AudioTemplate(Template):

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        assert media_type == 'audio'
        if not self.use_chat_template:
            return ['<|audio_bos|><|AUDIO|><|audio_eos|>\n']
        else:
            return [f'Audio {index + 1}: <|audio_bos|><|AUDIO|><|audio_eos|>\n']

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        if inputs.audios:
            sampling_rate = get_env_args('sampling_rate', int, self.processor.feature_extractor.sampling_rate)
            audios = load_batch(inputs.audios, load_func=partial(load_audio, sampling_rate=sampling_rate))
            audio_inputs = self.processor.feature_extractor(
                audios, sampling_rate=sampling_rate, return_attention_mask=True, return_tensors='pt')
            audio_inputs['feature_attention_mask'] = audio_inputs.pop('attention_mask')
            encoded.update(audio_inputs)
        return encoded

    def _data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        res = super()._data_collator(batch, padding_to=padding_to)
        input_features = [b['input_features'] for b in batch if b.get('input_features') is not None]
        feature_attention_mask = [
            b['feature_attention_mask'] for b in batch if b.get('feature_attention_mask') is not None
        ]
        if input_features:
            res['input_features'] = torch.concat(input_features)
            res['feature_attention_mask'] = torch.concat(feature_attention_mask)
        return res


register_template(QwenTemplateMeta(MLLMTemplateType.qwen2_audio, template_cls=Qwen2AudioTemplate))


class Qwen2VLTemplate(Template):
    image_token_id = 151655
    video_token_id = 151656
    placeholder_tokens = ['<|image_pad|>', '<|video_pad|>']
    version = 'v2'
    use_model = True

    SPECIAL_TOKEN_MISSIONSTARTTOKEN_customed_by_xwt = '<|I_AM_MISSON_START_TOKEN|>'
    SPECIAL_NAVTOKEN_customed_by_xwt = '<|NAV|>'
    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        # import pdb;pdb.set_trace()
        from qwen_vl_utils import fetch_image, fetch_video
        assert media_type in {'image', 'video'}
        if media_type == 'image':
            inputs.images[index] = fetch_image({'image': inputs.images[index]})
            if self.mode == 'lmdeploy':
                return ['<|vision_start|>', [-100], '<|vision_end|>']
            else:
                return ['<|vision_start|><|image_pad|><|vision_end|>']
        else:
            video = inputs.videos[index]
            if os.path.isdir(video):
                video = [os.path.join(video, fname) for fname in os.listdir(video)]
            video = fetch_video({'video': video})
            if isinstance(video, torch.Tensor):
                video = video.to(torch.uint8)
            inputs.videos[index] = video
            return ['<|vision_start|><|video_pad|><|vision_end|>']

    def replace_ref(self, ref: str, index: int, inputs: StdTemplateInputs) -> List[Context]:
        return [f'<|object_ref_start|>{ref}<|object_ref_end|>']

    def replace_bbox(self, bbox: List[int], index: int, inputs: StdTemplateInputs) -> List[Context]:
        return [f'<|box_start|>{self._get_bbox_str(bbox)}<|box_end|>']

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        processor = self.processor
        input_ids = encoded['input_ids']
        labels = encoded['labels']
        original_images = inputs.original_images
        images = inputs.images
        videos = inputs.videos
        for media_type in ['images', 'videos']:
            if locals()[media_type]:
                if media_type == 'images':
                    media_token = self.image_token_id
                    media_inputs = processor.image_processor( # 这里进入transformers/src/transformers/models/qwen2_vl/image_processing_qwen2_vl.py文件的Qwen2VLImageProcessor类的preprocess函数
                        images=images, videos=None, return_tensors='pt', do_resize=False, original_images = original_images)
                    media_grid_thw = media_inputs['image_grid_thw']
                else:
                    if hasattr(processor, 'video_processor'):
                        processor_func = processor.video_processor
                    else:
                        processor_func = processor.image_processor
                    media_inputs = processor_func(images=None, videos=videos, return_tensors='pt', do_resize=False)
                    media_grid_thw = media_inputs['video_grid_thw']
                    media_token = self.video_token_id
                    if self.version == 'v2_5':
                        from qwen_vl_utils import vision_process
                        media_inputs['second_per_grid_ts'] = [
                            processor.image_processor.temporal_patch_size / vision_process.FPS
                        ] * len(media_grid_thw)
                idx_list = findall(input_ids, media_token)
                merge_length = processor.image_processor.merge_size**2

                def _get_new_tokens(i):
                    token_len = (media_grid_thw[i].prod() // merge_length)
                    return [media_token] * token_len

                input_ids, labels = self._extend_tokens(input_ids, labels, idx_list, _get_new_tokens)
                encoded.update(media_inputs)

        encoded['input_ids'] = input_ids
        encoded['labels'] = labels
        return encoded

    def compute_loss_context(self, model, inputs):
        if 'real_position_ids' not in inputs:
            return super().compute_loss_context(model, inputs)
        if self.version == 'v2':
            from transformers.models.qwen2_vl import modeling_qwen2_vl as modeling_module
        elif self.version == 'v2_5':
            from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl as modeling_module
        elif self.version == 'omni':
            from transformers.models.qwen2_5_omni import modeling_qwen2_5_omni as modeling_module
        position_ids = inputs['position_ids']
        inputs['position_ids'] = inputs.pop('real_position_ids')
        return self._patch_flash_attention_forward(modeling_module, position_ids)

    # 原版的不带帧压缩的_post_encode版本
    def _post_encode(self, model, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if not self.is_training:
            return inputs



        input_ids = inputs['input_ids']              # [B, L]
        special_token_id_NAVTOKEN_customed_by_xwt = self.tokenizer.encode(self.SPECIAL_NAVTOKEN_customed_by_xwt)[0]
        NAVTOKEN_index = input_ids == special_token_id_NAVTOKEN_customed_by_xwt

        whether_apply_compressing_code_customed_by_XWT = True
        if NAVTOKEN_index.flatten().sum() > 0:
            # 是导航任务
            if whether_apply_compressing_code_customed_by_XWT:
                try:
                    return self._post_encode_compresshistoricalimages_and_concat_with_currentimage(model, inputs)
                except Exception as e:
                    print("********* ERROR ***********")
                    print("self._post_encode_compresshistoricalimages_and_concat_with_currentimage")
                    print(e)
                    print("********* ERROR ***********\n\n\n\n")

        # 不是导航任务，走vqa

        input_ids = inputs['input_ids']
        pixel_values = inputs.get('pixel_values')
        pixel_values_videos = inputs.get('pixel_values_videos')
        image_grid_thw = inputs.get('image_grid_thw')
        video_grid_thw = inputs.get('video_grid_thw')

        base_model = self.get_base_model(model)
        if hasattr(base_model.model, 'embed_tokens'):
            inputs_embeds = base_model.model.embed_tokens(input_ids)
        else:
            inputs_embeds = base_model.model.language_model.embed_tokens(input_ids)
        # import pdb;pdb.set_trace()
        dtype = model.visual.get_dtype() if self.version == 'v2' else model.visual.dtype
        if pixel_values is None and pixel_values_videos is None:  # plain-text
            if is_deepspeed_enabled():
                from PIL import Image
                images = [Image.new('RGB', (32, 32), (0, 0, 0))]
                media_inputs = self.processor.image_processor(images=images, videos=None, return_tensors='pt')
                device = input_ids.device
                media_inputs = to_device(media_inputs, device)
                pixel_values = media_inputs['pixel_values'].type(dtype)
                image_embeds = model.visual(pixel_values, grid_thw=media_inputs['image_grid_thw'])
                inputs_embeds += image_embeds.mean() * 0.
        else:
            if pixel_values is not None:
                # 在这里调用model内部封装好的函数，去维护memory、得到增强后的特征这些
                # 
                pixel_values = pixel_values.type(dtype)
                image_embeds = model.visual(pixel_values, grid_thw=image_grid_thw)
                image_mask = (input_ids == model.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds) # inputs_embeds在scatter前后的维度不变

            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(dtype)
                video_embeds = model.visual(pixel_values_videos, grid_thw=video_grid_thw)
                video_mask = (input_ids == model.config.video_token_id).unsqueeze(-1).expand_as(inputs_embeds)
                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        return {'inputs_embeds': inputs_embeds}



    def _post_encode_compresshistoricalimages_and_concat_with_currentimage(self, model, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        inputs.keys()
            dict_keys(['input_ids', 'labels', 'attention_mask', 'pixel_values', 'image_grid_thw', 'memory_images_pixel_values', 'current_image_vision_grid_thws', 'current_image_pixel_values', 'memory_images_vision_grid_thws', 'position_ids', 'future_images', 'action_label', 'id'])
        """

        # 这个版本不缩放原始rgb图，是在函数中对historical image的embedding调用model.pool进行压缩，再把压缩后的memory和current image拼接后，覆盖到输入序列embedding中的image token位置
        # 但是有bug，在训练到第2个step时，显卡会报显存异常访问的错.....
        # return self._post_encode_compresshistoricalimages_and_concat_with_currentimage__________do_not_resize_original_rgb_image__sequence_length_modified_bug_exists_when_iter_for_2_times(
        #     model, inputs
        # )

        # 这个版本缩放原始rgb图，但同时会把没有缩放的rgb图也传到这里
        # 在函数中对未缩放的图的image embedding调用model.pool进行压缩，pool后的每张history image的embedding数和resize rgb后得到的embedding数相同
        # 然后直接覆盖到序列中对应的位置
        return self._post_encode_compresshistoricalimages_and_concat_with_currentimage__________resize_original_rgb_image__sequence_length_the_same__pool_original_size_embedding_to_fit_the_same_length(
            model, inputs
        )


    def _post_encode_compresshistoricalimages_and_concat_with_currentimage__________resize_original_rgb_image__sequence_length_the_same__pool_original_size_embedding_to_fit_the_same_length(self, model, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        inputs.keys() 要想增加key-values对，得走swift/llm/template/template/qwen.py文件的_data_collator_mm_data函数去添加keys_want，函数返回的key-value就对应到这里的inputs里
            dict_keys(['input_ids', 'labels', 'attention_mask', 'pixel_values', 'image_grid_thw', 'memory_images_pixel_values', 'current_image_vision_grid_thws', 'current_image_pixel_values', 'memory_images_vision_grid_thws', 'position_ids', 'future_images', 'action_label', 'id'])
        """
        # import time
        # start_1 = time.time()
        if not self.is_training:
            return inputs
        # import pdb;pdb.set_trace()
        ##########
        ##########
        ##########
        ##########
        ##########
        # 硬编码
        image_embedding_count_per_image_after_modelvisual = 437 # 暂时写死为437，这个是未resize的原图经model.visual后的token数
        image_embedding_count_per_resized_image_after_modelvisual = 30 # 暂时写死为30，resize过后的原图经model.visual后的token数
        ##########
        ##########
        ##########
        ##########
        ##########

        device = inputs['input_ids'].device
        base_model = self.get_base_model(model)
        image_token_id = model.config.image_token_id

        input_ids = inputs['input_ids']              # [B, L]
        labels = inputs['labels']                    # [B, L]
        attention_mask = inputs['attention_mask']    # [B, L]
        position_ids = inputs['position_ids']        # [*, B, L]
        pixel_values = inputs['pixel_values']        # [43700, 1176]
        image_grid_thw = inputs['image_grid_thw']    # [25, 3]
        # assert all(torch.equal(x, image_grid_thw[0]) for x in image_grid_thw), '' # 暂时写死，反正所有输入图像的大小都一样，切片方式也肯定是一样的，写死了方便后面compress
        original_current_image_pixel_values, original_current_image_vision_grid_thws, original_memory_images_pixel_values, original_memory_images_vision_grid_thws = inputs['original_current_image_pixel_values'], inputs['original_current_image_vision_grid_thws'], inputs['original_memory_images_pixel_values'], inputs['original_memory_images_vision_grid_thws']
        # import pdb;pdb.set_trace()
        special_token_id_NAVTOKEN_customed_by_xwt = self.tokenizer.encode(self.SPECIAL_NAVTOKEN_customed_by_xwt)[0]
        special_token_id_MISSIONSTARTTOKEN_customed_by_xwt = self.tokenizer.encode(self.SPECIAL_TOKEN_MISSIONSTARTTOKEN_customed_by_xwt)[0]
        MISSIONSTARTTOKEN_index = input_ids == special_token_id_MISSIONSTARTTOKEN_customed_by_xwt # b,n
        NAVTOKEN_index = input_ids == special_token_id_NAVTOKEN_customed_by_xwt
        # MISSIONSTARTTOKEN_index = MISSIONSTARTTOKEN_index.int().argmax(1) # b，代表每个sample中mission内容开始的下标，直接取这个下标之后的内容就是mission text的embedding
        # 1. 得到每个样本的True索引
        idxs_mission_start_token = torch.argmax(MISSIONSTARTTOKEN_index.int(), dim=1)   # shape (4,)
        idxs_nav_token = torch.argmax(NAVTOKEN_index.int(), dim=1)   # shape (4,)
        # 2. 扩展 arange 到 batch 维度
        length = input_ids.size(1)
        arange_mission_start_token = torch.arange(length, device=input_ids.device).expand(input_ids.size(0), length)
        arange_nav_token = torch.arange(length, device=input_ids.device).expand(input_ids.size(0), length)
        # 3. 构造 mask
        mask_mission_start_token = arange_mission_start_token >= idxs_mission_start_token.unsqueeze(1)    # shape (4, 5), True/False
        mask_nav_token = arange_nav_token <= idxs_nav_token.unsqueeze(1)    # shape (4, 5), True/False
        # 4. 更新 tensor
        MISSIONTEXT_filter_out_mask = mask_nav_token & mask_mission_start_token # 为


        inputs_embeds = None
        base_model = self.get_base_model(model)
        if hasattr(base_model.model, 'embed_tokens'):
            inputs_embeds = base_model.model.embed_tokens(input_ids)
        else:
            inputs_embeds = base_model.model.language_model.embed_tokens(input_ids)

        # 只取mission_text对应的embedding
        mission_text_embed_list = [row[mask_i] for row, mask_i in zip(inputs_embeds, MISSIONTEXT_filter_out_mask)]

        if pixel_values is None and pixel_values_videos is None:  # plain-text
            if is_deepspeed_enabled():
                from PIL import Image
                images = [Image.new('RGB', (32, 32), (0, 0, 0))]
                media_inputs = self.processor.image_processor(images=images, videos=None, return_tensors='pt')
                device = input_ids.device
                media_inputs = to_device(media_inputs, device)
                pixel_values = media_inputs['pixel_values'].type(dtype)
                image_embeds = model.visual(pixel_values, grid_thw=media_inputs['image_grid_thw'])
                inputs_embeds += image_embeds.mean() * 0.
        else:
            pixel_values = None
            del pixel_values
            if not all(torch.equal(x, image_grid_thw[0]) for x in image_grid_thw):
                # print('\n\n\n\n****************')
                # print('***** WARNING !*******')
                # print('The elements in image _grid_thw are not all the same!')
                # print('****************\n\n\n\n')
                pass
            num_image_tokens_per_sample = (input_ids == image_token_id).sum(dim=1).tolist()  # [737, 737, 437, 467] 从这里面能看出每个sample的image embedding个数，然后如果单个sample只有1个<image> special token，那就知道经modle.visual过后会有多少个embedding出现在输入序列中。然后，比较两条sample，一个sample有1个<image> token另一个有2个<image> token，所以多出来的30就是resize过后的图片对应的embeddin数

            batch_size = input_ids.shape[0]

            dtype = model.visual.get_dtype() if hasattr(model.visual, 'get_dtype') else model.visual.dtype
            # pixel_values = pixel_values.to(device, dtype=dtype)
            image_grid_thw = image_grid_thw.to(device)
            total_image_count_across_all_samples_in_a_batch = len(image_grid_thw)








            # 统计每个sample中有多少张图片
            unique_rows, counts = torch.unique(image_grid_thw, return_counts=True, dim=0)
            mask = counts == batch_size
            image_grid_thw_belong_to_current_image = unique_rows[mask][0]
            image_token_count_of_each_sample = []
                
            # Step 3: 找出关键图像在原序列中的所有位置
            is_key = (image_grid_thw == image_grid_thw_belong_to_current_image).all(dim=1)  # [N], bool
            key_indices = torch.nonzero(is_key, as_tuple=False).squeeze(1)  # [batch_size], GPU

            # Step 4: 构造分段边界
            boundaries = torch.cat([torch.tensor([0], dtype=torch.long, device=device), key_indices + 1, torch.tensor([image_grid_thw.size(0)], dtype=torch.long, device=device)])  # [batch_size + 2]

            # Step 5: 相邻边界之差 = 每段长度
            image_count_of_per_sample = boundaries[1:] - boundaries[:-1]  # [batch_size + 1]
            image_count_of_per_sample = image_count_of_per_sample[image_count_of_per_sample!=0]
            memory_image_count_of_per_sample = image_count_of_per_sample - 1
            memory_embedding_count_after_compreesion_of_per_sample = image_embedding_count_per_resized_image_after_modelvisual * memory_image_count_of_per_sample # batch中的每个sample的memory有多少个图片
            # import pdb;pdb.set_trace()







            # 编码memory
            # memory_images_pixel_values, original_memory_images_pixel_values = inputs['memory_images_pixel_values'], inputs['original_memory_images_pixel_values']
            original_memory_images_pixel_values = inputs['original_memory_images_pixel_values']
            # memory_images_vision_grid_thws, original_memory_images_vision_grid_thws = inputs['memory_images_vision_grid_thws'], inputs['original_memory_images_vision_grid_thws']
            # memory_images_vision_grid_thws, original_memory_images_vision_grid_thws = memory_images_vision_grid_thws.to(dtype=image_grid_thw.dtype, device=image_grid_thw.device), original_memory_images_vision_grid_thws.to(dtype=image_grid_thw.dtype, device=image_grid_thw.device)
            original_memory_images_vision_grid_thws = inputs['original_memory_images_vision_grid_thws']
            original_memory_images_vision_grid_thws = original_memory_images_vision_grid_thws.to(dtype=image_grid_thw.dtype, device=image_grid_thw.device)

            # start_2 = time.time()
            original_memory_images_embedding = model.visual(original_memory_images_pixel_values, grid_thw=original_memory_images_vision_grid_thws) # (600,3584) (8740, 3584)





            original_current_image_pixel_values = inputs['original_current_image_pixel_values'] # 不resize的情况下，每个image的pixel_values有1748个patch，(1748, 1176)
            original_current_image_vision_grid_thws = inputs['original_current_image_vision_grid_thws']
            original_current_image_vision_grid_thws = original_current_image_vision_grid_thws.to(dtype=image_grid_thw.dtype, device=image_grid_thw.device)
            # start_4 = time.time()
            original_current_image_embedding = model.visual(original_current_image_pixel_values, grid_thw=original_current_image_vision_grid_thws) # torch.Size([1748, 3584])，1748/4=437，相当于不resize rgb的情况下，每个image有437个embedding
            # start_5 = time.time()
            num_tokens_per_image = [image_embedding_count_per_image_after_modelvisual for _ in range(batch_size)]
            # num_tokens_per_image = [437 for _ in range(4)]
            original_current_image_embedding_list_batch = torch.split(original_current_image_embedding, num_tokens_per_image, dim=0) # [1748, 3584] -> [4, 437, 3584]
            original_current_image_embedding_list_batch = [item for item in original_current_image_embedding_list_batch]




            # start_3 = time.time()
            # 压缩memory
            merge_size = getattr(self.processor.image_processor, 'merge_size', 2)
            compressed_original_memory_images_embedding_list_batch = model.model.compress_historical_image_features(
                original_memory_images_embedding,  # [N, D]
                image_grid_thw = original_memory_images_vision_grid_thws,
                merge_size = merge_size,
                mission_text_embed_list = mission_text_embed_list,
                memory_image_count_of_per_sample = memory_image_count_of_per_sample,
                original_current_image_embedding_list_batch = original_current_image_embedding_list_batch
            )  # [bxn, d],   torch.Size([8740, 3584])->torch.Size([600, 3584])
            compressed_original_memory_images_embedding_list_batch = torch.split(compressed_original_memory_images_embedding_list_batch, memory_embedding_count_after_compreesion_of_per_sample.tolist(), dim=0)
            # 恢复batch结构
            compressed_original_memory_images_embedding_list_batch = [item for item in compressed_original_memory_images_embedding_list_batch]
            # self._preprocess( image, do_resize=do_resize, size=size, resample=resample, do_rescale=do_rescale, rescale_factor=rescale_factor, do_normalize=do_normalize, image_mean=image_mean, image_std=image_std, patch_size=patch_size, temporal_patch_size=temporal_patch_size, merge_size=merge_size, data_format=data_format, do_convert_rgb=do_convert_rgb, input_data_format=input_data_format, )




            # 拼接压缩后的memory和current image，按batch_index拼接，相当于还原每个sample的image顺序
            final_image_embedding_list = []

            augmented_mission_text_embed_list = []
            # import pdb;pdb.set_trace()
            for batch_index in range(batch_size):
                image_count_of_current_sample = image_count_of_per_sample[batch_index]
                current_sample_image_embedding = torch.cat([compressed_original_memory_images_embedding_list_batch[batch_index], original_current_image_embedding_list_batch[batch_index]], dim = 0) # [n, d]

                final_image_embedding_list.append(
                    current_sample_image_embedding
                ) # 每个元素都是memory+current image的顺序，都是[n,d]维度
            final_image_embedding = torch.cat(final_image_embedding_list, dim=0) # 需要对比这个和直接pixel_values经过model.visual得到的embedding的维度区别，应该要相同

            # import pdb;pdb.set_trace()
            
            image_mask = (input_ids == model.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
            final_image_embedding = final_image_embedding.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, final_image_embedding) # inputs_embeds在scatter前后的维度不变
        # time_4 = time.time()

        del final_image_embedding, original_current_image_embedding_list_batch, compressed_original_memory_images_embedding_list_batch, original_current_image_embedding, original_current_image_pixel_values, original_memory_images_embedding, original_memory_images_pixel_values

        # print("original_memory_images_embedding = model.visual: {}s".format(start_3 - start_2))
        # print("original_current_image_embedding = model.visual: {}s".format(start_5 - start_4))
        # print("encode function {}s".format(time_4 - start_1))

        return {
            'inputs_embeds': inputs_embeds,
        }

    def _data_collator_mm_data(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        res = super()._data_collator_mm_data(batch)
        second_per_grid_ts = self.gather_list(batch, 'second_per_grid_ts')
        if second_per_grid_ts:
            res['second_per_grid_ts'] = second_per_grid_ts
        for media_type in ['image', 'video']:
            grid_thw = self.concat_tensor(batch, f'{media_type}_grid_thw', 0)
            if grid_thw is not None:
                res[f'{media_type}_grid_thw'] = grid_thw
        # print('_data_collator_mm in swift/llm/template/template/qwen.py')
        # print([item.keys() for item in batch])
        # print()
        # import pdb;pdb.set_trace()
        t_keys_list = [list(item.keys()) for item in batch]
        keys_list = []
        for item in t_keys_list:
            keys_list += item
        keys_list = set(keys_list)
        keys_want = ["memory_images_vision_grid_thws", "memory_images_pixel_values", "current_image_pixel_values", "current_image_vision_grid_thws",
            "original_memory_images_vision_grid_thws", "original_memory_images_pixel_values", "original_current_image_pixel_values", "original_current_image_vision_grid_thws",
        ]
        for k in keys_list:
            if k in keys_want:
                memory_images_vision_grid_thws = self.concat_tensor(batch, k, 0)
                if memory_images_vision_grid_thws is not None:
                    res[k] = memory_images_vision_grid_thws
        # print(res.keys())
        return res

    def packing_row(self, row: List[Tuple[Dict[str, Any], int]]) -> Dict[str, Any]:
        position_ids = []
        for r in row:
            r = r[0].copy()
            r['input_ids'] = torch.tensor(r['input_ids'])[None]
            position_ids.append(self._get_position_ids(r))
        packed = super().packing_row(row)
        packed['real_position_ids'] = torch.concat(position_ids, dim=-1)
        return packed

    def _get_position_ids(self, inputs: Dict[str, Any]):
        # fix https://github.com/huggingface/transformers/pull/33487
        kwargs = {}
        if self.version == 'v2_5':
            kwargs = {'second_per_grid_ts': inputs.get('second_per_grid_ts')}
        base_model = self.get_base_model(self.model)
        if hasattr(base_model, 'get_rope_index'):
            get_rope_index = base_model.get_rope_index
        else:
            get_rope_index = base_model.model.get_rope_index
        position_ids, _ = get_rope_index(
            inputs['input_ids'],
            inputs.get('image_grid_thw'),
            inputs.get('video_grid_thw'),
            attention_mask=inputs.get('attention_mask'),
            **kwargs)
        return position_ids.contiguous()

    def _data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        # print('_data_collator in swift/llm/template/template/qwen.py')
        # print([item.keys() for item in batch])
        # print()
        res = super()._data_collator(batch, padding_to=padding_to)
        if self._packing:
            res['real_position_ids'] = self.concat_tensor(batch, 'real_position_ids', -1)
        elif self.is_training:
            res['position_ids'] = self._get_position_ids(res)
        return res


register_template(QwenTemplateMeta(MLLMTemplateType.qwen2_vl, template_cls=Qwen2VLTemplate))

register_template(
    QwenTemplateMeta(
        MLLMTemplateType.qvq,
        default_system=('You are a helpful and harmless assistant. You are Qwen developed by Alibaba. '
                        'Answer in the language of the question. You should think step-by-step.'),
        template_cls=Qwen2VLTemplate,
    ))


class Qwen2_5VLTemplate(Qwen2VLTemplate):
    version = 'v2_5'
    norm_bbox = 'none'


register_template(QwenTemplateMeta(MLLMTemplateType.qwen2_5_vl, template_cls=Qwen2_5VLTemplate))

register_template(
    QwenTemplateMeta(
        MLLMTemplateType.mimo_vl,
        template_cls=Qwen2_5VLTemplate,
        default_system='You are MiMo, an AI assistant developed by Xiaomi.'))


class Qwen2_5OmniTemplate(Qwen2_5VLTemplate):
    version = 'omni'
    placeholder_tokens = ['<|IMAGE|>', '<|AUDIO|>', '<|VIDEO|>']

    def init_processor(self, processor) -> None:
        if processor is None:
            return
        super().init_processor(processor)
        from transformers.models.qwen2_5_omni.processing_qwen2_5_omni import Qwen2_5OmniProcessorKwargs
        default = Qwen2_5OmniProcessorKwargs._defaults
        self.seconds_per_chunk = default['videos_kwargs']['seconds_per_chunk']
        self.position_id_per_seconds = default['videos_kwargs']['position_id_per_seconds']
        self.use_audio_in_video = get_env_args('use_audio_in_video', bool, False)
        self.sampling_rate = get_env_args('sampling_rate', int, self.processor.feature_extractor.sampling_rate)

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        from qwen_omni_utils import fetch_image, fetch_video
        if media_type == 'image':
            inputs.images[index] = fetch_image({'image': inputs.images[index]})
            return ['<|vision_bos|><|IMAGE|><|vision_eos|>']
        elif media_type == 'audio':
            if self.mode != 'vllm':
                inputs.audios[index] = load_audio(inputs.audios[index], self.sampling_rate)
            return ['<|audio_bos|><|AUDIO|><|audio_eos|>']
        elif media_type == 'video':
            video = inputs.videos[index]
            inputs.videos[index] = fetch_video({'video': video}).to(torch.uint8)
            if self.use_audio_in_video:
                import librosa
                if video.startswith('http://') or video.startswith('https://'):
                    import audioread
                    video = audioread.ffdec.FFmpegAudioFile(video)
                video = librosa.load(video, sr=self.sampling_rate)[0]
                inputs.audios.insert(inputs.audio_idx, (video, 'video'))
                inputs.audio_idx += 1
                return ['<|vision_bos|><|audio_bos|><|VIDEO|><|audio_eos|><|vision_eos|>']
            return ['<|vision_bos|><|VIDEO|><|vision_eos|>']

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        # import pdb;pdb.set_trace()
        encoded = Template._encode(self, inputs)
        processor = self.processor
        video_audios_mask = []
        for i, audio in enumerate(inputs.audios):
            if isinstance(audio, tuple) and audio[1] == 'video':
                inputs.audios[i] = audio[0]
                video_audios_mask.append(True)
            else:
                video_audios_mask.append(False)
        video_audios_mask = torch.tensor(video_audios_mask)
        media_inputs = processor(
            text='',
            audio=inputs.audios or None,
            images=inputs.images or None,
            videos=inputs.videos or None,
            do_resize=False,
            return_tensors='pt')
        media_inputs.pop('input_ids')
        media_inputs.pop('attention_mask')
        media_inputs = to_float_dtype(media_inputs, self.model_info.torch_dtype)
        input_ids = encoded['input_ids']
        labels = encoded['labels']
        # audio
        audio_token_id = self._tokenize('<|AUDIO|>')
        idx_list = findall(input_ids, audio_token_id)
        feature_attention_mask = media_inputs.get('feature_attention_mask')
        if feature_attention_mask is not None:
            audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
            audio_lengths = (((audio_feature_lengths - 1) // 2 + 1 - 2) // 2 + 1)
        else:
            audio_lengths = None
        audio_lengths_origin = audio_lengths
        if idx_list:
            if self.use_audio_in_video:
                audio_lengths = audio_lengths[~video_audios_mask]

            def _get_new_audio_tokens(i):
                return audio_token_id * audio_lengths[i]

            input_ids, labels = self._extend_tokens(input_ids, labels, idx_list, _get_new_audio_tokens)

        for media_type in ['image', 'video']:
            token = f'<|{media_type.upper()}|>'
            token_id = self._tokenize(token)
            idx_list = findall(input_ids, token_id)
            if idx_list:
                merge_size = processor.image_processor.merge_size
                media_grid_thw = media_inputs.get(f'{media_type}_grid_thw')
                if media_type == 'video' and self.use_audio_in_video:
                    audio_lengths = audio_lengths_origin[video_audios_mask]
                    video_second_per_grid = media_inputs['video_second_per_grid']

                    def _get_new_tokens_use_audio_in_video(i):
                        audio_token_indices = torch.arange(audio_lengths[i])
                        grid_thw = media_grid_thw[i]
                        height = grid_thw[1] // merge_size
                        width = grid_thw[2] // merge_size
                        video_token_indices = torch.arange(grid_thw[0]).reshape(-1, 1, 1)
                        video_token_indices = torch.broadcast_to(
                            video_token_indices, (video_token_indices.shape[0], height, width)).reshape(-1)
                        video_token_indices = (
                            video_token_indices * video_second_per_grid[i] * self.position_id_per_seconds)
                        tokens_per_chunk = int(self.position_id_per_seconds * self.seconds_per_chunk)
                        video_chunk_indexes = processor.get_chunked_index(video_token_indices, tokens_per_chunk)
                        audio_chunk_indexes = processor.get_chunked_index(audio_token_indices, tokens_per_chunk)

                        res = []
                        for j in range(max(len(video_chunk_indexes), len(audio_chunk_indexes))):
                            if j < len(video_chunk_indexes):
                                video_seq_length = video_chunk_indexes[j][1] - video_chunk_indexes[j][0]
                                res += token_id * video_seq_length
                            if j < len(audio_chunk_indexes):
                                audio_seq_length = audio_chunk_indexes[j][1] - audio_chunk_indexes[j][0]
                                res += audio_token_id * audio_seq_length
                        return res

                    input_ids, labels = self._extend_tokens(input_ids, labels, idx_list,
                                                            _get_new_tokens_use_audio_in_video)

                else:

                    def _get_new_tokens(i):
                        token_len = (media_grid_thw[i].prod() // (merge_size**2))
                        return token_id * token_len

                    input_ids, labels = self._extend_tokens(input_ids, labels, idx_list, _get_new_tokens)

        encoded['input_ids'] = input_ids
        encoded['labels'] = labels
        encoded.update(media_inputs)
        return encoded

    def _post_encode(self, model, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return Template._post_encode(self, model, inputs)

    def _get_position_ids(self, inputs: Dict[str, Any]):
        feature_attention_mask = inputs.get('feature_attention_mask')
        if feature_attention_mask is not None:
            audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
        else:
            audio_feature_lengths = None
        video_second_per_grid = inputs.pop('video_second_per_grid', None)
        input_ids = inputs['input_ids']
        attention_mask = inputs.get('attention_mask')
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        position_ids, _ = self.model.thinker.get_rope_index(
            input_ids,
            inputs.get('image_grid_thw'),
            inputs.get('video_grid_thw'),
            attention_mask,
            self.use_audio_in_video,
            audio_feature_lengths,
            video_second_per_grid,
        )
        return position_ids.contiguous()

    def _data_collator_mm_data(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        res = super()._data_collator_mm_data(batch)
        video_second_per_grid = self.gather_list(batch, 'video_second_per_grid')
        if video_second_per_grid:
            res['video_second_per_grid'] = video_second_per_grid
        input_features = [b['input_features'] for b in batch if b.get('input_features') is not None]
        feature_attention_mask = [
            b['feature_attention_mask'] for b in batch if b.get('feature_attention_mask') is not None
        ]
        if input_features:
            res['input_features'] = torch.concat(input_features)
            res['feature_attention_mask'] = torch.concat(feature_attention_mask)
        return res

    def generate(self, model, *args, **kwargs):
        if kwargs.get('video_grid_thw') is not None:
            kwargs['use_audio_in_video'] = self.use_audio_in_video
        return super().generate(model, *args, **kwargs)


register_template(QwenTemplateMeta(MLLMTemplateType.qwen2_5_omni, template_cls=Qwen2_5OmniTemplate))


class Ovis1_6Template(Template):
    skip_prompt = False
    use_model = True

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        assert media_type == 'image'
        return [[-200], '\n']

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        images = inputs.images
        input_ids = encoded['input_ids']
        labels = encoded['labels']
        idx_list = findall(input_ids, [-200])
        added_tokens_len = 0
        pixel_values = []
        for i, idx in enumerate(idx_list):
            max_partition = get_env_args('max_partition', int, 9)
            raw_pixel_values, image_placeholders = self.model.visual_tokenizer.preprocess_image(
                images[i], max_partition=max_partition)
            input_ids = input_ids[:idx] + image_placeholders + input_ids[idx + 1:]
            if labels is not None:
                labels = labels[:idx] + [-100] * len(image_placeholders) + labels[idx + 1:]
            pixel_values.append(raw_pixel_values)
            added_tokens_len += len(image_placeholders) - 1
        dtype = self.model.visual_tokenizer.dtype
        if pixel_values:
            pixel_values = torch.cat(pixel_values, dim=0).to(dtype)
        else:
            pixel_values = torch.zeros((1, 3, 384, 384), dtype=dtype)  # dummpy
        encoded.update({'input_ids': input_ids, 'labels': labels})
        encoded['pixel_values'] = [pixel_values]
        return encoded

    def _post_encode(self, model, inputs: Dict[str, Any]) -> Dict[str, Any]:
        padding_side = self.padding_side if self.is_training else 'left'
        if self.max_length is not None:
            model.config.multimodal_max_length = self.max_length
        input_ids = inputs['input_ids']
        labels = inputs.get('labels')
        if labels is None:
            labels = input_ids.new_full(input_ids.shape, -100)
        _, inputs_embeds, labels, attention_mask = model.merge_multimodal(
            text_input_ids=input_ids,
            text_attention_masks=torch.ones_like(input_ids),  # not use, only compat
            text_labels=labels,
            pixel_values=inputs['pixel_values'],
            left_padding=padding_side == 'left')
        if inputs.get('labels') is None:
            labels = None
        return {'inputs_embeds': inputs_embeds, 'labels': labels, 'attention_mask': attention_mask}

    def _data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        pixel_values = self.gather_list(batch, 'pixel_values')
        res = super()._data_collator(batch, padding_to=padding_to)
        res['pixel_values'] = pixel_values
        return res


register_template(
    TemplateMeta(
        MLLMTemplateType.ovis1_6,
        prefix=['<bos>'],
        prompt=['<start_of_turn>user\n{{QUERY}}<end_of_turn>\n<start_of_turn>model\n'],
        chat_sep=['<end_of_turn>\n'],
        suffix=['<end_of_turn>'],
        system_prefix=['<bos><start_of_turn>system\n{{SYSTEM}}<end_of_turn>\n'],
        template_cls=Ovis1_6Template,
    ))

register_template(
    Llama3TemplateMeta(
        MLLMTemplateType.ovis1_6_llama3,
        default_system='You are a helpful and honest multimodal assistant.',
        template_cls=Ovis1_6Template,
    ))


class Ovis2Template(Ovis1_6Template):
    placeholder_tokens = ['<|image_pad|>', '<|video_pad|>']
    nframes = 12

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        if media_type == 'image':
            return [[-200], '\n']
        elif media_type == 'video':
            nframes = get_env_args('nframes', int, self.nframes)
            inputs.images = load_video_ovis2(inputs.videos[index], nframes)
            return [[-200] * nframes, '\n']


register_template(QwenTemplateMeta(
    MLLMTemplateType.ovis2,
    template_cls=Ovis2Template,
))


@dataclass
class MarcoO1TemplateMeta(QwenTemplateMeta):
    default_system: Optional[str] = """
你是一个经过良好训练的AI助手，你的名字是Marco-o1.由阿里国际数字商业集团的AI Business创造.
        \n## 重要！！！！！
当你回答问题时，你的思考应该在<Thought>内完成，<Output>内输出你的结果。
<Thought>应该尽可能是英文，但是有2个特例，一个是对原文中的引用，另一个是是数学应该使用markdown格式，<Output>内的输出需要遵循用户输入的语言。
        """


register_template(MarcoO1TemplateMeta(LLMTemplateType.marco_o1))
