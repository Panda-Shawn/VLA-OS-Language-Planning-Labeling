from __future__ import annotations

import time
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from transformers import LlamaTokenizerFast

from experiments.bridge.reasoning_client import ReasoningClient
from prismatic.models.vlms.prismatic import PrismaticVLM
from prismatic.overwatch import initialize_overwatch
from prismatic.util.cot_utils import CotTag
from prismatic.vla.action_tokenizer import ActionTokenizer

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


class OpenVLA(PrismaticVLM):
    def __init__(
        self,
        *args,
        norm_stats: Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]],
        action_tokenizer: ActionTokenizer,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.norm_stats = norm_stats
        self.action_tokenizer = action_tokenizer

        self.use_cot = True
        self.use_async = False
        self.use_interactive = False

        self.base_prompt = f"{CotTag.TASK.value}"

        self.max_freezing_time = 5
        self.time_frozen = 0
        self.frozen_prompt = self.base_prompt

        if self.use_async:
            self.reasoning_client = ReasoningClient()
            self.last_reasoning = None
            self.pending = False
            self.prev_time = time.time()
            self.local_trials = 0

    def raw_generate(self, input_ids, pixel_values):
        if isinstance(pixel_values, torch.Tensor):
            pixel_values = pixel_values.to(self.device).bfloat16()
        elif isinstance(pixel_values, dict):
            pixel_values = {k: v.to(self.device).bfloat16() for k, v in pixel_values.items()}

        return super(PrismaticVLM, self).generate(
            input_ids=input_ids.to(self.device),
            pixel_values=pixel_values,
            max_new_tokens=1024,
        )

    def reset_async(self):
        self.time_frozen = 0
        if self.use_async:
            self.reasoning_client = ReasoningClient()
            self.last_reasoning = None
            self.pending = False
            self.local_trials = 0

    def enable_cot(self, enable: bool):
        self.use_cot = enable

    @torch.inference_mode()
    def predict_action(
        self,
        image: Image,
        instruction: str,
        unnorm_key: Optional[str] = None,
        info_dict: Optional[dict] = None,
        **kwargs: str,
    ) -> np.ndarray:
        """
        Core function for VLA inference; maps input image and task instruction to continuous action and reasoning.

        @param image: PIL Image as [height, width, 3]
        @param instruction: Task instruction string
        @param unnorm_key: Optional dataset name for retrieving un-normalizing statistics; if None, checks that model
                           was trained only on a single dataset, and retrieves those statistics.

        @return Unnormalized (continuous) action vector --> end-effector deltas.
        """
        image_transform, tokenizer = self.vision_backbone.image_transform, self.llm_backbone.tokenizer

        # Build VLA prompt
        prompt_builder = self.get_prompt_builder()
        prompt_builder.add_turn(role="human", message=f"What action should the robot take to {instruction.lower()}?")
        prompt_text = prompt_builder.get_prompt()

        # Prepare Inputs
        init_input_ids = tokenizer(prompt_text, truncation=True, return_tensors="pt").input_ids.to(self.device)

        def build_prompt(prompt_prefix, input_ids):
            if isinstance(tokenizer, LlamaTokenizerFast):
                # Note: We start the answer with "TASK:" to force generating the reasoning part.
                return torch.cat(
                    (
                        input_ids,
                        tokenizer(prompt_prefix, return_tensors="pt").input_ids.to(self.device)[:, 1:],
                    ),
                    dim=1,
                )
            else:
                raise ValueError(f"Unsupported `tokenizer` type = {type(tokenizer)}")

        # Preprocess Image
        pixel_values = image_transform(image)
        if isinstance(pixel_values, torch.Tensor):
            pixel_values = pixel_values[None, ...].to(self.device)
        elif isinstance(pixel_values, dict):
            pixel_values = {k: v[None, ...].to(self.device) for k, v in pixel_values.items()}
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

        # Invoke super().generate --> taps into `GenerationMixin` which (redirects) to `forward()`
        autocast_dtype = self.llm_backbone.half_precision_dtype
        with torch.autocast("cuda", dtype=autocast_dtype, enabled=self.enable_mixed_precision_training):
            # fmt: off
            if self.use_async:
                if self.local_trials >= 4:
                    self.local_trials = 0
                    self.reasoning_client.join()

                if self.pending and self.reasoning_client.done():
                    self.pending = False
                    self.local_trials = 0

                    generated_ids = self.reasoning_client.get_result()
                    decoded_tokens = self.llm_backbone.tokenizer.decode(generated_ids[0, :-1])

                    self.last_reasoning = decoded_tokens.split("ASSISTANT:")[-1].strip()  # for siglip
                    self.last_reasoning = decoded_tokens.split("\nOut: ")[-1].strip()  # for prism-dinosiglip

                    self.last_reasoning = \
                        self.last_reasoning.split(CotTag.MOVE_REASONING.value)[0] + CotTag.MOVE_REASONING.value + " "

                if not self.pending:
                    self.reasoning_client.request((build_prompt(self.base_prompt, init_input_ids), pixel_values))
                    self.pending = True

                if self.last_reasoning is not None:
                    generated_ids = self.raw_generate(build_prompt(self.last_reasoning, init_input_ids), pixel_values)
                    self.local_trials += 1
                else:
                    # first step, wait for the full reasoning
                    self.reasoning_client.join()
                    generated_ids = self.reasoning_client.get_result()
            else:
                if self.use_interactive:
                    import pyautogui
                    prompt = self.base_prompt

                    while True:
                        generated_ids = self.raw_generate(build_prompt(prompt, init_input_ids), pixel_values)
                        decoded_tokens = self.llm_backbone.tokenizer.decode(generated_ids[0, :-1])

                        prompt = decoded_tokens.split("\nOut: ")[-1]
                        prompt = prompt.split(" ASSISTANT: ")[-1]
                        prompt = prompt.split(" ACTION: ")[0].replace('\n', ' ')

                        print(
                            f"\nProposed reasoning: {prompt}\n\n[Leave unchanged to accept, or modify to regenerate]\n"
                            "[type \"[continue]\" to accept and exit the interactive mode]\n"
                            "[Use [Ctrl]+[left/right] to navigate the text, [Ctrl]+w to delete a word, "
                            "[Ctrl]+u to delete a line]\n")
                        pyautogui.write(prompt)
                        user_response = input()

                        if user_response == "[continue]":
                            print('\033[92m\033[1mReasoning accepted!\033[0m')
                            self.use_interactive = False
                            break
                        elif user_response == prompt:
                            print('\033[92m\033[1mReasoning accepted!\033[0m')
                            break
                        else:
                            prompt = user_response
                else:
                    # run generate and wait for the result
                    print(f"Prompt freezing: {self.time_frozen} turns left.")
                    if self.time_frozen <= 0:
                        self.frozen_prompt = self.base_prompt
                        self.time_frozen = self.max_freezing_time

                    self.time_frozen -= 1
                    generated_ids = self.raw_generate(build_prompt(self.frozen_prompt, init_input_ids), pixel_values)

                    decoded_tokens = self.llm_backbone.tokenizer.decode(generated_ids[0, :-1])
                    prompt = decoded_tokens.split("\nOut: ")[-1]
                    prompt = prompt.split(" ASSISTANT: ")[-1]

                    if " MOVE REASONING: " in prompt:
                        prompt = prompt.split(" MOVE REASONING: ")[0]
                    else:
                        prompt = prompt.split(" GRIPPER POSITION: ")[0]
                    self.frozen_prompt = prompt

            generated_ids = generated_ids[:, :-1]  # remove the EOS token
            decoded_tokens = self.llm_backbone.tokenizer.decode(generated_ids[0])

            print("Reasoning:", decoded_tokens)
            if info_dict is not None:
                info_dict["decoded_tokens"] = decoded_tokens
            # fmt: on

        # Extract predicted action tokens and translate into (normalized) continuous actions
        predicted_action_token_ids = generated_ids[0, -self.get_action_dim(unnorm_key) :]
        normalized_actions = self.action_tokenizer.decode_token_ids_to_actions(predicted_action_token_ids.cpu().numpy())

        # Un-normalize Actions
        action_norm_stats = self.get_action_stats(unnorm_key)
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )

        return actions

    @staticmethod
    def _check_unnorm_key(norm_stats: Dict, unnorm_key: str) -> str:
        if unnorm_key is None:
            assert len(norm_stats) == 1, (
                f"Your model was trained on more than one dataset, please pass a `unnorm_key` from the following "
                f"options to choose the statistics used for un-normalizing actions: {norm_stats.keys()}"
            )
            unnorm_key = next(iter(norm_stats.keys()))

        # Error Handling
        assert (
            unnorm_key in norm_stats
        ), f"The `unnorm_key` you chose is not in the set of available statistics; choose from: {norm_stats.keys()}"

        return unnorm_key

    def get_action_dim(self, unnorm_key: Optional[str] = None) -> int:
        """Dimensionality of the policy's action space."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)

        return len(self.norm_stats[unnorm_key]["action"]["q01"])

    def get_action_stats(self, unnorm_key: Optional[str] = None) -> Dict:
        """Dimensionality of the policy's action space."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)

        return self.norm_stats[unnorm_key]["action"]
