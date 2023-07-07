import os
import torch
import argparse
import numpy as np
import PIL
from PIL import Image
from tqdm import tqdm
from einops import rearrange
from diffusers_img2img.src.diffusers.schedulers import DDIMScheduler, DDIMInverseScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from transformers import AutoTokenizer, PretrainedConfig
from diffusers_img2img.src.diffusers import AutoencoderKL, UNet2DConditionModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_model_path",
        type=str,
        default=None,
        required=True,
        help="Path to the base model"
    )
    parser.add_argument(
        "--target_model_path",
        type=str,
        default=None,
        required=True,
        help="Path to target model"
    )
    parser.add_argument(
        "--inverted_image_path",
        type=str,
        default=None,
        required=True,
        help="Path to the inverted image latent"
    )
    parser.add_argument(
        "--style_prompt",
        type=str,
        default=None,
        required=True,
        help="Rare-token identifier"
    )
    parser.add_argument(
        "--alpha_blend",
        type=float,
        default=None,
        help="Blending strength"
    )
    parser.add_argument(
        "--layer_swap",
        type=int,
        default=None,
        help="How many layers to swap"
    )
    parser.add_argument(
        "--self_attention_layers",
        action="store_true",
        help="Extract self attention layers"
    )
    parser.add_argument(
        "--resnet_block_layers",
        action="store_true",
        help="Extract resnet block layers"
    )
    parser.add_argument(
        "--save_intermediate_images_path",
        type=str,
        default=None,
        help="Path to save the intermediate images from the DDIM Inversion"
    )

    args = parser.parse_args()

    return args


class I2IPipeline:
    def __init__(self, args):
        self.base_model_path = args.base_model_path
        self.target_model_path = args.target_model_path
        self.save_intermediate_images_path = args.save_intermediate_images_path

        self.prompt = ""
        self.style_prompt = args.style_prompt

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.alpha_blend = args.alpha_blend
        self.layer_swap = args.layer_swap

        self.self_attention_layers = args.self_attention_layers
        self.resnet_block_layers = args.resnet_block_layers

        self.base_latent = torch.load(args.inverted_image_path)
        self.target_latent = torch.load(args.inverted_image_path)

        # load base model
        self.tokenizer_base = AutoTokenizer.from_pretrained(self.base_model_path, subfolder="tokenizer", use_fast=False)
        self.noise_scheduler_base = DDIMScheduler.from_pretrained(self.base_model_path, subfolder="scheduler")
        self.text_encoder_base = CLIPTextModel.from_pretrained(self.base_model_path, subfolder="text_encoder")
        self.vae_base = AutoencoderKL.from_pretrained(self.base_model_path, subfolder="vae")
        self.unet_base = UNet2DConditionModel.from_pretrained(self.base_model_path, subfolder="unet")

        self.vae_base.to(self.device)
        self.text_encoder_base.to(self.device)
        self.unet_base.to(self.device)

        self.vae_base.eval()
        self.text_encoder_base.eval()
        self.unet_base.eval()


        # load target model
        self.tokenizer_target = AutoTokenizer.from_pretrained(self.target_model_path, subfolder="tokenizer", use_fast=False)
        self.noise_scheduler_target = DDIMScheduler.from_pretrained(self.target_model_path, subfolder="scheduler")
        self.text_encoder_target = CLIPTextModel.from_pretrained(self.target_model_path, subfolder="text_encoder")
        self.vae_target = AutoencoderKL.from_pretrained(self.target_model_path, subfolder="vae")
        self.unet_target = UNet2DConditionModel.from_pretrained(self.target_model_path, subfolder="unet")

        self.vae_target.to(self.device)
        self.text_encoder_target.to(self.device)
        self.unet_target.to(self.device)

        self.vae_target.eval()
        self.text_encoder_target.eval()
        self.unet_target.eval()

    def save_sampled_img(self, x, i, save_path=None):
        latents = 1 / self.vae_target.config.scaling_factor * x
        image = self.vae_target.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        image = (image * 255).round().astype("uint8")
        img = Image.fromarray(image[0])
        if save_path:
            img.save(os.path.join(save_path, f"{i}.png"))
        else:
            img.save(f"{i}.png")

    def extract_features(self, layer_swap, self_attention_layers, resnet_block_layers):
        num_cross_attn_down_blocks = len(self.unet_base.down_block_types[:-1]) if self_attention_layers else len(self.unet_base.down_block_types)
        num_cross_attn_up_blocks = len(self.unet_base.up_block_types[1:]) if self_attention_layers else len(self.unet_base.up_block_types)
        num_attentions_down = len(self.unet_base.down_blocks[0].attentions)
        num_attentions_up = len(self.unet_base.up_blocks[1].attentions)
        swap_range = np.arange(layer_swap) + 1 if layer_swap else None
        layer_up = 1

        if self_attention_layers:
            attn1_features = {}

            for cross_attn_down_block in range(num_cross_attn_down_blocks):
                attn1_features[f"cross_attn_down_block_{cross_attn_down_block}"] = {}
                for attention in range(num_attentions_down):
                    attn1_features[f"cross_attn_down_block_{cross_attn_down_block}"][f"attention_{attention}"] = []
                    attn1_features[f"cross_attn_down_block_{cross_attn_down_block}"][f"attention_{attention}"].append(
                        self.unet_base.down_blocks[cross_attn_down_block].attentions[attention].transformer_blocks[0].attn1.q.cpu())
                    attn1_features[f"cross_attn_down_block_{cross_attn_down_block}"][f"attention_{attention}"].append(
                        self.unet_base.down_blocks[cross_attn_down_block].attentions[attention].transformer_blocks[0].attn1.k.cpu())
                    attn1_features[f"cross_attn_down_block_{cross_attn_down_block}"][f"attention_{attention}"].append(
                        self.unet_base.down_blocks[cross_attn_down_block].attentions[attention].transformer_blocks[0].attn1.v.cpu())
                    attn1_features[f"cross_attn_down_block_{cross_attn_down_block}"][f"attention_{attention}"].append(
                        self.unet_base.down_blocks[cross_attn_down_block].attentions[attention].transformer_blocks[0].attn1.out.cpu())

            for cross_attn_up_block in range(num_cross_attn_up_blocks):
                attn1_features[f"cross_attn_up_block_{cross_attn_up_block + 1}"] = {}
                for attention in range(num_attentions_up):
                    attn1_features[f"cross_attn_up_block_{cross_attn_up_block + 1}"][f"attention_{attention}"] = []
                    attn1_features[f"cross_attn_up_block_{cross_attn_up_block + 1}"][f"attention_{attention}"].append(
                        self.unet_base.up_blocks[cross_attn_up_block + 1].attentions[attention].transformer_blocks[0].attn1.q.cpu())
                    attn1_features[f"cross_attn_up_block_{cross_attn_up_block + 1}"][f"attention_{attention}"].append(
                        self.unet_base.up_blocks[cross_attn_up_block + 1].attentions[attention].transformer_blocks[0].attn1.k.cpu())
                    attn1_features[f"cross_attn_up_block_{cross_attn_up_block + 1}"][f"attention_{attention}"].append(
                        self.unet_base.up_blocks[cross_attn_up_block + 1].attentions[attention].transformer_blocks[0].attn1.v.cpu())
                    attn1_features[f"cross_attn_up_block_{cross_attn_up_block + 1}"][f"attention_{attention}"].append(
                        self.unet_base.up_blocks[cross_attn_up_block + 1].attentions[attention].transformer_blocks[0].attn1.out.cpu())
                    if layer_swap:
                        if layer_up in swap_range:
                            attn1_features[f"cross_attn_up_block_{cross_attn_up_block + 1}"][f"attention_{attention}"].append(0)
                        else:
                            attn1_features[f"cross_attn_up_block_{cross_attn_up_block + 1}"][f"attention_{attention}"].append(1)
                        layer_up += 1

            return attn1_features

        elif resnet_block_layers:
            resnet_features = {}

            for cross_attn_down_block in range(num_cross_attn_down_blocks):
                resnet_features[f"cross_attn_down_block_{cross_attn_down_block}"] = {}
                for attention in range(num_attentions_down):
                    resnet_features[f"cross_attn_down_block_{cross_attn_down_block}"][f"resnet_{attention}"] = []
                    resnet_features[f"cross_attn_down_block_{cross_attn_down_block}"][f"resnet_{attention}"].append(
                        self.unet_base.down_blocks[cross_attn_down_block].resnets[attention].conv1_layer.cpu())
                    resnet_features[f"cross_attn_down_block_{cross_attn_down_block}"][f"resnet_{attention}"].append(
                        self.unet_base.down_blocks[cross_attn_down_block].resnets[attention].conv2_layer.cpu())

            for cross_attn_up_block in range(num_cross_attn_up_blocks):
                resnet_features[f"cross_attn_up_block_{cross_attn_up_block}"] = {}
                for attention in range(num_attentions_up):
                    resnet_features[f"cross_attn_up_block_{cross_attn_up_block}"][f"resnet_{attention}"] = []
                    resnet_features[f"cross_attn_up_block_{cross_attn_up_block}"][f"resnet_{attention}"].append(
                        self.unet_base.up_blocks[cross_attn_up_block].resnets[attention].conv1_layer.cpu())
                    resnet_features[f"cross_attn_up_block_{cross_attn_up_block}"][f"resnet_{attention}"].append(
                        self.unet_base.up_blocks[cross_attn_up_block].resnets[attention].conv2_layer.cpu())
                    if layer_swap:
                        if layer_up in swap_range:
                            resnet_features[f"cross_attn_up_block_{cross_attn_up_block}"][f"resnet_{attention}"].append(0)
                        else:
                            resnet_features[f"cross_attn_up_block_{cross_attn_up_block}"][f"resnet_{attention}"].append(1)
                        layer_up += 1

            return resnet_features


    @torch.no_grad()
    def run_i2i(self):
        if self.save_intermediate_images_path:
            os.makedirs(self.save_intermediate_images_path, exist_ok=True)

        prompt_embeds_base = self.tokenizer_base(self.prompt,
                                       truncation=True,
                                       padding="max_length",
                                       max_length=self.tokenizer_base.model_max_length,
                                       return_tensors="pt").input_ids

        prompt_embeds_target = self.tokenizer_target(self.style_prompt,
                                                 truncation=True,
                                                 padding="max_length",
                                                 max_length=self.tokenizer_target.model_max_length,
                                                 return_tensors="pt").input_ids

        encoder_hidden_states_base = self.text_encoder_base(prompt_embeds_base.to(self.device))[0]
        encoder_hidden_states_target = self.text_encoder_target(prompt_embeds_target.to(self.device))[0]

        self.noise_scheduler_base.set_timesteps(999, device=self.device)
        timesteps_base = self.noise_scheduler_base.timesteps

        self.noise_scheduler_target.set_timesteps(50, device="cuda")
        timesteps_target = self.noise_scheduler_target.timesteps

        step_ratio = 1000 // 50
        timesteps_to_save = (np.arange(0, 50) * step_ratio + 1)

        progress_bar = tqdm(enumerate(timesteps_base), desc='DDIM Inversion', total=999)

        extracted_features = []

        for i, t in progress_bar:
            scaled_latents = self.noise_scheduler_base.scale_model_input(self.base_latent, t)
            model_pred = self.unet_base(scaled_latents, t, encoder_hidden_states_base).sample
            scheduler_output = self.noise_scheduler_base.step(model_pred, t, self.base_latent)
            self.base_latent, pred_x0 = scheduler_output.prev_sample, scheduler_output.pred_original_sample

            if t.item() in timesteps_to_save:
                extracted_features.append(self.extract_features(layer_swap=self.layer_swap, self_attention_layers=self.self_attention_layers, resnet_block_layers=self.resnet_block_layers))

        progress_bar = tqdm(enumerate(timesteps_target), desc='Running I2I Translation', total=50)
        injection_threshold = np.arange(0, 30)

        for i, t in progress_bar:
            scaled_latents = self.noise_scheduler_target.scale_model_input(self.target_latent, t)
            if i in injection_threshold:
                model_pred = self.unet_target(scaled_latents, t, encoder_hidden_states_target, inject_features=extracted_features[i], self_attention_layers=self.self_attention_layers, alpha_blend=self.alpha_blend).sample
            else:
                model_pred = self.unet_target(scaled_latents, t, encoder_hidden_states_target).sample
            scheduler_output = self.noise_scheduler_target.step(model_pred, t, self.target_latent)
            self.target_latent, pred_x0 = scheduler_output.prev_sample, scheduler_output.pred_original_sample

            if self.save_intermediate_images_path:
                self.save_sampled_img(pred_x0, i, self.save_intermediate_images_path)

        self.save_sampled_img(self.target_latent, "output")


if __name__ == "__main__":
    args = parse_args()

    i2i_pipeline = I2IPipeline(args)
    i2i_pipeline.run_i2i()
