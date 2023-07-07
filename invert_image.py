import os
import torch
import argparse
import numpy as np
import PIL
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from diffusers_img2img.src.diffusers.schedulers import DDIMInverseScheduler
from transformers import CLIPTextModel
from transformers import AutoTokenizer
from diffusers_img2img.src.diffusers import AutoencoderKL, UNet2DConditionModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model"
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default=None,
        required=True,
        help="Path to the image that will be inverted"
    )
    parser.add_argument(
        "--save_intermediate_images_path",
        type=str,
        default=None,
        help="Path to save the intermediate images from the DDIM Inversion"
    )

    args = parser.parse_args()

    return args


class DDIMInversion:
    def __init__(self, args):
        self.pretrained_model_path = args.pretrained_model_path
        self.image_path = args.image_path
        self.save_intermediate_images_path = args.save_intermediate_images_path

        self.prompt = ""

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(512),
            ]
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_path, subfolder="tokenizer", use_fast=False)
        self.noise_scheduler_inverse = DDIMInverseScheduler.from_pretrained(self.pretrained_model_path, subfolder="scheduler")
        self.text_encoder = CLIPTextModel.from_pretrained(self.pretrained_model_path, subfolder="text_encoder")
        self.vae = AutoencoderKL.from_pretrained(self.pretrained_model_path, subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained(self.pretrained_model_path, subfolder="unet")

        self.vae.to(self.device)
        self.text_encoder.to(self.device)
        self.unet.to(self.device)

        self.vae.eval()
        self.text_encoder.eval()
        self.unet.eval()


    def preprocess_image(self, image):
        if isinstance(image, torch.Tensor):
            return image
        elif isinstance(image, PIL.Image.Image):
            image = [image]

        if isinstance(image[0], PIL.Image.Image):
            w, h = image[0].size
            w, h = map(lambda x: x - x % 8, (w, h))  # resize to integer multiple of 8

            image = [np.array(i.resize((w, h)))[None, :] for i in image]
            image = np.concatenate(image, axis=0)
            image = np.array(image).astype(np.float32) / 255.0
            image = image.transpose(0, 3, 1, 2)
            image = 2.0 * image - 1.0
            image = torch.from_numpy(image)

        elif isinstance(image[0], torch.Tensor):
            image = torch.cat(image, dim=0)

        image = self.image_transforms(image)

        return image


    def save_sampled_img(self, x, i, save_path):
        latents = 1 / self.vae.config.scaling_factor * x
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        image = (image * 255).round().astype("uint8")
        img = Image.fromarray(image[0])
        img.save(os.path.join(save_path, f"{i}.png"))


    @torch.no_grad()
    def run_inversion(self):
        image = Image.open(self.image_path).convert("RGB")
        image_preprocessed = self.preprocess_image(image)
        image_preprocessed = image_preprocessed.to(self.device)

        if self.save_intermediate_images_path:
            os.makedirs(self.save_intermediate_images_path, exist_ok=True)

        prompt_embeds = self.tokenizer(self.prompt,
                                       truncation=True,
                                       padding="max_length",
                                       max_length=self.tokenizer.model_max_length,
                                       return_tensors="pt").input_ids

        encoder_hidden_states = self.text_encoder(prompt_embeds.to(self.device))[0]

        self.noise_scheduler_inverse.set_timesteps(999, device=self.device)
        timesteps = self.noise_scheduler_inverse.timesteps

        latents = self.vae.encode(image_preprocessed).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor

        progress_bar = tqdm(enumerate(timesteps), desc='DDIM Inversion', total=999)

        for i, t in progress_bar:
            scaled_latents = self.noise_scheduler_inverse.scale_model_input(latents, t)
            model_pred = self.unet(scaled_latents, t, encoder_hidden_states).sample
            latents, pred_x0 = self.noise_scheduler_inverse.step(model_pred, t, latents, return_dict=False)

            if self.save_intermediate_images_path:
                self.save_sampled_img(pred_x0, i, self.save_intermediate_images_path)

        torch.save(latents, "./inverted_image.pt")


if __name__ == "__main__":
    args = parse_args()

    ddim_inversion = DDIMInversion(args)
    ddim_inversion.run_inversion()
