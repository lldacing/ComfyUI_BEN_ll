import os
import torch
from torchvision import transforms
from torch.hub import download_url_to_file
import comfy
from comfy import model_management
import folder_paths
from .libs.model import BEN_Base
from .libs.model_BEN2 import BEN_Base as BEN2_Base
from .util import filter_mask, add_mask_as_alpha, refine_foreground

deviceType = model_management.get_torch_device().type

models_dir_key = "ben"
models_path_default = folder_paths.get_folder_paths(models_dir_key)[0]

def download_models(model_root, model_urls):
    if not os.path.exists(model_root):
        os.makedirs(model_root, exist_ok=True)

    for local_file, url in model_urls:
        local_path = os.path.join(model_root, local_file)
        if not os.path.exists(local_path):
            local_path = os.path.abspath(os.path.join(model_root, local_file))
            download_url_to_file(url, dst=local_path)


def download_ben_model():
    """
    Downloading model from huggingface.
    """
    model_root = os.path.join(models_path_default)
    model_urls = (
        ("BEN_Base.pth",
         "https://huggingface.co/PramaLLC/BEN/resolve/main/BEN_Base.pth"),
    )
    download_models(model_root, model_urls)


interpolation_modes_mapping = {
    "nearest": 0,
    "bilinear": 2,
    "bicubic": 3,
    "nearest-exact": 0,
    # "lanczos": 1, #不支持
}

class ImagePreprocessor:
    def __init__(self, resolution, upscale_method="bilinear") -> None:
        interpolation = interpolation_modes_mapping.get(upscale_method, 2)
        self.transform_image = transforms.Compose([
            transforms.Resize(resolution, interpolation=interpolation),
            # transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def proc(self, image) -> torch.Tensor:
        image = self.transform_image(image)
        return image

class LoadRembgByBenModel:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (folder_paths.get_filename_list(models_dir_key),),
                "device": (["AUTO", "CPU"], )
            }
        }

    RETURN_TYPES = ("BEN",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "rembg/Ben"
    DESCRIPTION = "Load BEN model from folder models/rembg/ben or the path of ben configured in the extra YAML file"

    def load_model(self, model, device):

        model_path = folder_paths.get_full_path(models_dir_key, model)

        if device == "AUTO":
            device_type = deviceType
        else:
            device_type = "cpu"

        index = model.find("BEN2_Base")
        if index > -1:
            ben_model = BEN2_Base()
        else:
            ben_model = BEN_Base()
        ben_model.loadcheckpoints(model_path)
        ben_model.to(device_type)
        ben_model.eval()
        return (ben_model, )


class GetMaskByBen:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("BEN",),
                "images": ("IMAGE",),
                "width": ("INT",
                          {
                              "default": 1024,
                              "min": 0,
                              "max": 16384,
                              "tooltip": "The width of the pre-processing image, does not affect the final output image size"
                          }),
                "height": ("INT",
                           {
                               "default": 1024,
                               "min": 0,
                               "max": 16384,
                               "tooltip": "The height of the pre-processing image, does not affect the final output image size"
                           }),
                "upscale_method": (["bilinear", "nearest", "nearest-exact", "bicubic"],
                                   {
                                       "default": "bilinear",
                                       "tooltip": "Interpolation method for pre-processing image and post-processing mask"
                                   }),
                "mask_threshold": ("FLOAT", {"default": 0.000, "min": 0.0, "max": 1.0, "step": 0.004, }),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "get_mask"
    CATEGORY = "rembg/Ben"

    def get_mask(self, model, images, width=1024, height=1024, upscale_method='bilinear', mask_threshold=0.000):
        model_device_type = next(model.parameters()).device.type
        b, h, w, c = images.shape
        image_bchw = images.permute(0, 3, 1, 2)

        image_preproc = ImagePreprocessor(resolution=(height, width), upscale_method=upscale_method)
        im_tensor = image_preproc.proc(image_bchw)

        del image_preproc

        _mask_bchw = []
        for each_image in im_tensor:
            with torch.no_grad():
                each_mask = model(each_image.unsqueeze(0).to(model_device_type)).cpu()
            _mask_bchw.append(each_mask)
            del each_mask

        mask_bchw = torch.cat(_mask_bchw, dim=0)
        del _mask_bchw
        # 遮罩大小需还原为与原图一致
        mask = comfy.utils.common_upscale(mask_bchw, w, h, upscale_method, "disabled")
        # (b, 1, h, w)
        if mask_threshold > 0:
            mask = filter_mask(mask, threshold=mask_threshold)

        return mask.squeeze(1),


class BlurFusionForegroundEstimation:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "masks": ("MASK",),
                "blur_size": ("INT", {"default": 91, "min": 1, "max": 255, "step": 2, }),
                "blur_size_two": ("INT", {"default": 7, "min": 1, "max": 255, "step": 2, }),
                "fill_color": ("BOOLEAN", {"default": False}),
                "color": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFF, "step": 1, "display": "color"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK",)
    RETURN_NAMES = ("image", "mask",)
    FUNCTION = "get_foreground"
    CATEGORY = "rembg/Ben"
    DESCRIPTION = "Approximate Fast Foreground Colour Estimation. https://github.com/Photoroom/fast-foreground-estimation"

    def get_foreground(self, images, masks, blur_size=91, blur_size_two=7, fill_color=False, color=None):
        b, h, w, c = images.shape
        if b != masks.shape[0]:
            raise ValueError("images and masks must have the same batch size")

        image_bchw = images.permute(0, 3, 1, 2)

        if masks.dim() == 3:
            # (b, h, w) => (b, 1, h, w)
            out_masks = masks.unsqueeze(1)

        # (b, c, h, w)
        _image_masked = refine_foreground(image_bchw, out_masks, r1=blur_size, r2=blur_size_two)
        # (b, c, h, w) => (b, h, w, c)
        _image_masked = _image_masked.permute(0, 2, 3, 1)
        if fill_color and color is not None:
            r = torch.full([b, h, w, 1], ((color >> 16) & 0xFF) / 0xFF)
            g = torch.full([b, h, w, 1], ((color >> 8) & 0xFF) / 0xFF)
            b = torch.full([b, h, w, 1], (color & 0xFF) / 0xFF)
            # (b, h, w, 3)
            background_color = torch.cat((r, g, b), dim=-1)
            # (b, 1, h, w) => (b, h, w, 3)
            apply_mask = out_masks.permute(0, 2, 3, 1).expand_as(_image_masked)
            out_images = _image_masked * apply_mask + background_color * (1 - apply_mask)
            # (b, h, w, 3)=>(b, h, w, 3)
            del background_color, apply_mask
            out_masks = out_masks.squeeze(1)
        else:
            # (b, 1, h, w) => (b, h, w)
            out_masks = out_masks.squeeze(1)
            # image的非mask对应部分设为透明 => (b, h, w, 4)
            out_images = add_mask_as_alpha(_image_masked.cpu(), out_masks.cpu())

        del _image_masked

        return out_images, out_masks


class RembgByBenAdvanced(GetMaskByBen, BlurFusionForegroundEstimation):

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("BEN",),
                "images": ("IMAGE",),
                "width": ("INT",
                          {
                              "default": 1024,
                              "min": 0,
                              "max": 16384,
                              "tooltip": "The width of the pre-processing image, does not affect the final output image size"
                          }),
                "height": ("INT",
                           {
                               "default": 1024,
                               "min": 0,
                               "max": 16384,
                               "tooltip": "The height of the pre-processing image, does not affect the final output image size"
                           }),
                "upscale_method": (["bilinear", "nearest", "nearest-exact", "bicubic"],
                                   {
                                       "default": "bilinear",
                                       "tooltip": "Interpolation method for pre-processing image and post-processing mask"
                                   }),
                "blur_size": ("INT", {"default": 91, "min": 1, "max": 255, "step": 2, }),
                "blur_size_two": ("INT", {"default": 7, "min": 1, "max": 255, "step": 2, }),
                "fill_color": ("BOOLEAN", {"default": False}),
                "color": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFF, "step": 1, "display": "color"}),
                "mask_threshold": ("FLOAT", {"default": 0.000, "min": 0.0, "max": 1.0, "step": 0.001, }),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK",)
    RETURN_NAMES = ("image", "mask",)
    FUNCTION = "rem_bg"
    CATEGORY = "rembg/Ben"

    def rem_bg(self, model, images, upscale_method='bilinear', width=1024, height=1024, blur_size=91, blur_size_two=7, fill_color=False, color=None, mask_threshold=0.000):

        masks = super().get_mask(model, images, width, height, upscale_method, mask_threshold)

        out_images, out_masks = super().get_foreground(images, masks=masks[0], blur_size=blur_size, blur_size_two=blur_size_two, fill_color=fill_color, color=color)

        return out_images, out_masks


class RembgByBen(RembgByBenAdvanced):

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("BEN",),
                "images": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK",)
    RETURN_NAMES = ("image", "mask",)
    FUNCTION = "rem_bg"
    CATEGORY = "rembg/Ben"

    def rem_bg(self, model, images):
        return super().rem_bg(model, images)


NODE_CLASS_MAPPINGS = {
    "LoadRembgByBenModel": LoadRembgByBenModel,
    "RembgByBen": RembgByBen,
    "GetMaskByBen": GetMaskByBen,
    "RembgByBenAdvanced": RembgByBenAdvanced,
    "BlurFusionForegroundEstimationForBen": BlurFusionForegroundEstimation,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadRembgByBenModel": "LoadRembgByBenModel",
    "RembgByBen": "RembgByBen",
    "GetMaskByBen": "GetMaskByBen",
    "RembgByBenAdvanced": "RembgByBenAdvanced",
    "BlurFusionForegroundEstimationForBen": "BlurFusionForegroundEstimationForBen",
}
