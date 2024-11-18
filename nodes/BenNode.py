import os
import torch
from torchvision import transforms
from torch.hub import download_url_to_file
import comfy
from comfy import model_management
import folder_paths
from .libs.model import BEN_Base
from .util import tensor_to_pil, normalize_mask, apply_mask_to_image

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


proc_img = transforms.Compose(
    [
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

class LoadRembgByBenModel:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (folder_paths.get_filename_list(models_dir_key),),
                "device": (["AUTO", "CPU"], )
            }
        }

    RETURN_TYPES = ("BenMODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "rembg/Ben"
    DESCRIPTION = "Load BEN model from folder models/rembg/ben or the path of ben configured in the extra YAML file"

    def load_model(self, model, device):

        ben_model = BEN_Base()

        model_path = folder_paths.get_full_path(models_dir_key, model)

        if device == "AUTO":
            device_type = deviceType
        else:
            device_type = "cpu"

        ben_model.loadcheckpoints(model_path)
        ben_model.to(device_type)
        ben_model.eval()
        return (ben_model, )


class RembgByBen:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("BenMODEL",),
                "images": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK",)
    RETURN_NAMES = ("image", "mask",)
    FUNCTION = "rem_bg"
    CATEGORY = "rembg/Ben"

    def rem_bg(self, model, images):
        model_device_type = next(model.parameters()).device.type
        _images = []
        _masks = []

        for image in images:
            h, w, c = image.shape
            pil_image = tensor_to_pil(image)
            im_tensor = proc_img(pil_image).unsqueeze(0)

            with torch.no_grad():
                mask = model(im_tensor.to(model_device_type)).cpu()

            # 遮罩大小需还原为与原图一致
            mask = comfy.utils.common_upscale(mask, w, h, 'bilinear', "disabled")

            mask = normalize_mask(mask)
            # image的非mask对应部分设为透明
            image = apply_mask_to_image(image.cpu(), mask.cpu())

            _images.append(image)
            _masks.append(mask.squeeze(0))

        out_images = torch.cat(_images, dim=0)
        out_masks = torch.cat(_masks, dim=0)

        return out_images, out_masks


NODE_CLASS_MAPPINGS = {
    "LoadRembgByBenModel": LoadRembgByBenModel,
    "RembgByBen": RembgByBen,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadRembgByBiRefNetModel": "LoadRembgByBiRefNetModel",
    "RembgByBen": "RembgByBen",
}
