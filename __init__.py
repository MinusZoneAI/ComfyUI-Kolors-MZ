import inspect
import json
import os
import folder_paths
import importlib


NODE_CLASS_MAPPINGS = {
}


NODE_DISPLAY_NAME_MAPPINGS = {
}

AUTHOR_NAME = "MinusZone"
CATEGORY_NAME = f"{AUTHOR_NAME} - Kolors"
folder_paths.add_model_folder_path(
    "LLM", os.path.join(folder_paths.models_dir, "LLM"))


class MZ_ChatGLM3Loader:
    @classmethod
    def INPUT_TYPES(s):
        # from .mz_kolors_utils import Utils
        # llm_dir = os.path.join(Utils.get_models_path(), "LLM")
        # print("llm_dir:", llm_dir)
        # llm_models = Utils.listdir_models(llm_dir)

        return {"required": {
            "chatglm3_checkpoint": (folder_paths.get_filename_list("LLM"),),
        }}

    RETURN_TYPES = ("CHATGLM3MODEL",)
    RETURN_NAMES = ("chatglm3_model",)
    FUNCTION = "load_chatglm3"
    CATEGORY = CATEGORY_NAME

    def load_chatglm3(self, **kwargs):
        from . import mz_kolors_core
        importlib.reload(mz_kolors_core)
        return mz_kolors_core.MZ_ChatGLM3Loader_call(kwargs)


NODE_CLASS_MAPPINGS["MZ_ChatGLM3Loader"] = MZ_ChatGLM3Loader
NODE_DISPLAY_NAME_MAPPINGS["MZ_ChatGLM3Loader"] = f"{AUTHOR_NAME} - ChatGLM3Loader"


class MZ_ChatGLM3TextEncodeV2:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "chatglm3_model": ("CHATGLM3MODEL", ),
                "text": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)

    FUNCTION = "encode"
    CATEGORY = CATEGORY_NAME

    def encode(self, **kwargs):
        from . import mz_kolors_core
        importlib.reload(mz_kolors_core)
        return mz_kolors_core.MZ_ChatGLM3TextEncodeV2_call(kwargs)


NODE_CLASS_MAPPINGS["MZ_ChatGLM3_V2"] = MZ_ChatGLM3TextEncodeV2
NODE_DISPLAY_NAME_MAPPINGS[
    "MZ_ChatGLM3_V2"] = f"{AUTHOR_NAME} - ChatGLM3TextEncodeV2"


class MZ_KolorsUNETLoaderV2():
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                "unet_name": (folder_paths.get_filename_list("unet"), ),
                }}

    RETURN_TYPES = ("MODEL",)

    RETURN_NAMES = ("model",)

    FUNCTION = "load_unet"

    CATEGORY = CATEGORY_NAME

    def load_unet(self, **kwargs):
        from . import mz_kolors_core
        importlib.reload(mz_kolors_core)
        return mz_kolors_core.MZ_KolorsUNETLoaderV2_call(kwargs)


NODE_CLASS_MAPPINGS["MZ_KolorsUNETLoaderV2"] = MZ_KolorsUNETLoaderV2
NODE_DISPLAY_NAME_MAPPINGS[
    "MZ_KolorsUNETLoaderV2"] = f"{AUTHOR_NAME} - KolorsUNETLoaderV2"


class MZ_KolorsControlNetPatch:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "control_net": ("CONTROL_NET", ),
                "model": ("MODEL", ),
            }
        }

    RETURN_TYPES = ("CONTROL_NET",)

    FUNCTION = "start"
    CATEGORY = CATEGORY_NAME

    def start(self, **kwargs):
        from . import mz_kolors_core
        importlib.reload(mz_kolors_core)
        return mz_kolors_core.MZ_KolorsControlNetPatch_call(kwargs)


NODE_CLASS_MAPPINGS["MZ_KolorsControlNetPatch"] = MZ_KolorsControlNetPatch
NODE_DISPLAY_NAME_MAPPINGS[
    "MZ_KolorsControlNetPatch"] = f"{AUTHOR_NAME} - KolorsControlNetPatch"


class MZ_KolorsCLIPVisionLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"clip_name": (folder_paths.get_filename_list("clip_vision"), ),
                             }}
    RETURN_TYPES = ("CLIP_VISION",)
    FUNCTION = "load_clip"

    CATEGORY = CATEGORY_NAME + "/Legacy"

    def load_clip(self, **kwargs):
        from . import mz_kolors_core
        importlib.reload(mz_kolors_core)
        return mz_kolors_core.MZ_KolorsCLIPVisionLoader_call(kwargs)


NODE_CLASS_MAPPINGS["MZ_KolorsCLIPVisionLoader"] = MZ_KolorsCLIPVisionLoader
NODE_DISPLAY_NAME_MAPPINGS["MZ_KolorsCLIPVisionLoader"] = f"{AUTHOR_NAME} - KolorsCLIPVisionLoader - Legacy"


from .ComfyUI_IPAdapter_plus.IPAdapterPlus import IPAdapterAdvanced, IPAdapterModelLoader

IPAdapterModelLoader.CATEGORY = CATEGORY_NAME + "/Legacy"
NODE_CLASS_MAPPINGS["MZ_IPAdapterModelLoaderKolors"] = IPAdapterModelLoader
NODE_DISPLAY_NAME_MAPPINGS[
    "MZ_IPAdapterModelLoaderKolors"] = f"IPAdapterModelLoader(kolors) - Legacy"

IPAdapterAdvanced.CATEGORY = CATEGORY_NAME + "/Legacy"
NODE_CLASS_MAPPINGS["MZ_IPAdapterAdvancedKolors"] = IPAdapterAdvanced
NODE_DISPLAY_NAME_MAPPINGS[
    "MZ_IPAdapterAdvancedKolors"] = f"IPAdapterAdvanced(kolors) - Legacy"

from . import mz_kolors_legacy
NODE_CLASS_MAPPINGS.update(mz_kolors_legacy.NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(mz_kolors_legacy.NODE_DISPLAY_NAME_MAPPINGS)
