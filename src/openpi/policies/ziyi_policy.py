import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_ziyi_example(*, state_dim: int = 15) -> dict:
    """Example input for a Tianji policy."""
    return {
        "observation.state": np.random.rand(state_dim),
        "observation.images.head": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation.images.left_wrist": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation.images.right_wrist": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    """Convert LeRobot image tensors to uint8 HWC arrays."""
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class ZiyiInputs(transforms.DataTransformFn):
    """Map Tianji 15D data into the standard pi input schema."""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        images = data["images"]
        inputs = {
            "state": data["state"],
            "image": {
                "base_0_rgb": _parse_image(images["head"]),
                "left_wrist_0_rgb": _parse_image(images["left_wrist"]),
                "right_wrist_0_rgb": _parse_image(images["right_wrist"]),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class ZiyiOutputs(transforms.DataTransformFn):
    """Trim model actions back to the Tianji action space."""

    action_dim: int = 15

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, : self.action_dim])}


@dataclasses.dataclass(frozen=True)
class ZiyiDataTransforms:
    """Pickle-safe transform factory for Tianji policy configs."""

    action_dim: int = 15

    def __call__(self, model_config):
        return transforms.Group(
            inputs=[ZiyiInputs(model_type=model_config.model_type)],
            outputs=[ZiyiOutputs(action_dim=self.action_dim)],
        )
