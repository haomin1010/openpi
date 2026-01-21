import dataclasses
import numpy as np
from openpi import transforms
from openpi.models import model as _model
from openpi.policies.libero_policy import _parse_image  # 或复制同样逻辑

KINOVA_DIM = 8  # 7 joints + 1 gripper

@dataclasses.dataclass(frozen=True)
class KinovaInputs(transforms.DataTransformFn):
    model_type: _model.ModelType
    valid_dim: int = KINOVA_DIM

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        state = np.asarray(data["observation/state"])[..., : self.valid_dim]

        out = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        if "actions" in data:
            out["actions"] = np.asarray(data["actions"])[..., : self.valid_dim]

        if "prompt" in data:
            out["prompt"] = data["prompt"].decode("utf-8") if isinstance(data["prompt"], bytes) else data["prompt"]

        return out

@dataclasses.dataclass(frozen=True)
class KinovaOutputs(transforms.DataTransformFn):
    valid_dim: int = KINOVA_DIM

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, : self.valid_dim])}