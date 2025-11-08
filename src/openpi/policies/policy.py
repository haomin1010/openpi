from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device
        self.old_cls_head = None
        self.force_sample = True
        self.replan_count = 1
        self.task_id = -1

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._rng = rng or jax.random.key(0)

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        # 提取 task_id（如果存在）
        task_id = obs.get("task_id", None)
        if task_id != self.task_id:
            print("1111111111111111111")
            print("1111111111111111111")
            print("1111111111111111111")
            self.task_id = task_id
        self.force_sample = obs.get("force_sample", True)

        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        if self.old_cls_head is None and not self._is_pytorch_model:
            # 安全地获取维度，如果模型没有 suf_cls_param 则跳过初始化
            if hasattr(self._model, 'suf_cls_param') and hasattr(self._model.suf_cls_param, 'value'):
                cls_dim = self._model.suf_cls_param.value.shape[2]
                cls_dtype = self._model.suf_cls_param.value.dtype
                self.old_cls_head = jnp.array([0.0625,0.0634766,-0.0620117,0.0605469,-0.0610352,-0.0620117,-0.0629883,-0.0644531,-0.0610352,-0.0620117,-0.0603027,-0.0620117,0.0649414,-0.0629883,-0.0634766,-0.0620117,0.0625,0.0634766,-0.0610352,0.0620117,0.0610352,-0.0629883,-0.0634766,0.0617676,0.0625,0.0605469,-0.0625,0.0629883,-0.0595703,0.0605469,0.0644531,-0.0612793,0.0634766,-0.0598145,0.0620117,0.0605469,-0.0634766,0.0625,0.0629883,-0.0598145,-0.0634766,0.065918,0.0639648,-0.0617676,0.0634766,0.0620117,-0.0625,-0.0639648,0.0620117,0.0634766,-0.0629883,0.0634766,-0.0605469,-0.0612793,0.0634766,0.0629883,0.0634766,0.0625,-0.0605469,-0.0617676,0.0620117,0.0603027,0.0634766,0.0578613,0.0639648,-0.0620117,0.0617676,-0.0654297,0.0634766,0.0620117,-0.0629883,0.0612793,0.0620117,-0.0629883,0.0644531,-0.0629883,-0.0612793,0.0598145,0.0605469,0.0620117,0.0625,-0.0585938,-0.065918,-0.0620117,-0.0620117,-0.0644531,-0.0605469,-0.0605469,-0.0634766,-0.0644531,0.0629883,-0.0634766,0.0634766,-0.0620117,0.0612793,-0.0629883,-0.0620117,0.0625,0.0620117,-0.0605469,0.0629883,-0.0620117,0.0620117,0.0620117,-0.0629883,0.0625,-0.0625,0.0617676,0.0629883,0.0629883,0.0629883,-0.0625,0.0612793,0.0634766,0.0639648,-0.0617676,-0.0629883,-0.0629883,0.0620117,0.0629883,0.0617676,0.0617676,-0.0629883,0.0634766,0.0634766,-0.0639648,-0.0612793,0.0605469,0.0603027,0.0629883,0.0629883,-0.0649414,0.0620117,0.0610352,0.0617676,-0.0620117,-0.0603027,0.0625,-0.0617676,-0.0625,0.0629883,0.0598145,-0.0620117,-0.0620117,-0.0610352,0.0617676,-0.0605469,0.0612793,-0.0634766,0.0639648,-0.0649414,-0.0634766,0.0625,0.0620117,-0.0644531,0.0598145,0.0612793,-0.0617676,-0.0629883,-0.0634766,-0.0634766,0.0620117,0.0603027,0.0598145,-0.0634766,-0.0612793,0.0634766,-0.0605469,-0.0634766,-0.0617676,-0.0644531,0.0620117,0.0620117,0.0605469,-0.0610352,0.0629883,0.0639648,0.0644531,-0.0649414,-0.0629883,0.0639648,0.0629883,-0.0612793,-0.0620117,-0.0649414,0.0612793,0.0634766,0.0620117,0.0634766,-0.0629883,0.0634766,-0.0617676,-0.0612793,0.0625,0.0612793,0.0617676,-0.0634766,-0.0634766,0.0634766,-0.0649414,0.0644531,-0.0644531,-0.0603027,-0.0610352,0.0617676,-0.0617676,0.0610352,-0.0629883,0.0598145,-0.0629883,-0.0629883,0.0629883,0.0634766,0.0639648,-0.0620117,-0.0629883,0.0629883,-0.0603027,-0.0603027,0.0595703,0.0629883,0.0617676,-0.0598145,0.0612793,0.0649414,0.0634766,-0.0625,0.0649414,0.0603027,0.0612793,0.0620117,-0.0634766,-0.0612793,-0.0634766,-0.0629883,-0.0620117,-0.0634766,-0.0634766,-0.0620117,-0.0612793,0.0625,0.0620117,-0.0629883,0.0639648,-0.0620117,-0.0629883,-0.0617676,-0.0605469,-0.0634766,-0.0629883,0.0625,0.0625,0.059082,0.0620117,-0.0634766,-0.0634766])

        sample_kwargs["old_obs_cls_head"] = self.old_cls_head
        sample_kwargs["force_sample"] = self.force_sample

        start_time = time.monotonic()
        actions, cls_head, have_sample =  self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs)
        self.force_sample = not have_sample
        outputs = {
            "state": inputs["state"],
            "actions": actions,
        }
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        outputs = self._output_transform(outputs)
        if not have_sample:
            print("--------")
            outputs["actions"] = None
            self.replan_count += 1
        else:
            self.old_cls_head = cls_head
            self.replan_count = 1

        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
