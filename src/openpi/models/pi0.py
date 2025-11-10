import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at
from jax import lax

logger = logging.getLogger("openpi")

class MLP(nnx.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, *, rngs):
        self.fc1 = nnx.Linear(in_dim, hidden_dim, rngs=rngs)
        self.fc2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.fc3 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        # self.fc4 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        # self.fc5 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.fc6 = nnx.Linear(hidden_dim, out_dim, rngs=rngs)

        self.activation_1 = nnx.swish
        self.activation_2 = nnx.swish
        self.activation_3 = nnx.swish
        # self.activation_4 = nnx.swish
        # self.activation_5 = nnx.swish

    def __call__(self, x):
        x = self.activation_1(self.fc1(x))
        x = self.activation_2(self.fc2(x))
        x = self.activation_3(self.fc3(x))
        # x = self.activation_4(self.fc4(x))
        # x = self.activation_5(self.fc5(x))
        x = self.fc6(x)

        return x


def vicreg_loss(z1, z2, lambda_param=25.0, mu_param=25.0, nu_param=1.0, gamma=1.0, eps=1e-4):
    """VICReg loss with Variance-Invariance-Covariance Regularization.

    Args:
        z1: First representation [batch, num_tokens, dim]
        z2: Second representation [batch, num_tokens, dim]
        lambda_param: Weight for invariance loss (default: 25.0)
        mu_param: Weight for variance loss (default: 25.0)
        nu_param: Weight for covariance loss (default: 1.0)
        gamma: Target standard deviation (default: 1.0)
        eps: Small constant for numerical stability

    Returns:
        VICReg loss value [batch*num_tokens]
    """
    batch_size, num_tokens, dim = z1.shape

    z1_bt = jnp.reshape(z1, (-1, dim))
    z2_bt = jnp.reshape(z2, (-1, dim))
    bt = z1_bt.shape[0]

    # Use L2 distance for invariance loss
    invariance_loss = jnp.mean(jnp.square(z1_bt - z2_bt), axis=-1)  # [batch*num_tokens]


    # Pairwise L2 distance matrix: ||z1[i] - z2[j]||^2
    # Expanding: ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a^T*b
    z1_norm_sq = jnp.sum(z1_bt ** 2, axis=-1, keepdims=True)  # [bt, 1]
    z2_norm_sq = jnp.sum(z2_bt ** 2, axis=-1, keepdims=True)  # [bt, 1]
    pairwise_l2_sq = z1_norm_sq + z2_norm_sq.T - 2 * jnp.matmul(z1_bt, z2_bt.T)  # [bt, bt]

    matrix_l2_mean = jnp.mean(jnp.sqrt(jnp.maximum(pairwise_l2_sq, 0)))  # avoid negative due to numerical error
    diag_mean = jnp.mean(jnp.sqrt(jnp.maximum(jnp.diag(pairwise_l2_sq), 0)))
    offdiag_l2 = jnp.sqrt(jnp.maximum(pairwise_l2_sq, 0)) * (1.0 - jnp.eye(bt))
    offdiag_mean = jnp.sum(offdiag_l2) / (bt * bt - bt)

    # jax.debug.print("VICReg dims: B={b}, T={t}, D={d}", b=z1.shape[0], t=z1.shape[1], d=z1.shape[2])
    # jax.debug.print("invariance_loss mean={m}", m=jnp.mean(invariance_loss))

    # Reshape to (batch*num_tokens, dim) for global variance/covariance computation
    z1_flat = jnp.reshape(z1, (-1, dim))  # [batch*num_tokens, dim]
    z2_flat = jnp.reshape(z2, (-1, dim))  # [batch*num_tokens, dim]
    n_samples = z1_flat.shape[0]  # batch * num_tokens

    # Compute variance across all batch and tokens
    std_z1 = jnp.sqrt(jnp.var(z1_flat, axis=0) + eps)  # [dim]
    std_z2 = jnp.sqrt(jnp.var(z2_flat, axis=0) + eps)  # [dim]

    variance_loss = jnp.mean(jax.nn.relu(gamma - std_z1)) + jnp.mean(jax.nn.relu(gamma - std_z2))

    # Compute covariance across all batch and tokens
    z1_centered = z1_flat - jnp.mean(z1_flat, axis=0, keepdims=True)  # [batch*num_tokens, dim]
    z2_centered = z2_flat - jnp.mean(z2_flat, axis=0, keepdims=True)  # [batch*num_tokens, dim]

    cov_z1 = (z1_centered.T @ z1_centered) / (n_samples - 1)  # [dim, dim]
    cov_z2 = (z2_centered.T @ z2_centered) / (n_samples - 1)  # [dim, dim]

    off_diagonal_mask = 1 - jnp.eye(dim)
    offdiag_z1 = cov_z1 * off_diagonal_mask
    offdiag_z2 = cov_z2 * off_diagonal_mask

    # Covariance loss
    cov_loss_z1 = jnp.sum(jnp.square(offdiag_z1)) / dim
    cov_loss_z2 = jnp.sum(jnp.square(offdiag_z2)) / dim
    covariance_loss = cov_loss_z1 + cov_loss_z2

    # Off-diagonal ratio diagnostics
    fro_z1 = jnp.linalg.norm(cov_z1)
    fro_z2 = jnp.linalg.norm(cov_z2)
    ratio_z1 = jnp.linalg.norm(offdiag_z1) / (fro_z1 + eps)
    ratio_z2 = jnp.linalg.norm(offdiag_z2) / (fro_z2 + eps)

    # Summarize diagnostics
    jax.debug.print(
        "invariance_loss z1={a}",
        a=jnp.mean(lambda_param * invariance_loss),
    )

    # Log the matrix-level L2 distance diagnostic
    jax.debug.print(
        "matrix_l2_mean={a}",
        a=matrix_l2_mean,
    )
    jax.debug.print(
        "pairwise L2: diag_mean={a}, offdiag_mean={b}",
        a=diag_mean,
        b=offdiag_mean,
    )

    jax.debug.print(
        "std_mean z1={a}, z2={b}, variance_loss={c}",
        a=jnp.mean(std_z1),
        b=jnp.mean(std_z2),
        c=mu_param * variance_loss
    )
    jax.debug.print(
        "cov_offdiag_ratio z1={a}, z2={b}",
        a=ratio_z1,
        b=ratio_z2,
    )
    jax.debug.print(
        "cov_loss split mean: z1={a}, z2={b}, covariance_loss={c}",
        a=cov_loss_z1,
        b=cov_loss_z2,
        c=nu_param * covariance_loss
    )

    total_loss = (
            lambda_param * invariance_loss +
            mu_param * variance_loss +
            nu_param * covariance_loss
    )

    return total_loss


def make_attn_mask_interleaved(input_mask, group_size=11, num_groups=5):
    """Create attention mask for interleaved tokens where each group can only attend to itself.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      group_size: int, size of each group (default 11: 10 action tokens + 1 cls token)
      num_groups: int, number of groups (default 5)

    Returns:
      Attention mask where tokens in each group can only attend to tokens in the same group.
    """
    batch_size = input_mask.shape[0]
    seq_len = input_mask.shape[1]

    # Create group indices for each token
    # Group 0: [0, 1, ..., 10], Group 1: [11, 12, ..., 21], etc.
    token_indices = jnp.arange(seq_len)
    group_indices = token_indices // group_size  # [0,0,0,...,0, 1,1,1,...,1, ...]

    # Create attention mask: token i can attend to token j if they're in the same group
    attn_mask = group_indices[None, :, None] == group_indices[None, None, :]  # [1, seq_len, seq_len]
    attn_mask = jnp.broadcast_to(attn_mask, (batch_size, seq_len, seq_len))

    # Apply valid mask
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]

    return jnp.logical_and(attn_mask, valid_mask)

def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class Pi0(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])

        action_cls_head = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )

        action_cls_head.lazy_init(rngs=rngs, method="init", use_adarms=[False])
        self.action_cls_head_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)

        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img, action_cls_head=action_cls_head)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)


        self.cls_head_count = 5
        cls_head_count = self.cls_head_count
        self.obs_cls_proj = nnx.Dict(
            **{
                f"head_{i}": MLP(
                    paligemma_config.width, 2 * paligemma_config.width, 256, rngs=rngs
                )
                for i in range(cls_head_count)
            }
        )
        self.act_cls_proj = nnx.Dict(head_0=MLP(
                    action_expert_config.width, 2 * paligemma_config.width, 256, rngs=rngs
                )
        )

        # 添加两个可学习的参数（避免把初始化函数作为模块静态字段）
        self.pre_cls_param = nnx.Param(nnx.initializers.normal()(rngs(), (1, 1, paligemma_config.width)))
        self.suf_cls_param = nnx.Param(nnx.initializers.normal()(rngs(), (1, 1, action_expert_config.width)))

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation, with_cls: bool = False
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)

        if with_cls:
            tokens = jnp.concatenate([tokens, jnp.repeat(self.pre_cls_param.value, repeats=tokens.shape[0], axis=0)],axis=-2)
            ar_mask += [True]
            input_mask.append(jnp.ones((tokens.shape[0], 1), dtype=jnp.bool_))

        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            # add a single state token
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    @override
    def compute_loss(
            self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *,
            train: bool = False, cls_train: bool = False, gamma: float = 1.0, t_step: float= 0.0
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)

        if cls_train:
            # Use dedicated function that freezes all params except pre_cls_param and suf_cls_param
            return self._compute_cls_loss_with_frozen_params(rng, observation, actions, gamma, t_step)

        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # one big forward pass of prefix + suffix at once
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        return jnp.mean(jnp.square(v_t - u_t), axis=-1)

    def _compute_cls_loss_with_frozen_params(
            self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, gamma: float = 1.0,
            t_step: float = 0.0
    ) -> at.Float[at.Array, "*b"]:
        """Compute cls loss with all parameters frozen except pre_cls_param and suf_cls_param.

        Parameter freezing is handled at the optimizer level via freeze_filter,
        so stop_gradient is not needed here.
        """
        observation = _model.preprocess_observation(None, observation, train=False)
        batch_size = observation.state.shape[0]

        # Embed prefix (parameters are frozen via optimizer filter)
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        for name in observation.images:
            image_tokens, _ = self.PaliGemma.img(observation.images[name], train=False)
            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    observation.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            ar_mask += [False] * image_tokens.shape[1]

        # add language
        if observation.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(observation.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(observation.tokenized_prompt_mask)
            ar_mask += [False] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)

        # pre_cls_param is trainable (not frozen by optimizer filter)
        pre_cls_tokens = jnp.repeat(self.pre_cls_param.value, repeats=tokens.shape[0], axis=0)
        prefix_tokens = jnp.concatenate([tokens, pre_cls_tokens], axis=-2)
        ar_mask = ar_mask + [True]
        input_mask.append(jnp.ones((tokens.shape[0], 1), dtype=jnp.bool_))
        prefix_mask = jnp.concatenate(input_mask, axis=1)
        prefix_ar_mask = jnp.array(ar_mask)

        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1

        # PaliGemma output
        (prefix_out, _), kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask,
                                                       positions=positions)

        x_t= actions

        suffix_input_mask = []
        suffix_tokens = []
        action_expert_tokens = self.action_cls_head_proj(x_t)

        suffix_tokens.append(action_expert_tokens)
        suffix_tokens_concat = jnp.concatenate(suffix_tokens, axis=1)
        suf_cls_value = jnp.repeat(self.suf_cls_param.value, 5, axis=1)
        suf_cls_tokens = jnp.repeat(suf_cls_value, repeats=suffix_tokens_concat.shape[0], axis=0)

        # 将 suf_cls_tokens 的每个头插入到对应位置
        # suf_cls_tokens shape: (batch, 5, width)
        # suffix_tokens_concat shape: (batch, 50, width)
        tokens_parts = []
        for i in range(self.cls_head_count):
            start_idx = i * 10
            end_idx = (i + 1) * 10
            # 添加这一段的 action tokens (10个)
            tokens_parts.append(suffix_tokens_concat[:, start_idx:end_idx, :])
            # 添加对应的 cls token (1个)
            tokens_parts.append(suf_cls_tokens[:, i:i + 1, :])

        suffix_tokens = jnp.concatenate(tokens_parts, axis=1)

        suffix_input_mask.append(jnp.ones(suffix_tokens.shape[:2], dtype=jnp.bool_))

        suffix_mask = jnp.concatenate(suffix_input_mask, axis=1)

        # 使用新的交错掩码函数，每11个token（10个action + 1个cls）形成一组
        suffix_attn_mask = make_attn_mask_interleaved(suffix_mask, group_size=11, num_groups=5)

        positions = jnp.tile(jnp.arange(11), 5)[None, :]  # shape: (1, 55)
        positions = jnp.broadcast_to(positions, (batch_size, 55))

        (suffix_out,), _ = self.PaliGemma.action_cls_head(
            [suffix_tokens],
            mask=suffix_attn_mask,
            positions=positions,
            kv_cache=None,
            adarms_cond=[None],
        )

        obs_cls_out = jnp.stack(
            [getattr(self.obs_cls_proj, f"head_{i}")(prefix_out[:, :1, :]) for i in range(self.cls_head_count)],
            axis=1,
        )
        obs_cls_out = jnp.squeeze(obs_cls_out, axis=2)

        act_cls_out = jnp.stack(
            [
                getattr(self.act_cls_proj, "head_0")(suffix_out[:, i * 11 + 10: i * 11 + 11, :])
                for i in range(self.cls_head_count)
            ],
            axis=1,
        )
        act_cls_out = jnp.squeeze(act_cls_out, axis=2)

        # 直接使用原始表征，不做归一化
        vicreg = vicreg_loss(
            obs_cls_out,
            act_cls_out,
            lambda_param=50 * jax.nn.sigmoid(t_step / 300 - 3),
            mu_param=50.0,
            nu_param=1.0,
            gamma=0.5,  # 降低到现实可达的目标，避免与协方差损失冲突
        )

        return vicreg

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
        old_obs_cls_repr: at.Float[at.Array, "hd"] = None,
        force_sample: bool = False,
        delta_replan: int = 0,
    ) -> (_model.Actions, at.Float[at.Array, "hd"], bool):
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation, with_cls=True)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        (prefix_out, _), kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        cache_k, cache_v = kv_cache
        cache_k = cache_k[:, :, :-1, :, :]  # 去掉序列维度的最后一个
        cache_v = cache_v[:, :, :-1, :, :]  # 去掉序列维度的最后一个
        kv_cache = (cache_k, cache_v)

        now_obs_cls_repr = prefix_out[0, 0]
        head_idx = jnp.minimum(delta_replan, self.cls_head_count - 1)
        old_obs_cls_head = (
            None
            if old_obs_cls_repr is None
            else self._apply_obs_cls_head(head_idx, old_obs_cls_repr)
        )
        now_obs_cls_head = getattr(self.obs_cls_proj, "head_0")(now_obs_cls_repr)


        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        def skip(carry):
            x_t, time = carry
            return x_t, time

        # if old_obs_cls_head is not None:
        #     jax.debug.print("old_obs_cls_head={a}", a=old_obs_cls_head[:20])
        #     jax.debug.print("now_obs_cls_head={a}", a=now_obs_cls_head[:20])
        #     jax.debug.print("now_obs_cls_repr={a}", a=now_obs_cls_repr[:20])
        if old_obs_cls_head is not None:
            # Use L2 distance instead of cosine similarity
            l2_distance = jnp.mean(jnp.square(now_obs_cls_head - old_obs_cls_head), axis=-1)
            # 使用L2距离阈值（需要根据实际表征尺度调整）
            should_sample = l2_distance > 0.3
            jax.debug.print("l2_distance={a}", a=l2_distance)
        else:
            # If old_obs_cls_head is None, always sample
            should_sample = jnp.array(True)

        force_sample_array = jnp.asarray(force_sample, dtype=jnp.bool_)
        should_sample = jnp.logical_or(jnp.any(should_sample), force_sample_array)

        x_0, _ = lax.cond(
            should_sample,
            lambda operand: jax.lax.while_loop(cond, step, operand),
            lambda operand: skip(operand),
            (noise, 1.0)
        )
        #jax.debug.print("x_0={a}", a=x_0[1, :10,:7])
        #jax.debug.print("x_1={a}", a=x_0[1, 10:20, :7])
        return x_0, now_obs_cls_repr, should_sample
        #return x_0, old_obs_cls_head