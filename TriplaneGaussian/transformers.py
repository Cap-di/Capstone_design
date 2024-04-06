# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn.functional as F
from torch import nn

from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.activations import get_activation
from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import CombinedTimestepLabelEmbeddings

from dataclasses import dataclass
from tgs.utils.base import BaseModule
from tgs.utils.typing import *


## xFormers라이브러리의 memory_efficient_attention() 함수를 이용해 메모리 효율적인 어텐션을 활성화하거나 비활성화 하는 기능 제공
class MemoryEfficientAttentionMixin:
    def enable_xformers_memory_efficient_attention(         # xFormers의 메모리 효율적인 어텐션을 활성화하는 메서드
        self, attention_op: Optional[Callable] = None
    ):
        r"""
        [xFormers](https://facebookresearch.github.io/xformers/)에서 메모리 효율적인 어텐션을 활성화합니다. 
        이 옵션을 활성화하면 GPU 메모리 사용량이 줄어들고 추론 중에 속도 향상이 가능합니다. 훈련 중의 속도 향상은 보장되지 않습니다.
                    
        <Tip warning={true}>

        ⚠️ 메모리 효율적인 어텐션과 슬라이스 어텐션이 모두 활성화된 경우, 메모리 효율적인 어텐션이 우선합니다.

        </Tip>

        Parameters:
            attention_op (`Callable`, *optional*):
                xFormers의 
                [`memory_efficient_attention()`](https://facebookresearch.github.io/xformers/components/ops.html#xformers.ops.memory_efficient_attention) 
                함수의 `op` 인자로 사용할 기본 `None` 연산자를 재정의합니다.

        Examples:
        ```py
        >>> import torch
        >>> from diffusers import DiffusionPipeline
        >>> from xformers.ops import MemoryEfficientAttentionFlashAttentionOp

        >>> pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16)
        >>> pipe = pipe.to("cuda")
        >>> pipe.enable_xformers_memory_efficient_attention(attention_op=MemoryEfficientAttentionFlashAttentionOp)
<<<<<<< HEAD
=======
        ## 메모리 어텐션 비활성화
>>>>>>> 7560ecd41fe757fff5908faf1aff8bf71a9818eb
        >>> # Flash Attention을 사용하는 VAE에서 어텐션 형태를 받아들이지 않는 문제에 대한 해결책
        >>> pipe.vae.enable_xformers_memory_efficient_attention(attention_op=None)
        ```
        """
        self.set_use_memory_efficient_attention_xformers(True, attention_op)    # 메모리 효율적인 어텐션을 활성화

    def disable_xformers_memory_efficient_attention(self):
        r"""
        [xFormers](https://facebookresearch.github.io/xformers/) 의 메모리 효율적인 어텐션을 비활성화합니다.
        """
        self.set_use_memory_efficient_attention_xformers(False)


    def set_use_memory_efficient_attention_xformers(
        self, valid: bool, attention_op: Optional[Callable] = None
    ) -> None:
        # 메모리 효율적인 어텐션 변형기 사용 여부를 설정.
        # Parameters:
        # valid (bool): 메모리 효율적인 어텐션 변형기 사용 여부를 나타내는 불리언 값.
        # attention_op (Optional[Callable]): 선택적 매개변수로, 메모리 효율적인 어텐션 연산을 수행하는 함수. 기본값은 None.
        # Returns: None

        # torch.nn.Module 클래스를 상속받는 모든 자식 모듈에 대해 재귀적으로 탐색.
        # set_use_memory_efficient_attention_xformers 메서드를 제공하는 자식 모듈은 메시지를 받습니다.
        # 메소드를 가지고 있는지 확인. 만약 해당 메소드를 가지고 있다면, 그 메소드를 호출하여 메모리 효율적인 어텐션를 설정하거나 해제
<<<<<<< HEAD
=======
        
>>>>>>> 7560ecd41fe757fff5908faf1aff8bf71a9818eb
        def fn_recursive_set_mem_eff(module: torch.nn.Module):
            if hasattr(module, "set_use_memory_efficient_attention_xformers"):
                module.set_use_memory_efficient_attention_xformers(valid, attention_op)

            for child in module.children():
                fn_recursive_set_mem_eff(child)

        for module in self.children():
            if isinstance(module, torch.nn.Module):
                fn_recursive_set_mem_eff(module)


<<<<<<< HEAD
=======
## 이 class가 사용된 모든 모델의 child모듈에 효율적인 attention을 적용 
## attention은 딥러닝 모델에서 주어진 입력 시퀀스의 각 요소가 다른 요소들과 어떻게 관련되는지를 나타내는 메커니즘
>>>>>>> 7560ecd41fe757fff5908faf1aff8bf71a9818eb
# Transformer 모델에서 사용되는 게이트가 있는 self-attention 구현
@maybe_allow_in_graph
class GatedSelfAttentionDense(nn.Module):                                           # 시각 특징과 객체 특징을 결합하는 게이트된 셀프 어텐션 덴스 레이어를 정의
  ## 일반적인 어텐션 쿼리, 키, 값 + 게이트라는 개념을 추가 -> 정보의 흐름을 제어?
    r"""
    visual 특징과 object 특징을 결합하는 게이트된 self-attention dense layer입니다.

    매개변수:
        query_dim (`int`): query의 채널 수입니다.
        context_dim (`int`): context의 채널 수입니다.
        n_heads (`int`): attention에 사용할 head의 수입니다.
        d_head (`int`): 각 head의 채널 수입니다.
    """

    def __init__(self, query_dim: int, context_dim: int, n_heads: int, d_head: int):            # Transformer 모델의 초기화 메서드.
        super().__init__()

        # visual feature와 obj feature를 결합하기 위해 linear projection이 필요합니다.
<<<<<<< HEAD
        self.linear = nn.Linear(context_dim, query_dim)                                     # PyTorch의 레이어로, 선형 변환 수행.

        self.attn = Attention(query_dim=query_dim, heads=n_heads, dim_head=d_head)          # Attention
        self.ff = FeedForward(query_dim, activation_fn="geglu")                             # FeedForward

        self.norm1 = nn.LayerNorm(query_dim)                                                # 정규화
        self.norm2 = nn.LayerNorm(query_dim)

        self.register_parameter("alpha_attn", nn.Parameter(torch.tensor(0.0)))              # 게이트 메커니즘에서 사용되는 파라미터
        self.register_parameter("alpha_dense", nn.Parameter(torch.tensor(0.0)))             # 게이트 메커니즘에서 사용되는 파라미터
=======
        self.linear = nn.Linear(context_dim, query_dim)                                     # PyTorch의 레이어로, 객체 차원 -> 쿼리 차원으로 선형변환

        self.attn = Attention(query_dim=query_dim, heads=n_heads, dim_head=d_head)          # 쿼리 차원에 대한 어텐션 메커니즘 구현, 여러 헤드와 헤드당 차원 지정
        self.ff = FeedForward(query_dim, activation_fn="geglu")                             # FeedForward 연산 수행, 지정된 함수 활성화

        self.norm1 = nn.LayerNorm(query_dim)                                                # 정규화 레이어 생성 : 입력을 정규화 하여 레이어 생성
        self.norm2 = nn.LayerNorm(query_dim)                                                # 각 시퀀스마다 차원이 다를 수 있어 이를 정규화

        self.register_parameter("alpha_attn", nn.Parameter(torch.tensor(0.0)))              # 게이트 메커니즘에서 사용되는 파라미터
        self.register_parameter("alpha_dense", nn.Parameter(torch.tensor(0.0)))             # 레이어의 가중치/ FeedForward연산에 사용 -> 모델이 입력 데이터를 효과적으로 사용할 수 있도록
>>>>>>> 7560ecd41fe757fff5908faf1aff8bf71a9818eb

        self.enabled = True                                                                 # 게이트가 활성화되어 있는지 여부를 나타내는 변수

    def forward(self, x: torch.Tensor, objs: torch.Tensor) -> torch.Tensor:
        # 입력으로 주어진 x와 objs를 사용하여 forward 연산을 수행.
        #     매개변수:
        #         x (`torch.Tensor`): 입력 텐서.
        #         objs (`torch.Tensor`): 객체 텐서.
        #     반환값:
        #         `torch.Tensor`: forward 연산의 결과인 텐서.
            
        if not self.enabled:                                                                # 게이트가 비활성화되어 있는 경우, x를 반환.
            return x

<<<<<<< HEAD
        n_visual = x.shape[1]                                                               # x의 shape[1]을 n_visual로 저장.
=======
        n_visual = x.shape[1]                                                               # x의 shape[1]을 n_visual, 차원-시각적 정보갯수 로 저장.
>>>>>>> 7560ecd41fe757fff5908faf1aff8bf71a9818eb
        objs = self.linear(objs)                                                            # objs에 대해 linear projection을 수행.

        x = (
            x
            + self.alpha_attn.tanh()                                                        # 게이트 메커니즘을 적용하여 x에 대한 attention을 계산.
            * self.attn(self.norm1(torch.cat([x, objs], dim=1)))[:, :n_visual, :]           # visual과 obj를 결합하여 attention을 계산.
<<<<<<< HEAD
        )
=======
        )                                                                                   # 어텐션 메커니즘에 대한 연산 수행, x+objs(선형변환된 객체)->어텐션 메커니즘 가중치를 적용해 연산 수행, 결과=>x로
>>>>>>> 7560ecd41fe757fff5908faf1aff8bf71a9818eb
        x = x + self.alpha_dense.tanh() * self.ff(self.norm2(x))                            # 게이트 메커니즘을 적용하여 x에 대한 feed-forward를 계산.

        return x


@maybe_allow_in_graph
class BasicTransformerBlock(nn.Module, MemoryEfficientAttentionMixin):
    r"""
    A basic Transformer block.

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        cross_attention_dim (`int`, *optional*): The size of the encoder_hidden_states vector for cross attention.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        num_embeds_ada_norm (:
            obj: `int`, *optional*): The number of diffusion steps used during training. See `Transformer2DModel`.
        attention_bias (:
            obj: `bool`, *optional*, defaults to `False`): Configure if the attentions should contain a bias parameter.
        only_cross_attention (`bool`, *optional*):
            Whether to use only cross-attention layers. In this case two cross attention layers are used.
        double_self_attention (`bool`, *optional*):
            Whether to use two self-attention layers. In this case no cross attention layers are used.
        upcast_attention (`bool`, *optional*):
            Whether to upcast the attention computation to float32. This is useful for mixed precision training.
        norm_elementwise_affine (`bool`, *optional*, defaults to `True`):
            Whether to use learnable elementwise affine parameters for normalization.
        norm_type (`str`, *optional*, defaults to `"layer_norm"`):
            The normalization layer to use. Can be `"layer_norm"`, `"ada_norm"` or `"ada_norm_zero"`.
        final_dropout (`bool` *optional*, defaults to False):
            Whether to apply a final dropout after the last feed-forward layer.
        attention_type (`str`, *optional*, defaults to `"default"`):
            The type of attention to use. Can be `"default"` or `"gated"` or `"gated-text-image"`.
    """

    def __init__(
        self,
        dim: int,  # 트랜스포머의 인코더와 디코더에서의 정해진 입력과 출력의 크기
        num_attention_heads: int,  # 멀티헤드 어텐션 모델의 헤드 수
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        cond_dim_ada_norm_continuous: Optional[int] = None,
        attention_bias: bool = False,
        only_cross_attention: bool = False,
        double_self_attention: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_type: str = "layer_norm",
        final_dropout: bool = False,
        attention_type: str = "default",
    ):
        # 초기화
        super().__init__() 
        # 크로스 어텐션 층 사용 여부
        self.only_cross_attention = only_cross_attention
        # 확산단계수와 정규화층 유형에 따라 변수값 지정
        self.use_ada_layer_norm_zero = (
            num_embeds_ada_norm is not None
        ) and norm_type == "ada_norm_zero"
        self.use_ada_layer_norm = (
            num_embeds_ada_norm is not None
        ) and norm_type == "ada_norm"
        self.use_ada_layer_norm_continuous = (
            cond_dim_ada_norm_continuous is not None
        ) and norm_type == "ada_norm_continuous"

        assert (
            int(self.use_ada_layer_norm)
            + int(self.use_ada_layer_norm_continuous)
            + int(self.use_ada_layer_norm_zero)
            <= 1
        )

        if norm_type in ("ada_norm", "ada_norm_zero") and num_embeds_ada_norm is None:
            raise ValueError(
                f"`norm_type` is set to {norm_type}, but `num_embeds_ada_norm` is not defined. Please make sure to"
                f" define `num_embeds_ada_norm` if setting `norm_type` to {norm_type}."
            )

        # Define 3 blocks. Each block has its own normalization layer.
        # 1. Self-Attn
        # 위에서 결정된 변수값으로 첫 번째 정규화 층 결정(초기화)
        if self.use_ada_layer_norm:
            self.norm1 = AdaLayerNorm(dim, num_embeds_ada_norm)
        elif self.use_ada_layer_norm_continuous:
            self.norm1 = AdaLayerNormContinuous(dim, cond_dim_ada_norm_continuous)
        elif self.use_ada_layer_norm_zero:
            self.norm1 = AdaLayerNormZero(dim, num_embeds_ada_norm)
        else:
            self.norm1 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim if only_cross_attention else None,
            upcast_attention=upcast_attention,
        )

        # 2. Cross-Attn
        if cross_attention_dim is not None or double_self_attention:    # 일때만 크로스어텐션을 초기화
            # We currently only use AdaLayerNormZero for self attention where there will only be one attention block.
            # I.e. the number of returned modulation chunks from AdaLayerZero would not make sense if returned during
            # the second cross attention block.
            if self.use_ada_layer_norm:
                self.norm2 = AdaLayerNorm(dim, num_embeds_ada_norm)
            elif self.use_ada_layer_norm_continuous:
                self.norm2 = AdaLayerNormContinuous(dim, cond_dim_ada_norm_continuous)
            else:
                self.norm2 = nn.LayerNorm(
                    dim, elementwise_affine=norm_elementwise_affine
                )

            self.attn2 = Attention(
                query_dim=dim,
                cross_attention_dim=cross_attention_dim
                if not double_self_attention
                else None,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
            )  # is self-attn if encoder_hidden_states is none
        else:
            self.norm2 = None
            self.attn2 = None

        # 3. Feed-forward
        if self.use_ada_layer_norm_continuous:
            self.norm3 = AdaLayerNormContinuous(dim, cond_dim_ada_norm_continuous)
        else:
            self.norm3 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)
        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
        )

        # 4. Fuser
        if attention_type == "gated" or attention_type == "gated-text-image":
            self.fuser = GatedSelfAttentionDense(
                dim, cross_attention_dim, num_attention_heads, attention_head_dim
            )

        # let chunk size default to None
        self._chunk_size = None
        self._chunk_dim = 0

    def set_chunk_feed_forward(self, chunk_size: Optional[int], dim: int):
        # Sets chunk feed-forward
        self._chunk_size = chunk_size
        self._chunk_dim = dim

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        modulation_cond: Optional[torch.FloatTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        class_labels: Optional[torch.LongTensor] = None,
    ) -> torch.FloatTensor:
        # Notice that normalization is always applied before the real computation in the following blocks.
        # 0. Self-Attention  norm_hidden_states 정의
        # 셀프 어텐션 은닉 상태에 대한 정규화 선택적 수행
        if self.use_ada_layer_norm:
            norm_hidden_states = self.norm1(hidden_states, timestep)
        elif self.use_ada_layer_norm_continuous:
            norm_hidden_states = self.norm1(hidden_states, modulation_cond)
        elif self.use_ada_layer_norm_zero:
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
            )
        else:
            norm_hidden_states = self.norm1(hidden_states)

        # 1. Retrieve lora scale.     #  scale 하기
        # lora_scale 초기화
        lora_scale = (
            cross_attention_kwargs.get("scale", 1.0)
            if cross_attention_kwargs is not None
            else 1.0
        )

        # 2. Prepare GLIGEN inputs
        cross_attention_kwargs = (
            cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}
        )
        gligen_kwargs = cross_attention_kwargs.pop("gligen", None)

        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states
            if self.only_cross_attention
            else None,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )
        if self.use_ada_layer_norm_zero:
            attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = attn_output + hidden_states

        # 2.5 GLIGEN Control
        # GLIGEN 처리(둘을 결합)
        if gligen_kwargs is not None:
            hidden_states = self.fuser(hidden_states, gligen_kwargs["objs"])
        # 2.5 ends

        # 3. Cross-Attention
        if self.attn2 is not None:
            # 정규화 수행
            if self.use_ada_layer_norm:
                norm_hidden_states = self.norm2(hidden_states, timestep)
            elif self.use_ada_layer_norm_continuous:
                norm_hidden_states = self.norm2(hidden_states, modulation_cond)
            else:
                norm_hidden_states = self.norm2(hidden_states)
            
            # 크로스 어텐션 계산
            attn_output = self.attn2(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                **cross_attention_kwargs,
            )
            hidden_states = attn_output + hidden_states

        # 4. Feed-forward
        if self.use_ada_layer_norm_continuous:
            norm_hidden_states = self.norm3(hidden_states, modulation_cond)
        else:    
            norm_hidden_states = self.norm3(hidden_states)

        if self.use_ada_layer_norm_zero:
            norm_hidden_states = (
                norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
            )
        # Feed-Forward 계층 적용    _chunk_size가 지정되어 있다면, 메모리 절약을 위해 입력 텐서를 여러 청크로 나누어 계산합니다. 그렇지 않다면 전체 입력 텐서에 대해 한 번에 계산합니다.
        if self._chunk_size is not None:
            # "feed_forward_chunk_size" can be used to save memory
            if norm_hidden_states.shape[self._chunk_dim] % self._chunk_size != 0:
                raise ValueError(
                    f"`hidden_states` dimension to be chunked: {norm_hidden_states.shape[self._chunk_dim]} has to be divisible by chunk size: {self._chunk_size}. Make sure to set an appropriate `chunk_size` when calling `unet.enable_forward_chunking`."
                )

            num_chunks = norm_hidden_states.shape[self._chunk_dim] // self._chunk_size
            ff_output = torch.cat(
                [
                    self.ff(hid_slice, scale=lora_scale)
                    for hid_slice in norm_hidden_states.chunk(
                        num_chunks, dim=self._chunk_dim
                    )
                ],
                dim=self._chunk_dim,
            )
        else:
            ff_output = self.ff(norm_hidden_states, scale=lora_scale)

        if self.use_ada_layer_norm_zero:
            ff_output = gate_mlp.unsqueeze(1) * ff_output

        hidden_states = ff_output + hidden_states

        return hidden_states


class FeedForward(nn.Module):   # 피드 포워드 레이어를 정의. 입력데이터가 선형변환을 통해 새로운 공간에 매핑, 활성화 함수를 통과해 결과 출력
    r"""
    A feed-forward layer.

    Parameters:
        dim (`int`): The number of channels in the input. ## 입력의 채널 수
        dim_out (`int`, *optional*): The number of channels in the output. If not given, defaults to `dim`. ## 출력의 채널 수
        mult (`int`, *optional*, defaults to 4): The multiplier to use for the hidden dimension. ## hidden layer의 크기 결정
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use. ## 
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward. ## 활성화 함수 선택, 기본값:geglu
        final_dropout (`bool` *optional*, defaults to False): Apply a final dropout. ## dropout 결정 여부
    """

    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: int = 4,
        dropout: float = 0.0,
        activation_fn: str = "geglu",
        final_dropout: bool = False,
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim
        linear_cls = nn.Linear
        
        ## acivation_fn 을 사용해 활성화 함수를 선택할 수 있도록
        if activation_fn == "gelu":
            act_fn = GELU(dim, inner_dim)
        if activation_fn == "gelu-approximate":
            act_fn = GELU(dim, inner_dim, approximate="tanh")
        elif activation_fn == "geglu":
            act_fn = GEGLU(dim, inner_dim)
        elif activation_fn == "geglu-approximate":
            act_fn = ApproximateGELU(dim, inner_dim)

        self.net = nn.ModuleList([])
        # project in
        self.net.append(act_fn)
        # project dropout
        self.net.append(nn.Dropout(dropout))
        # project out
        self.net.append(linear_cls(inner_dim, dim_out))
        # FF as used in Vision Transformer, MLP-Mixer, etc. have a final dropout
        if final_dropout:
            self.net.append(nn.Dropout(dropout))

    def forward(self, hidden_states: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


class GELU(nn.Module):  ## 비선형 활성화 함수 : 입력값을 반환해 다음 층으로 전달
    r"""
    GELU activation function with tanh approximation support with `approximate="tanh"`.
    ## `appimate="tanh"를 사용하여 tanh 근사를 지원하는 GELU 활성화 함수

    Parameters:
        dim_in (`int`): The number of channels in the input.
        dim_out (`int`): The number of channels in the output.
        approximate (`str`, *optional*, defaults to `"none"`): If `"tanh"`, use tanh approximation.
    """

    def __init__(self, dim_in: int, dim_out: int, approximate: str = "none"):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out) ## 선형 변환 레이어
        self.approximate = approximate  ## 기본값 = none/ tanh를 사용시 tanh근사값 사용
                                        ## GELU 함수는 신경망에서 활성화 함수로 사용, 입력값에 비선형을 추가, 근사화를 통해 메모리 절약 but 정확도가 떨어질 수 있음
    def gelu(self, gate: torch.Tensor) -> torch.Tensor:
        if gate.device.type != "mps":   ## 현재 device type이 mps가 아니라면 
            return F.gelu(gate, approximate=self.approximate)   ## pytorch의 F.gelu함수 호출
        # mps: gelu is not implemented for float16
        return F.gelu(gate.to(dtype=torch.float32), approximate=self.approximate).to(   
            dtype=gate.dtype    ## 현재 device type이 mps이면 float32로 변환 후 GELU함수 적용
        )

    def forward(self, hidden_states):
        hidden_states = self.proj(hidden_states)
        hidden_states = self.gelu(hidden_states)
        return hidden_states
        ## hidden_states를 입력으로 받아 선형으로 변환  

class GEGLU(nn.Module): ## gated linear unit : 입력을 두 부분으로 나눠서 한쪽엔 sigmoid활성화 함수를 통해 0, 1사이의 값으로 변환 -> 원래의 입력값과 곱한다
                        ## 이 과정으로 중요한 정보를 강조하거나 억제 / CNN등의 모델에서 입력과 출력의 차원을 맞추기 위해 ?????
    r"""                
    A variant of the gated linear unit activation function from https://arxiv.org/abs/2002.05202.

    Parameters:
        dim_in (`int`): The number of channels in the input.
        dim_out (`int`): The number of channels in the output.
    """

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        linear_cls = nn.Linear  ## 입려과 출력의 차원을 갖는 선형변환모듈생성

        self.proj = linear_cls(dim_in, dim_out * 2)

    ## gate에 대해 GELU 활성화함수 적용(입력gate : GELU함수에 전달되는 입력, 신경망의 이전 레이어에서 출력되는 값)
    def gelu(self, gate: torch.Tensor) -> torch.Tensor: ## 입력값 * 표준정규분포(입력값)
        if gate.device.type != "mps":
            return F.gelu(gate)
        # mps: gelu is not implemented for float16
        return F.gelu(gate.to(dtype=torch.float32)).to(dtype=gate.dtype)

    def forward(self, hidden_states, scale: float = 1.0):
        args = ()
        hidden_states, gate = self.proj(hidden_states, *args).chunk(2, dim=-1)
        return hidden_states * self.gelu(gate)


class ApproximateGELU(nn.Module):   # 활성화 함수를 정의
    r"""
    The approximate form of Gaussian Error Linear Unit (GELU). For more details, see section 2:
    https://arxiv.org/abs/1606.08415.

    Parameters:
        dim_in (`int`): The number of channels in the input.
        dim_out (`int`): The number of channels in the output.
    """

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return x * torch.sigmoid(1.702 * x)


class AdaLayerNorm(nn.Module):   # 시간 단계 임베딩을 포함하는 정규화 레이어를 정의
    r"""
    Norm layer modified to incorporate timestep embeddings.

    Parameters:
        embedding_dim (`int`): 임베딩 할 벡터의 차원 (하이퍼파라미터)
        num_embeddings (`int`): 임베딩을 할 단어들의 개수 (단어 집합의 크기)
    """

    def __init__(self, embedding_dim: int, num_embeddings: int):
        super().__init__()
        # 임베딩 층 생성 -> lookup table 생성
        self.emb = nn.Embedding(num_embeddings, embedding_dim)
        # activation function으로 silu함수 사용 -> Sigmoid linear unit
        # https://sanghyu.tistory.com/182
        self.silu = nn.SiLU()
        # 선형 변환 모델
        # 출력 텐서가 입력 텐서 크기의 두배 -> 하나는 scale, 하나는 shift
        self.linear = nn.Linear(embedding_dim, embedding_dim * 2)
        # 임베딩 테이블의 Feature에 대한 정규화
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False)

    def forward(self, x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        # timestep은 hidden state와 현재 입력을 연결한 정보
        # timestep을 임베딩하고 activation함수를 적용한 뒤 선형모델에 넣는다
        emb = self.linear(self.silu(self.emb(timestep)))
        # emb 반으로 나누어 scale과 shift에 넣는다
        scale, shift = torch.chunk(emb, 2, dim=1)
        # scale과 shift를 unsqueeze하여 차원을 늘려야 x normalization값과 브로드캐스팅 할 수 있다
        # 정규화된 x값에 scale을 곱하고 shift를 더해 x값을 업데이트해준다.
        x = self.norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return x


class AdaLayerNormContinuous(nn.Module):
    r"""
    Norm layer modified to incorporate arbitrary continuous embeddings.

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
    """

    def __init__(self, embedding_dim: int, condition_dim: int):

        # condition_dim은 조건부 차원으로 긴 시퀀스의 데이터를 다룰 때 조건부로 세분화할 수 있다
        super().__init__()
        self.silu = nn.SiLU()
        self.linear1 = nn.Linear(condition_dim, condition_dim)
        self.linear2 = nn.Linear(condition_dim, embedding_dim * 2)
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:

        # 조건부 차원의 데이터를 선형모델에 넣고 activation함수를 적용하고 선형모델에 넣어 scale과 shift를 얻는다
        emb = self.linear2(self.silu(self.linear1(condition)))
        scale, shift = torch.chunk(emb, 2, dim=1)
        x = self.norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return x


class Modulation(nn.Module):
    def __init__(self, embedding_dim: int, condition_dim: int, zero_init: bool = False, single_layer: bool = False):

        # 출력 텐서를 입력 조건 텐서에 맞는 형태로 변조해주는 클레스
        super().__init__()
        self.silu = nn.SiLU()
        # layer가 하나면 그대로 출력
        if single_layer:
            self.linear1 = nn.Identity()
        # 입력 조건 텐서에 맞게 선형 모델 적용
        else:
            self.linear1 = nn.Linear(condition_dim, condition_dim)

        self.linear2 = nn.Linear(condition_dim, embedding_dim * 2)

        # Only zero init the last linear layer
        if zero_init:
            # scale과 shift를 0으로 초기화
            nn.init.zeros_(self.linear2.weight)
            nn.init.zeros_(self.linear2.bias)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        emb = self.linear2(self.silu(self.linear1(condition)))
        scale, shift = torch.chunk(emb, 2, dim=1)
        x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return x


class AdaLayerNormZero(nn.Module):
    r"""
    Norm layer adaptive layer norm zero (adaLN-Zero).
    
    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`): The size of the dictionary of embeddings.
    """

    def __init__(self, embedding_dim: int, num_embeddings: int):
        super().__init__()
        # timestep과 임베딩이 라벨링된 상태의 텐서
        self.emb = CombinedTimestepLabelEmbeddings(num_embeddings, embedding_dim)
        self.silu = nn.SiLU()
        # output이 6개인 선형 모델 추가 dimension 1 -> 6
        self.linear = nn.Linear(embedding_dim, 6 * embedding_dim, bias=True)
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        class_labels: torch.LongTensor,
        hidden_dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        emb = self.linear(
            self.silu(self.emb(timestep, class_labels, hidden_dtype=hidden_dtype))
        )
        # msa -> multi-head self attention
        # mlp -> multi-layer perceptron
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(
            6, dim=1
        )
        # scale_msa[:, None] -> scale_msa.unsqueeze(1)인데 추가한 차원을 None으로 초기화
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


class AdaGroupNorm(nn.Module):
    r"""
    GroupNorm layer modified to incorporate timestep embeddings.

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`): The size of the dictionary of embeddings.
        num_groups (`int`): The number of groups to separate the channels into.
        act_fn (`str`, *optional*, defaults to `None`): The activation function to use.
        eps (`float`, *optional*, defaults to `1e-5`): The epsilon value to use for numerical stability.
    """

    def __init__(
        self,
        embedding_dim: int,
        out_dim: int,
        num_groups: int,
        act_fn: Optional[str] = None,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.num_groups = num_groups
        self.eps = eps

        if act_fn is None:
            self.act = None
        else:
            self.act = get_activation(act_fn)
            
        self.linear = nn.Linear(embedding_dim, out_dim * 2)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        if self.act:
            emb = self.act(emb)
        emb = self.linear(emb)
        emb = emb[:, :, None, None]
        scale, shift = emb.chunk(2, dim=1)
        # self.num_groups만큼 그룹을 나누어 정규화
        x = F.group_norm(x, self.num_groups, eps=self.eps)
        x = x * (1 + scale) + shift
        return x

class Transformer1D(BaseModule, MemoryEfficientAttentionMixin):
    #  1D(1차원 순차 데이터) Transformer 모델을 정의
    #  입력 레이어, Transformer 블록, 출력 레이어로 구성되어 있습니다.
    #  모델 구성 매개변수를 설정할 수 있습니다.

    """
    A 1D Transformer model for sequence data.
    ## 시퀀스 데이터를 위한 1D 변환기 모델
    Parameters:
        num_attention_heads (`int`, *optional*, defaults to 16): The number of heads to use for multi-head attention.
        ## multi-head attention에 사용할 head의 수
        attention_head_dim (`int`, *optional*, defaults to 88): The number of channels in each head.
        ## 각 head의 체널 수
        in_channels (`int`, *optional*):
            The number of channels in the input and output (specify if the input is **continuous**).
        num_layers (`int`, *optional*, defaults to 1): The number of layers of Transformer blocks to use.
        ## 사용할 Transformer 블록의 레이어 수
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        ## 사용할 dropout의 확률
        cross_attention_dim (`int`, *optional*): The number of `encoder_hidden_states` dimensions to use.
        ## encoder_hidden_states 차원의 수
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to use in feed-forward.
        ## feed-forward에 사용할 활성화 함수
        num_embeds_ada_norm ( `int`, *optional*):
            The number of diffusion steps used during training.
            Pass if at least one of the norm_layers is
            `AdaLayerNorm`. This is fixed during training since it is used to learn a number of embeddings that are
            added to the hidden states.
            ## 훈련 중 사용되는 확산단계 수, norm_layers중 하나 이상이 다음과 같으면 통과 adalayernorm은 다수의 임베딩을 학습하는데 사용, 훈련중에 수정 

            During inference, you can denoise for up to but not more steps than `num_embeds_ada_norm`.
            inference 중, `num_embeds_ada_norm`보다 최대 단계까지 노이즈를 제거할 수 있음
        attention_bias (`bool`, *optional*):
            Configure if the `TransformerBlocks` attention should contain a bias parameter.
            `TransformerBlocks` 어텐션에 바이어스 매개변수가 포함되어야 하는지 구성
    """

    @dataclass   # 데이터를 저장하는 클래스 사용
    class Config(BaseModule.Config):   # 매개변수 정의
        num_attention_heads: int = 16  # 멀티헤드 어텐션에서 사용할 어텐션 헤드의 수 지정
        attention_head_dim: int = 88 # 각 어텐션 헤드의 차원을 지정
        in_channels: Optional[int] = None  # 입력 데이터의 채널 수를 지정
        out_channels: Optional[int] = None  # 출력 데이터의 채널 수를 지정
        num_layers: int = 1   # Transformer 블록의 개수를 지정
        dropout: float = 0.0  # 드롭아웃(Dropout) 확률을 지정
        norm_num_groups: int = 32  # GroupNorm 레이어에서 사용할 그룹의 수를 지정
        cross_attention_dim: Optional[int] = None   # 크로스 어텐션에 사용될 인코더 hidden state의 차원을 지정
        attention_bias: bool = False  # 어텐션 레이어에 바이어스(bias) 매개변수를 추가할지 여부
        activation_fn: str = "geglu"  # 피드 포워드 레이어에서 사용할 활성화 함수를 지정
        num_embeds_ada_norm: Optional[int] = None  # 훈련 중 사용된 디노이징 단계 수를 지정 => AdaLayerNorm 레이어에서 사용
        cond_dim_ada_norm_continuous: Optional[int] = None  # AdaLayerNormContinuous 레이어에서 사용될 conditional input의 차원을 지정
        only_cross_attention: bool = False  # 크로스 어텐션 레이어만 사용할지 여부
        double_self_attention: bool = False  # 두 개의 셀프 어텐션 레이어를 사용할지 여부
        upcast_attention: bool = False
        norm_type: str = "layer_norm"  # 용할 정규화 레이어의 유형을 지정
        norm_elementwise_affine: bool = True
        attention_type: str = "default"  # 사용할 어텐션 유형을 지정. 옵션은 "default", "gated", "gated-text-image"
        enable_memory_efficient_attention: bool = False  # 메모리 효율적인 어텐션 기법을 사용할지 여부를 지
        gradient_checkpointing: bool = False  # 그래디언트 체크포인팅을 사용하여 메모리 사용량을 줄일지 여부를 지정


    cfg: Config

    def configure(self) -> None:
        super().configure()   # BaseModule 클래스의 configure 메서드를 호출

        self.num_attention_heads = self.cfg.num_attention_heads   # attention head 개수 설정
        self.attention_head_dim = self.cfg.attention_head_dim     # attention head dim 설정
        inner_dim = self.num_attention_heads * self.attention_head_dim  # inner_dim을 계산

        linear_cls = nn.Linear   # nn.Linear 객체 생성(from torch 라이브러리의 nn 클래스)
                                  # PyTorch의 선형 레이어(Linear Layer)를 정의하는 클래스
        # 선형 레이어는 입력 데이터에 가중치 행렬을 곱하고 편향(bias)을 더하는 간단한 행렬 곱셈 연산을 수행
        # 이는 딥러닝 모델에서 가장 기본적인 레이어 중 하나
        # class nn.Linear(in_features, out_features, bias=True)


        # 정규화 유형이 layer_norm인데 num_embeds_ada_norm이나 cond_dim_ada_norm_continuous가
        # 설정되어 있으면 ValueError

        if self.cfg.norm_type == "layer_norm" and (
            self.cfg.num_embeds_ada_norm is not None
            or self.cfg.cond_dim_ada_norm_continuous is not None
        ):
            raise ValueError("Incorrect norm_type.")

        # 2. Define input layers  # 인풋 레이어 정의
        self.in_channels = self.cfg.in_channels  # 채널 개수 설정

        self.norm = torch.nn.GroupNorm(    # GroupNorm 레이어를 norm에 할당

            num_groups=self.cfg.norm_num_groups,
            num_channels=self.cfg.in_channels,
            eps=1e-6,
            affine=True,
        )
        self.proj_in = linear_cls(self.cfg.in_channels, inner_dim)   # proj_in 레이어를 초기화

        # 3. Define transformers blocks   트랜스포머 블록 정의
        self.transformer_blocks = nn.ModuleList(   # transformer_blocks에 BasicTransformerBlock 저장
            [
                BasicTransformerBlock(
                    inner_dim,
                    self.num_attention_heads,
                    self.attention_head_dim,
                    dropout=self.cfg.dropout,
                    cross_attention_dim=self.cfg.cross_attention_dim,
                    activation_fn=self.cfg.activation_fn,
                    num_embeds_ada_norm=self.cfg.num_embeds_ada_norm,
                    cond_dim_ada_norm_continuous=self.cfg.cond_dim_ada_norm_continuous,
                    attention_bias=self.cfg.attention_bias,
                    only_cross_attention=self.cfg.only_cross_attention,
                    double_self_attention=self.cfg.double_self_attention,
                    upcast_attention=self.cfg.upcast_attention,
                    norm_type=self.cfg.norm_type,
                    norm_elementwise_affine=self.cfg.norm_elementwise_affine,
                    attention_type=self.cfg.attention_type,
                )
                for d in range(self.cfg.num_layers)
            ]
        )

        # 4. Define output layers    # 아웃풋 레이어 정의
        self.out_channels = (    # out_channels를 설정, 지정되지 않았다면 in_channels와 동일하게
            self.cfg.in_channels
            if self.cfg.out_channels is None
            else self.cfg.out_channels
        )

        # proj_out 레이어를 초기화하여 inner_dim에서 in_channels 차원으로 프로젝션합니다.
        self.proj_out = linear_cls(inner_dim, self.cfg.in_channels)  #

        # gradient_checkpointing
        self.gradient_checkpointing = self.cfg.gradient_checkpointing

        # 메모리 효율적 어텐션
        self.set_use_memory_efficient_attention_xformers(
            self.cfg.enable_memory_efficient_attention
        )

    def forward(
        # 입력 텐서를 받아 인코더에 통과시키기 => 출력 생성
        # 이 때 다양한 입력 및 조건부 텐서
        # (encoder_hidden_states, timestep, modulation_cond, class_labels 등) 사용할 수 있음
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,  # 크로스 어텐션(Cross-Attention)에 사용될 인코더의 hidden states, 지정되지 않으면 셀프 어텐션(Self-Attention)이 사용됨
        timestep: Optional[torch.LongTensor] = None,    # AdaLayerNorm에서 임베딩으로 사용
        modulation_cond: Optional[torch.FloatTensor] = None,
        class_labels: Optional[torch.LongTensor] = None,  # AdaLayerZeroNorm에서 임베딩으로 사용
        cross_attention_kwargs: Dict[str, Any] = None,  # 셀프 어텐션에 적용될 마스크 텐서
        attention_mask: Optional[torch.Tensor] = None,  # 크로스 어텐션에 적용될 마스크 텐서
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ):
        """
        The [`Transformer1DModel`] forward method.

        Args:
            hidden_states (`torch.LongTensor` of shape `(batch size, num latent pixels)` if discrete, `torch.FloatTensor` of shape `(batch size, channel, height, width)` if continuous):
                Input `hidden_states`.
            encoder_hidden_states ( `torch.FloatTensor` of shape `(batch size, sequence len, embed dims)`, *optional*):
                Conditional embeddings for cross attention layer. If not given, cross-attention defaults to
                self-attention.
            timestep ( `torch.LongTensor`, *optional*):
                Used to indicate denoising step. Optional timestep to be applied as an embedding in `AdaLayerNorm`.
            class_labels ( `torch.LongTensor` of shape `(batch size, num classes)`, *optional*):
                Used to indicate class labels conditioning. Optional class labels to be applied as an embedding in
                `AdaLayerZeroNorm`.
            cross_attention_kwargs ( `Dict[str, Any]`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            attention_mask ( `torch.Tensor`, *optional*):
                An attention mask of shape `(batch, key_tokens)` is applied to `encoder_hidden_states`. If `1` the mask
                is kept, otherwise if `0` it is discarded. Mask will be converted into a bias, which adds large
                negative values to the attention scores corresponding to "discard" tokens.
            encoder_attention_mask ( `torch.Tensor`, *optional*):
                Cross-attention mask applied to `encoder_hidden_states`. Two formats supported:

                    * Mask `(batch, sequence_length)` True = keep, False = discard.
                    * Bias `(batch, 1, sequence_length)` 0 = keep, -10000 = discard.

                If `ndim == 2`: will be interpreted as a mask, then converted into a bias consistent with the format
                above. This bias will be added to the cross-attention scores.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unet_2d_condition.UNet2DConditionOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        # ensure attention_mask is a bias, and give it a singleton query_tokens dimension.
        #   we may have done this conversion already, e.g. if we came here via UNet2DConditionModel#forward.
        #   we can tell by counting dims; if ndim == 2: it's a mask rather than a bias.
        # expects mask of shape:
        #   [batch, key_tokens]
        # adds singleton query_tokens dimension:
        #   [batch,                    1, key_tokens]
        # this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
        #   [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
        #   [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)
        if attention_mask is not None and attention_mask.ndim == 2:
            # assume that mask is expressed as:
            #   (1 = keep,      0 = discard)
            # convert mask into a bias that can be added to attention scores:
            #       (keep = +0,     discard = -10000.0)
            attention_mask = (1 - attention_mask.to(hidden_states.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # attention_mask와 encoder_attention_mask가 2차원 텐서인 경우,
        # 차원을 확장하고 마스크 값을 바이어스로 변환
        # 마스크 값을 바이어스(bias) 값으로 변환
        # attention_mask의 값이 0이면 -10000.0으로, 1이면 0으로 변환
        # 그리고 새로운 차원을 추가하여 브로드캐스팅을 용이

        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            encoder_attention_mask = (
                1 - encoder_attention_mask.to(hidden_states.dtype)
            ) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        # 입력 레이어
        # 1. Input
        batch, _, seq_len = hidden_states.shape  # hidden_states의 배치 크기와 시퀀스 길이를 추출
        residual = hidden_states  # 잔차 연결

        hidden_states = self.norm(hidden_states)  # hidden_states를 정규화하고
        inner_dim = hidden_states.shape[1]   # 채널 수 할당, hidden_states 텐서의 채널 수(channel dimension)를 inner_dim 변수에 할당
        hidden_states = hidden_states.permute(0, 2, 1).reshape(   # 차원을 변경하여
            batch, seq_len, inner_dim
        )
        hidden_states = self.proj_in(hidden_states)
        # Transformer 블록의 입력 차원에 맞게
        # proj_in 레이어를 통과시키기.

        # 2. Blocks
        # transformer_blocks에 있는 각 BasicTransformerBlock을 순차적으로 거친다
        for block in self.transformer_blocks:
            # 훈련 중이고 gradient_checkpointing이 활성화되어 있다면,
            # 그래디언트 체크포인팅을 사용하여 메모리 사용량을 줄임
            if self.training and self.gradient_checkpointing:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    block,
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    timestep,
                    modulation_cond,
                    cross_attention_kwargs,
                    class_labels,
                    use_reentrant=False,
                )
            else:  # 그렇지 않다면 일반적으로 BasicTransformerBlock을 통과
                hidden_states = block(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    timestep=timestep,
                    modulation_cond=modulation_cond,
                    cross_attention_kwargs=cross_attention_kwargs,
                    class_labels=class_labels,
                )

        # 3. Output
        # Transformer 블록의 출력 차원을 원하는 출력 차원으로
        hidden_states = self.proj_out(hidden_states)   # hidden_states를 proj_out 레이어를 통과시켜,
        hidden_states = (                              # 원래 차원으로 변환.
            hidden_states.reshape(batch, seq_len, inner_dim)
            .permute(0, 2, 1)
            .contiguous()
        )

        output = hidden_states + residual   # 잔차 연결을 적용하여 최종 출력 output을 계산

        return output   # output을 반환