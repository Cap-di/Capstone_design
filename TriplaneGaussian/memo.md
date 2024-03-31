BasicTransformerBlock 클래스

Self-Attention: 입력 시퀀스의 각 토큰을 다른 토큰과 비교하여 중요도를 계산합니다.
Cross-Attention: 입력 시퀀스와 다른 시퀀스(예: 이미지 캡션 생성에서 이미지 피처) 간의 관계를 모델링합니다.
Feed-Forward Network: 각 토큰에 대해 간단한 피드포워드 신경망을 적용하여 표현력을 높입니다.

매개변수:
dim (int): 입력과 출력의 채널 수입니다.
num_attention_heads (int): 멀티헤드 어텐션에 사용할 헤드 수입니다.
attention_head_dim (int): 각 어텐션 헤드의 채널 수입니다.
dropout (float, 선택사항, 기본값 0.0): 사용할 드롭아웃 확률입니다.
cross_attention_dim (int, 선택사항): 크로스 어텐션을 위한 인코더 은닉 상태 벡터의 크기입니다.
activation_fn (str, 선택사항, 기본값 "geglu"): 피드포워드에 사용할 활성화 함수입니다.
num_embeds_ada_norm (int, 선택사항): 학습 중 사용된 확산 단계 수입니다. Transformer2DModel을 참조하세요.
attention_bias (bool, 선택사항, 기본값 False): 어텐션에 바이어스 파라미터를 포함할지 여부를 설정합니다.
only_cross_attention (bool, 선택사항): 크로스 어텐션 층만 사용할지 여부입니다. 이 경우 두 개의 크로스 어텐션 층이 사용됩니다.
double_self_attention (bool, 선택사항): 두 개의 셀프 어텐션 층을 사용할지 여부입니다. 이 경우 크로스 어텐션 층은 사용되지 않습니다.
upcast_attention (bool, 선택사항): 어텐션 계산을 float32로 업캐스팅할지 여부입니다. 혼합 정밀도 학습에 유용합니다.
norm_elementwise_affine (bool, 선택사항, 기본값 True): 정규화를 위해 가중치와 편향을 사용할지 여부입니다.
norm_type (str, 선택사항, 기본값 "layer_norm"): 사용할 정규화 층의 유형입니다. "layer_norm", "ada_norm" 또는 "ada_norm_zero" 중 하나입니다.
final_dropout (bool, 선택사항, 기본값 False): 마지막 피드포워드 층 이후에 드롭아웃을 적용할지 여부입니다.
attention_type (str, 선택사항, 기본값 "default"): 사용할 어텐션 유형입니다. "default", "gated" 또는 "gated-text-image" 중 하나입니다.

306 - Ada 계층 정규화 기법 사용


이 코드는 Transformer 모델의 기본 블록인 BasicTransformerBlock 클래스를 정의하고 있습니다. 주요 기능은 다음과 같습니다:

셀프 어텐션(Self-Attention) 계층 처리
크로스 어텐션(Cross-Attention) 계층 처리
피드포워드(Feed-Forward) 계층 처리
GLIGEN(Gated Linear Cross-Attention) 처리
각 계층 전에 은닉 상태(hidden states)에 대한 정규화(normalization)를 선택적으로 수행할 수 있습니다. 정규화 방식으로 일반적인 Layer Normalization 외에도 Ada Layer Normalization의 변종들(AdaLayerNorm, AdaLayerNormContinuous, AdaLayerNormZero)을 사용할 수 있습니다.

또한 메모리 효율성을 위해 입력 텐서를 여러 청크(chunk)로 나누어 계산하는 기능도 지원합니다.

이 BasicTransformerBlock 클래스는 Transformer 기반 모델(예: 이미지 생성 모델)에서 사용될 수 있으며, 다양한 옵션을 통해 모델의 구조와 동작을 제어할 수 있습니다.

전반적으로 이 코드는 Transformer 모델의 핵심 계층들을 구현하고 있으며, 효율적인 계산과 다양한 정규화 기법을 지원하는 것으로 보입니다.