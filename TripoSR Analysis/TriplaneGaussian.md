Self-Attention: 입력 시퀀스의 각 토큰을 다른 토큰과 비교하여 중요도를 계산합니다.
Cross-Attention: 입력 시퀀스와 다른 시퀀스(예: 이미지 캡션 생성에서 이미지 피처) 간의 관계를 모델링합니다.
Feed-Forward Network: 각 토큰에 대해 간단한 피드포워드 신경망을 적용하여 표현력을 높입니다.


포인트 클라우드 기반 방법은 포인트 클라우드의 밀도가 낮으면 정확도가 떨어진다.

포인트 디코더는 단일 이미지에서 포인트 클라우드를 생성하고, 트라이플레이트 디코더는 포인트 클라우드를 트라이플레이트로 분할하고, 각 트라이플레이트를 가우시안 특징으로 표현한다. 이를 통해 3D 객체를 더욱 정확하게 재구성할 수 있다

포인트 클라우드: 3D 객체를 점들의 집합으로 표현하는 방법이다. 각 점은 3D 공간에서의 위치와 색상 등의 정보를 가지고 있다.

메쉬: 3D 객체를 삼각형들의 집합으로 표현하는 방법이다. 각 삼각형은 3D 공간에서의 위치와 방향 등의 정보를 가지고 있다.

레이 트레이싱(Ray Tracing): 3D 객체를 광선들의 반사와 굴절을 이용하여 렌더링하는 방법이다. 레이 트레이싱은 광선이 3D 객체와 만나는 지점을 계산하고, 해당 지점에서의 색상과 밝기를 계산한다.

제안하는 방법은 feed-forward 추론을 사용한다. feed-forward 추론은 네트워크의 입력과 출력 사이의 관계를 미리 학습하고, 이를 기반으로 입력에 대한 출력을 예측

3D gaussian reconstruction model 만듦 -> based on a transformer architecture 3D gaussian을 예측하는
point cloud -> geometry
triplane -> encodes an implicit feature field, where 3D Gaussian attributes can be decoded
3d gaussian splatting -> 3d rendering

DINOv2 : 저자들은 self-supervised 사전 학습만으로도 공개적으로 사용 가능한 weakly-supervised 모델과 경쟁할 수 있는 transfer 가능한 고정 feature을 학습하기에 좋은 후보
to obtain patch-wise feature tokens from the input image

we implement an adaptive
layer norm (adaLN) to modulate the DINOv2 features with
the camera features, which is similar to [20, 62]. Specifi
cally, we project the camera parameters, which are the con
catenation of flattened camera extrinsic matrix T ∈ R4×4
and normalized intrinsic matrix K ∈ R3×3, to high dimen
sional camera features fc ∈ R25 aligned with the DINOv2
feature dimension.

we use a set
of feature tokens for the latent features of
two different 3D representations, i.e., points and triplane,
respectively
transformer block comprises a self-attention layer, a cross
attention layer, and a feed-forward layer. The viewpoint
augmented image tokens guide the two decoders via cross
attention, respectively.

The point cloud decoder provides
the coarse geometry of the object, where 3D Gaussians can
be produced based on the coordinates of the points.
시간 걸려서 2048까지만

[3d dataset](https://objaverse.allenai.org/)
[google dataset](https://blog.research.google/2022/06/scanned-objects-by-google-research.html)

Snowflake point deconvolution : densify the point clouds from 2048 points to 16384 points

[3D gaussian](https://xoft.tistory.com/49)
[3D gaussian splatting](https://xoft.tistory.com/51)
[Point Cloud Diffusion for Single-Image 3D Reconstruction](https://arxiv.org/abs/2302.10668)
