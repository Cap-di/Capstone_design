## 3-2주차 회의록

### 목차

- [서비스 내용 구체화](#서비스-내용-구체화)
- [구현 방식](#ai-구현-방식)
- [필요한 데이터 정하고 어떻게 구할지]()
- [일정]()
- [역할 분담]()

---

### 서비스 내용 구체화

- 얼굴, 헤어스타일은 얼굴인식 하듯이 틀을 띄우고 그걸로 아바타의 얼굴로 만듦 &rarr; AI

- 정량 데이터

  1. 키, 몸무게, 허리둘레(=바지 사이즈), 가슴둘레(=상의 사이즈) + 상체,하체는 슬라이더로 비율 조정해서 다시 3D 아바타 생성
  2. InBody 데이터, BMI -> 데이터 찾을 때 고민

  - (선택사항) 발길이(=신발사이즈)

- 3D 모델 생성 &rarr; GLTF/FBX 형태의 파일일 가능성이 큼 &rarr; 파일을 읽어서 웹사이트에 뿌림

- (선택사항) 앱으로 서비스

---

### AI 구현 방식

- 3D

  - 얼굴, 옷

  2D to 3D AI 모델을 만들고 파인튜닝 해서 얼굴 to 3D Avatar, 2D 옷 to 3D 옷 적용

  - 몸

    몸 수치를 주면 3D로 만드는 모델

- 구현 방식

모델링 : GAN 돌리고 우리가 더 나은 모델을 선택

만든 2개의 3D 데이터를 잘 조합해서 아바타에 옷을 입히기

### 데이터 수집 방법

- 무신사 같은 쇼핑몰 크롤링을 통해 광고 사진을 가져오고, 그 데이터를 분류 모델을 통해 전면 사진, 옆면사진, 뒷면사진을 분류해서 짤라줌

### 일정

### 역할

### 웹사이트

- [3d esset](https://www.turbosquid.com/ko/Search/3D-Models/free/asset)
- [hugging face : Image-to-3D](https://huggingface.co/models?pipeline_tag=image-to-3d&sort=trending)
- [AIVAR](http://myfiit.ai/service/)
- [FASHION-HOW](https://fashion-how.org/ETRI23/)
- [3D viewer](https://3dviewer.net/)
