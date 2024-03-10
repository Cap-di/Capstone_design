# MJU 2024 Capstone Design

## Members

- **지도교수 - Hayoung Byun, Ph.D.**
  - 조장 - 유광열
    - 총무 - 김대형
    - 조원 - 조시현, [차성철](https://github.com/SungChul-CHA), 허연후

---

## Brainstorming

- 주제 : **Vision AI + 임베디드 + DPU**

---

|                  아이디어                  | 선호도  |
| :----------------------------------------: | :-----: |
|              자율 주행 자동차              |    O    |
|              쓰레기 청소 로봇              |    O    |
|               과일 수확 로봇               |    O    |
| 패션 스타일 추천 + 퍼스널 컬러 + 옷 입히기 |    O    |
|          반응 해주는 로봇 같은거           | &#9651; |
|                관상 판독기                 | &#9651; |
|          사진의 변조 수준을 검사           | &#9651; |
|            자세 교정(척추 + 목)            | &#9651; |
|              주식 그래프 예상              |    X    |
|          음식 사진 보고 위치 찾기          |    X    |
|                 키 맞추기                  |    X    |
|   건물 외관을 보고 구조 파악, 결함 확인    |    X    |

### Additional Goal

- 음성 인식을 통한 행동 지시

---

## WORKFLOW

### repository 구성 및 branch rule

- [CD repository](https://github.com/Cap-di/Capstone_design)의 main 으로 부터 branch는 각자 한 개 씩 총 5개의 추가 branch 로 작업. 필요할 경우 1~2개 추가

- local repository 에서 [remote repository](https://github.com/Cap-di/Capstone_design)의 각자 branch로 `push`, `pull` 가능

- remote repository의 `main` branch는 `pull` 만 가능 (branch rule 적용)

- remote repository의 `main` branch는 `merge` 하기 위해서 PR 필수 (branch rule 적용)

### commit rule

- 코드를 작성한 목적에 맞게 기능 별로 커밋 작성하기

- **무슨 기능을 하는지**, **왜 만들었는지**, **뭘 바궜는지**, 언제 코드 작성했는지, 어디서 작성했는지, 참고해야할 코드가 무엇인지, 누가 작성했는지
