## 전처리 프로그램
### 교통약자 주행 영상 데이터 학습 모델을 학습하기위한 전처리 프로그램

## 클래스 아이디 준비

- 학습시킬 클래스 아이디 준비
  - 클래스 아이디와 학습시킬 클래스가 들어있는 텍스트 파일
  - EX) class_id.txt
  ```bash
    0,flatness_A
    1,flatness_B
    2,flatness_C
    3,flatness_D
    4,flatness_E
    5,walkway_paved
    6,walkway_block
    7,paved_state_broken
    8,paved_state_normal
    9,block_state_broken
    10,block_state_normal
    11,block_kind_bad
    12,block_kind_good
    13,outcurb_rectangle
    14,outcurb_slide
    15,outcurb_rectangle_broken
    16,restspace
    17,sidegap_in
    18,sidegap_out
    19,sewer_cross
    20,sewer_line
    21,brailleblock_dot
    22,brailleblock_line
    23,brailleblock_dot_broken
    24,brailleblock_line_broken
    25,continuity_tree
    26,continuity_manhole
    27,ramp_yes
    28,ramp_no
    29,bicycleroad_broken
    30,bicycleroad_normal
    31,planecrosswalk_broken
    32,planecrosswalk_normal
    33,steepramp
    34,bump_slow
    35,weed
    36,floor_normal
    37,flowerbed
    38,parkspace
    39,tierbump
    40,stone
    41,enterrail
    42,stair_normal
    43,stair_broken
    44,wall
    45,window_sliding
    46,window_casement
    47,pillar
    48,lift
    49,door_normal
    50,lift_door
    51,resting_place_roof
    52,reception_desk
    53,protect_wall_protective
    54,protect_wall_guardrail
    55,protect_wall_kickplate
    56,handle_vertical
    57,handle_lever
    58,handle_circular
    59,lift_button_normal
    60,lift_button_openarea
    61,lift_button_layer
    62,lift_button_emergency
    63,direction_sign_left
    64,direction_sign_right
    65,direction_sign_straight
    66,direction_sign_exit
    67,sign_disabled_toilet
    68,sign_disabled_parking
    69,sign_disabled_elevator
    70,sign_disabled_callbell
    71,sign_disabled_icon
    72,braille_sign
    73,chair_multi
    74,chair_one
    75,chair_circular
    76,chair_back
    77,chair_handle
    78,number_ticker_machine
    79,beverage_vending_machine
    80,beverage_desk
    81,trash_can
    82,mailbox
  ```

## 학습모델(yolov5)을 학습시키기 위한 txt 파일 생성

- 프로그램 내에 yolov5 인풋에 맞는 txt를 생성하기위한 함수 존재
  - 라벨링된 데이터의 segmentaion points(rectangle, polygon)를 yolov5 인풋에 맞게 변환
    ```
    "segmentation": [
        [
          200.7246376811594,
          58.02898550724635
        ],
        [
          370.28985507246375,
          160.927536231884
        ]
    ```
    ```
    26 0.46806 0.469219 0.096425 0.062563
    ```
  - 변환된 txt 파일은 이미지 파일이 존재하는곳에 이미지 파일마다 생성  
  
## 프로그램 사용법
  - 프로그램 사용 예시
  ```
  $ python preprocessing_AI.py -p image_folder -n 1
  ```
## 프로그램 Arguments
  - (-p) 이미지가 들어있는 폴더
    - 모든 이미지 및 json 파일 들어가있는 폴더
    ```
    -p C:\Desktop\test\image\one
    ```  
  - (-n) 프로세스 처리
    - 파일이름 에러체크
    - 파일존재 에러체크
    - 에러난 파일 이동 리스트 생성 및 이동
    - 블러처리한 파일 이름 체크
    - 블러처리한 0P 파일이 있을 때 00 파일도 같이 있을경우 파일 이동
    - 라벨링된 데이터 segmentation points가 마이너스 값이 있을경우 수정
    - json 포맷 에러난 경우 json 포맷 수정
    - 에러난 파일들 리스트 생성 및 폴더 생성하여 파일 이동(실행한 path내에 error 폴더가 생성되고 폴더안 에러 체크된 이미지 및 json 파일 이동)
    - 구버전 json 포맷 있을경우 체크 및 파일 이동
