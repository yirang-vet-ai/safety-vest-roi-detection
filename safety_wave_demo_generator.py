# Modified for GitHub packaging by YIRANG JUNG
# Changes: default output path updated for repository structure.

import cv2
import numpy as np

# -----------------------------
# 설정값
# -----------------------------
width, height = 640, 480
fps = 20
duration_sec = 5  # 총 영상 길이
total_frames = fps * duration_sec

output_path = "demo/safety_wave_demo.mp4"

# ROI (관심 영역)
roi_x1, roi_y1 = 200, 150
roi_x2, roi_y2 = 440, 350

# 비디오 writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# -----------------------------
# 프레임 생성
# -----------------------------
for frame_idx in range(total_frames):
    frame = np.zeros((height, width, 3), dtype=np.uint8)

    # 배경 (건설현장 느낌: 회색)
    frame[:] = (50, 50, 50)

    # ROI 영역 표시
    cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 255), 2)
    cv2.putText(frame, "ROI", (roi_x1, roi_y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

    # -----------------------------
    # 작업자 위치 이동 (ROI로 진입)
    # -----------------------------
    person_x = int(50 + frame_idx * 2)  # 오른쪽으로 이동
    person_y = 250

    # 몸통 (사각형)
    cv2.rectangle(frame,
                  (person_x, person_y),
                  (person_x + 40, person_y + 80),
                  (255, 200, 0), -1)

    # 머리
    cv2.circle(frame,
               (person_x + 20, person_y - 20),
               15,
               (255, 220, 180), -1)

    # -----------------------------
    # 손 흔들기 (좌우 진동)
    # -----------------------------
    wave_offset = int(20 * np.sin(frame_idx * 0.3))

    hand_x = person_x + 40 + wave_offset
    hand_y = person_y + 20

    cv2.circle(frame, (hand_x, hand_y), 8, (0, 0, 255), -1)

    # 팔 (라인)
    cv2.line(frame,
             (person_x + 40, person_y + 20),
             (hand_x, hand_y),
             (0, 0, 255), 3)

    # -----------------------------
    # 상태 텍스트
    # -----------------------------
    if roi_x1 < person_x < roi_x2:
        cv2.putText(frame, "Worker in ROI",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0,255,0), 2)

    cv2.putText(frame, "Hand Waving Signal",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0,0,255), 2)

    # 저장
    out.write(frame)

# 종료
out.release()

print("영상 생성 완료:", output_path)