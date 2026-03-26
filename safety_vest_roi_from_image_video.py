# Modified for GitHub packaging by YIRANG JUNG
# Changes: default paths updated for repository structure and output folder usage.

import cv2
import numpy as np
import os

# =========================================================
# 1. 설정값
# =========================================================

VIDEO_PATH = "demo/safety_wave_demo.mp4"
OUTPUT_PATH = "outputs/result_safety_wave_demo_panel.mp4"

# 관심영역(Region of Interest, ROI) 비율
ROI_X1_RATIO = 0.30
ROI_Y1_RATIO = 0.20
ROI_X2_RATIO = 0.80
ROI_Y2_RATIO = 0.90

# 안전복 색상 범위 (HSV: Hue, Saturation, Value)
LOWER_YELLOW = np.array([20, 80, 80])
UPPER_YELLOW = np.array([40, 255, 255])

LOWER_ORANGE = np.array([5, 100, 100])
UPPER_ORANGE = np.array([19, 255, 255])

# 안전복 판정 임계치
MIN_VEST_RATIO = 0.03

# 오른쪽 정보 패널 너비
SIDE_PANEL_WIDTH = 360

# 패널 내부 여백
PANEL_PAD_X = 20
PANEL_PAD_Y = 20


# =========================================================
# 2. 공통 함수
# =========================================================

def draw_text(img, text, x, y, color=(255, 255, 255), scale=0.65, thickness=2):
    cv2.putText(
        img,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA
    )


def get_roi(frame):
    h, w = frame.shape[:2]
    x1 = int(w * ROI_X1_RATIO)
    y1 = int(h * ROI_Y1_RATIO)
    x2 = int(w * ROI_X2_RATIO)
    y2 = int(h * ROI_Y2_RATIO)

    # 안전장치
    x1 = max(0, min(x1, w - 1))
    x2 = max(1, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(1, min(y2, h))

    return x1, y1, x2, y2


def detect_safety_vest(roi_bgr):
    """
    ROI 내부에서 형광 노랑/주황 비율 계산
    """
    if roi_bgr is None or roi_bgr.size == 0:
        empty_mask = np.zeros((100, 100), dtype=np.uint8)
        return False, 0.0, empty_mask

    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)

    yellow_mask = cv2.inRange(hsv, LOWER_YELLOW, UPPER_YELLOW)
    orange_mask = cv2.inRange(hsv, LOWER_ORANGE, UPPER_ORANGE)

    combined_mask = cv2.bitwise_or(yellow_mask, orange_mask)

    # 노이즈 제거
    kernel = np.ones((5, 5), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

    vest_pixels = cv2.countNonZero(combined_mask)
    total_pixels = roi_bgr.shape[0] * roi_bgr.shape[1]

    vest_ratio = vest_pixels / total_pixels if total_pixels > 0 else 0.0
    has_vest = vest_ratio >= MIN_VEST_RATIO

    return has_vest, vest_ratio, combined_mask


def safe_resize_keep_ratio(img, max_w, max_h):
    """
    주어진 최대 너비/높이 안에 들어오도록 비율 유지 축소
    """
    h, w = img.shape[:2]

    if h <= 0 or w <= 0 or max_w <= 0 or max_h <= 0:
        return None

    scale = min(max_w / w, max_h / h)
    scale = min(scale, 1.0)  # 확대는 하지 않음

    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    resized = cv2.resize(img, (new_w, new_h))
    return resized


def build_side_panel(frame_h, frame_idx, total_frames, has_vest, vest_ratio, mask):
    """
    오른쪽 정보 패널 생성
    텍스트와 마스크는 모두 이 패널 안에만 출력
    """
    panel = np.full((frame_h, SIDE_PANEL_WIDTH, 3), 30, dtype=np.uint8)

    left = PANEL_PAD_X
    right = SIDE_PANEL_WIDTH - PANEL_PAD_X

    # 제목
    y = PANEL_PAD_Y + 20
    draw_text(panel, "Safety Vest Check", left, y, (255, 255, 255), 0.85, 2)
    y += 18
    cv2.line(panel, (left, y), (right, y), (90, 90, 90), 1)

    # 상태
    y += 35
    status_text = "STATUS: OK" if has_vest else "STATUS: NOT DETECTED"
    status_color = (0, 255, 0) if has_vest else (0, 0, 255)

    line_gap = 34

    draw_text(panel, status_text, left, y, status_color, 0.72, 2)
    y += line_gap
    draw_text(panel, f"Vest ratio : {vest_ratio:.4f}", left, y, (255, 255, 255), 0.62, 2)
    y += line_gap
    draw_text(panel, f"Threshold  : {MIN_VEST_RATIO:.4f}", left, y, (210, 210, 210), 0.62, 2)
    y += line_gap
    draw_text(panel, f"Frame      : {frame_idx}/{total_frames}", left, y, (210, 210, 210), 0.62, 2)
    y += line_gap
    draw_text(panel, "Keys       : SPACE pause, Q quit", left, y, (190, 190, 190), 0.54, 2)

    # 마스크 제목
    y += 45
    mask_title_y = y
    if mask_title_y < frame_h - 30:
        draw_text(panel, "Color Mask Preview", left, mask_title_y, (255, 255, 255), 0.68, 2)

    # 마스크 표시 가능 영역 계산
    mask_top = mask_title_y + 20
    mask_bottom_reserved = 110   # 하단 안내문용 최소 공간 확보
    max_mask_h = frame_h - mask_top - mask_bottom_reserved
    max_mask_w = SIDE_PANEL_WIDTH - 2 * PANEL_PAD_X

    if max_mask_h > 40 and max_mask_w > 40:
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        mask_small = safe_resize_keep_ratio(mask_bgr, max_mask_w, max_mask_h)

        if mask_small is not None:
            mh, mw = mask_small.shape[:2]
            px = left
            py = mask_top

            # 테두리
            cv2.rectangle(
                panel,
                (px - 2, py - 2),
                (px + mw + 2, py + mh + 2),
                (120, 120, 120),
                1
            )

            # 패널 범위 내에서만 삽입
            panel[py:py + mh, px:px + mw] = mask_small

            y_info = py + mh + 30
        else:
            y_info = mask_top + 20
    else:
        y_info = mask_top + 20

    # 안내문도 공간 있을 때만 출력
    info_lines = [
        "Green ROI box is inspection area.",
        "Only colors inside ROI are counted.",
        "Yellow / Orange pixels -> vest estimate"
    ]

    for line in info_lines:
        if y_info < frame_h - 20:
            draw_text(panel, line, left, y_info, (210, 210, 210), 0.54, 2)
            y_info += 28

    return panel


def analyze_frame(frame, frame_idx, total_frames):
    """
    왼쪽은 원본 영상 + ROI
    오른쪽은 별도 정보 패널
    """
    video_view = frame.copy()

    # ROI 계산
    x1, y1, x2, y2 = get_roi(frame)
    roi = frame[y1:y2, x1:x2]

    # 안전복 검사
    has_vest, vest_ratio, mask = detect_safety_vest(roi)

    # ROI 표시
    roi_color = (0, 255, 0) if has_vest else (0, 0, 255)
    cv2.rectangle(video_view, (x1, y1), (x2, y2), roi_color, 3)

    # 왼쪽 영상 하단에 한 줄만 표시
    h, w = video_view.shape[:2]
    box_x1, box_y1 = 10, max(0, h - 50)
    box_x2, box_y2 = min(w - 10, 300), h - 10
    cv2.rectangle(video_view, (box_x1, box_y1), (box_x2, box_y2), (20, 20, 20), -1)
    draw_text(video_view, "Inspection ROI Active", 20, h - 20, (255, 220, 0), 0.65, 2)

    # 오른쪽 패널 생성
    side_panel = build_side_panel(
        frame_h=h,
        frame_idx=frame_idx,
        total_frames=total_frames,
        has_vest=has_vest,
        vest_ratio=vest_ratio,
        mask=mask
    )

    combined = np.hstack([video_view, side_panel])

    return combined, has_vest, vest_ratio


# =========================================================
# 3. 실행 함수
# =========================================================

def run_video_mode(video_path, output_path):
    if not os.path.exists(video_path):
        print(f"동영상 파일이 없습니다: {video_path}")
        return

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"동영상을 열 수 없습니다: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 20.0

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_w = frame_w + SIDE_PANEL_WIDTH
    output_h = frame_h

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (output_w, output_h))

    print("===== 동영상 검사 시작 =====")
    print(f"입력 파일: {video_path}")
    print(f"출력 파일: {output_path}")
    print(f"입력 해상도: {frame_w} x {frame_h}")
    print(f"출력 해상도: {output_w} x {output_h}")
    print(f"FPS: {fps:.2f}")
    print(f"총 프레임 수: {total_frames}")

    paused = False
    frame_idx = 0
    detected_count = 0
    ratios = []

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("동영상 끝")
                break

            frame_idx += 1

            result_frame, has_vest, vest_ratio = analyze_frame(
                frame=frame,
                frame_idx=frame_idx,
                total_frames=total_frames
            )

            if has_vest:
                detected_count += 1
            ratios.append(vest_ratio)

            cv2.imshow("Safety Vest Check Result", result_frame)
            writer.write(result_frame)

        key = cv2.waitKey(30) & 0xFF

        if key == ord('q'):
            print("사용자 종료")
            break
        elif key == ord(' '):
            paused = not paused

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    print("\n===== 검사 결과 요약 =====")
    print(f"처리 프레임 수: {frame_idx}")

    if frame_idx > 0:
        avg_ratio = sum(ratios) / len(ratios)
        detect_percent = detected_count / frame_idx * 100.0
        print(f"안전복 검출 프레임 수: {detected_count}")
        print(f"검출 비율: {detect_percent:.2f}%")
        print(f"평균 vest ratio: {avg_ratio:.4f}")
    else:
        print("처리된 프레임이 없습니다.")


# =========================================================
# 4. 메인 실행
# =========================================================

if __name__ == "__main__":
    run_video_mode(VIDEO_PATH, OUTPUT_PATH)