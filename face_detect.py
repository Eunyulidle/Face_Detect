import cv2
import face_recognition
import os
import pickle
import time
import numpy as np

cam_number = 1
ENCODINGS_FILE = 'known_faces.pkl'
MAX_SAMPLES = 20
MAX_FAILS = 10  # 최대 연속 프레임 실패 허용 횟수

# 얼굴 데이터 불러오기
def load_known_faces():
    if os.path.exists(ENCODINGS_FILE):
        with open(ENCODINGS_FILE, 'rb') as f:
            return pickle.load(f)
    return {}

# 얼굴 데이터 저장하기
def save_known_faces(known_faces):
    with open(ENCODINGS_FILE, 'wb') as f:
        pickle.dump(known_faces, f)

# 얼굴 등록 함수
def register_face(name, cap, known_faces):
    print("📸 얼굴을 등록 중입니다. 정면을 바라보세요...")
    encodings = []

    while len(encodings) < MAX_SAMPLES:
        try:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("⚠️ 프레임 수집 실패")
                continue

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)

            if len(face_locations) == 1:
                encoding = face_recognition.face_encodings(rgb_frame, face_locations)[0]
                encodings.append(encoding)
                print(f"🟢 수집된 인코딩 수: {len(encodings)}")
            else:
                print("⚠️ 얼굴이 정확히 한 명 감지되지 않음")

            cv2.putText(frame, f"Learning... ({len(encodings)}/{MAX_SAMPLES})", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imshow('Face Recognition', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("❌ 등록 중단됨")
                return
        except Exception as e:
            print(f"⚠️ 얼굴 등록 중 오류 발생: {e}")
            continue

    known_faces[name] = encodings
    save_known_faces(known_faces)
    print(f"✅ '{name}' 얼굴이 등록되었습니다. 총 {len(encodings)}개 수집됨.")

# 거리 → 신뢰도 변환 함수
def distance_to_confidence(distance, threshold=0.6):
    if distance > threshold:
        return 0.0
    else:
        return 1.0 - (distance / threshold)

# 얼굴 인식 함수 (정확도 0.4 이하 시 Unknown 처리)
def recognize_faces(frame, known_faces):
    try:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations, num_jitters=3)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            name = "Unknown"
            confidence = 0.0
            try:
                best_confidence = 0.0
                best_name = "Unknown"

                for known_name, enc_list in known_faces.items():
                    valid_encodings = [
                        e for e in enc_list if isinstance(e, (list, tuple, np.ndarray)) and len(e) == 128
                    ]
                    if not valid_encodings:
                        continue

                    distances = face_recognition.face_distance(valid_encodings, face_encoding)
                    min_distance = min(distances)
                    conf = distance_to_confidence(min_distance)
                    if conf > best_confidence:
                        best_confidence = conf
                        best_name = known_name

                # 정확도 0.4 이하이면 Unknown 처리
                if best_confidence > 0.4:
                    name = best_name
                    confidence = best_confidence
                else:
                    name = "Unknown"
                    confidence = 0.0

            except Exception as match_err:
                print(f"⚠️ 얼굴 비교 중 오류 발생: {match_err}")

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            if name == "Unknown":
                cv2.putText(frame, name, (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)
            else:
                cv2.putText(frame, f"{name} ({confidence:.2f})", (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)

    except Exception as recog_err:
        print(f"⚠️ 얼굴 인식 중 오류 발생: {recog_err}")

# 메인 루프
def main():
    known_faces = load_known_faces()
    cap = cv2.VideoCapture(1)
    fail_count = 0

    if not cap.isOpened():
        print("❌ 웹캠을 열 수 없습니다.")
        return

    print("🎥 얼굴 인식 시스템 시작됨")
    print("👤 얼굴 등록: r 키, 종료: q 키")

    while True:
        try:
            ret, frame = cap.read()
            if not ret or frame is None:
                fail_count += 1
                print(f"⚠️ 프레임 수신 실패 ({fail_count}/{MAX_FAILS})")

                if fail_count >= MAX_FAILS:
                    print("🔄 카메라 재연결 시도 중...")
                    cap.release()
                    time.sleep(1)
                    cap = cv2.VideoCapture(cam_number)
                    fail_count = 0
                continue

            fail_count = 0  # 성공 시 초기화

            recognize_faces(frame, known_faces)
            cv2.imshow('Face Recognition', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('r'):
                name = input("Enter ID to register: ")
                register_face(name, cap, known_faces)
            elif key == ord('q'):
                break

        except Exception as e:
            print(f"❌ 예외 발생: {e}")
            cap.release()
            time.sleep(1)
            cap = cv2.VideoCapture(cam_number)
            continue

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
