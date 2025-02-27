import cv2
import mediapipe as mp
import pyautogui  # Для имитации нажатия клавиш и прокрутки

# Инициализируем модуль MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Запускаем детектор рук
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Захватываем видео с веб-камеры
cap = cv2.VideoCapture(0)

# Флаг, чтобы не нажимать Alt+Tab постоянно
alt_tab_pressed = False

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Переводим цветовое пространство в RGB (MediaPipe работает в RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Получаем координаты кончиков пальцев и оснований
            finger_tips = [hand_landmarks.landmark[i] for i in [4, 8, 12, 16, 20]]
            # Большой, указательный, средний, безымянный, мизинец
            finger_bases = [hand_landmarks.landmark[i - 2] for i in [4, 8, 12, 16, 20]]
            # Основания пальцев

            # Определяем, какие пальцы подняты (если кончик выше основания)
            fingers_up = [tip.y < base.y for tip, base in zip(finger_tips, finger_bases)]

            # Если все 5 пальцев подняты, нажимаем Alt+Tab и скроллим вниз
            if all(fingers_up) and not alt_tab_pressed:
                cv2.putText(frame, "All Fingers Up! Pressing Alt+Tab & Scrolling", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                pyautogui.hotkey("alt", "tab") 
                pyautogui.scroll(-500)         # Прокрутка вниз (отрицательное значение — вниз)
                alt_tab_pressed = True  # Устанавливаем флаг, чтобы избежать повторного срабатывания

            # Если рука изменила положение (не все пальцы подняты), сбрасываем флаг
            elif not all(fingers_up):
                alt_tab_pressed = False

    # Отображаем видео
    cv2.imshow("Hand Detection", frame)

    # Выход по клавише 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
