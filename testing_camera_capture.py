import cv2 as cv
import numpy as np
import math

cptr = cv.VideoCapture(0)
def nothing(x):
    pass

# Создание окна настроек + ползунки - через него настраивается изображение для выделения объектов
cv.namedWindow('HSV SETTINGS')
cv.createTrackbar('LH', 'HSV SETTINGS', 0, 180, nothing)
cv.createTrackbar('LS', 'HSV SETTINGS', 0, 255, nothing)
cv.createTrackbar('LV', 'HSV SETTINGS', 0, 255, nothing)
cv.createTrackbar('UH', 'HSV SETTINGS', 180, 180, nothing)
cv.createTrackbar('US', 'HSV SETTINGS', 255, 255, nothing)
cv.createTrackbar('UV', 'HSV SETTINGS', 255, 255, nothing)

# Через цикл воспроизводим видео
while cptr.isOpened():

    # Считывание кадра и уменьшение окна
    rec, frame = cptr.read()
    frame = cv.resize(frame, None, fx=0.4, fy=0.4)

    # Перевод в HSV палитру + сглаживание
    hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    hsv_frame_blurred = cv.GaussianBlur(hsv_frame, (5, 5), 0)

    setting_frame = hsv_frame_blurred

    # Получение значений с ползунков
    l_h = cv.getTrackbarPos('LH', 'HSV SETTINGS')
    l_s = cv.getTrackbarPos('LS', 'HSV SETTINGS')
    l_v = cv.getTrackbarPos('LV', 'HSV SETTINGS')

    u_h = cv.getTrackbarPos('UH', 'HSV SETTINGS')
    u_s = cv.getTrackbarPos('US', 'HSV SETTINGS')
    u_v = cv.getTrackbarPos('UV', 'HSV SETTINGS')

    # Задание нижней и верхней границы по цвету для фильтрации изображения
    l_filter = np.array([l_h, l_s, l_v], np.uint8)
    u_filter = np.array([u_h, u_s, u_v], np.uint8)

    # Создание маски:
    # Черно-белая
    thresh = cv.inRange(setting_frame, l_filter, u_filter)
    # Маска с HSV-параметрами - далее не используется
    FrameWithSettings = cv.bitwise_and(setting_frame, setting_frame, mask=thresh)
    cv.imshow('HSV SETTINGS', thresh)
    #cv.imshow('thresh', thresh)

    # kernel = np.ones((5, 5), np.uint8)
    # thresh = cv.erode(thresh, kernel)


    # Нахождение контуров через маску
    contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # Далее по циклу проходимся по каждому контуру
    for cnt in contours:

        # Узнаем площадь, для отсечения мусора в кадре
        AreaOfCnt = cv.contourArea(cnt)

        if AreaOfCnt > 500:

            # Эта функция соединяет точки прямыми линиями для более чёткого и ровного контура
            approx = cv.approxPolyDP(cnt, 0.01*cv.arcLength(cnt, True), True)
            #cv.drawContours(frame, [approx], -1, (255, 255, 0), 2)

            # Длинна аппроксимации понадобится для определения фигуры
            len_approx = len(approx)

            # Вычисление центра контура через момент
            M = cv.moments(approx)
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            # Отрисовка круга в центре
            cv.circle(frame, (cx, cy), 1, (0, 255, 0), -1)

            # Задаем параметры шрифта
            font = cv.FONT_HERSHEY_COMPLEX
            font_size = 0.4
            font_bold = 1


            # Определение квадрата
            if 4 <= len_approx <=8:
                rect = cv.minAreaRect(cnt)
                box = cv.boxPoints(rect)
                box = np.int0(box)
                cv.drawContours(frame, [box], 0, (255, 0, 0), 2)

                # Вычисление координат двух векторов, являющихся сторонам прямоугольника
                edge1 = np.int0((box[1][0] - box[0][0], box[1][1] - box[0][1]))
                edge2 = np.int0((box[2][0] - box[1][0], box[2][1] - box[1][1]))

                # Выясняем какой вектор больше
                usedEdge = edge1
                if cv.norm(edge2) > cv.norm(edge1):
                    usedEdge = edge2
                reference = (1, 0)

                # Вычисляем угол между самой длинной стороной прямоугольника и горизонтом
                angle = 180.0 / math.pi * math.acos(
                    (reference[0] * usedEdge[0] + reference[1] * usedEdge[1]) / (
                                cv.norm(reference) * cv.norm(usedEdge)))

                # Выводим значение угла на фигуру
                cv.putText(frame, str(int(angle)), (cx - 5, cy - 5),
                           font, font_size, (255, 255, 0), font_bold)


            # Определение круга
            elif 9 <= len_approx <= 15:
                ellipse = cv.fitEllipse(cnt)
                cv.ellipse(frame, ellipse, (0, 0, 255), 2)

            else:
                cv.drawContours(frame, [approx], 0, (255, 0, 0), 2)

    cv.imshow('frame', frame)

    key = cv.waitKey(1)
    if key == 27:
        break

cptr.release()