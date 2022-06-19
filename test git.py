import cv2
import mediapipe as mp
import time
import math
 
class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        #Số tay tối đa trong khung, mặc định là 2
        self.maxHands = maxHands
        #Độ tin cậy khi phát hiện
        self.detectionCon = detectionCon
        #Nếu theo dõi khung, thì theo dõi tin cậy
        self.trackCon = trackCon
        #Các hàm sử dụng khi dùng Handtracking
        self.mpHands = mp.solutions.hands
        #Hàm trả về tọa độ các landmark.
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,
        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        #List các đầu ngón tay
        self.tipIds = [4, 8, 12, 16, 20]
    
    #Hàm phát hiện bàn tay.
    def findHands(self, img, draw=True):
        #Xử lý Handstracking làm việc với ảnh RGB nên chúng ta chuyển BGR sang RGB
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        #Nếu mà nhận diện được bàn tay
        if self.results.multi_hand_landmarks:
            #Quét lần lượt 21 điểm ảnh
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    #Nối các Landmark lại với nhau
                    self.mpDraw.draw_landmarks(img, handLms,self.mpHands.HAND_CONNECTIONS)
        return img
    
    #Hàm tìm vị trí bàn tay
    def findPosition(self, img, handNo=0, draw=True):
        #Các mảng x,y chứa tọa độ của các Landmark
        xList = []
        yList = []
        #Mảng bbox chứa tọa độ của đường biên hình chữ nhật.
        bbox = []
        #Mảng chứa tọa độ của các Landmark
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            #Xét lần lượt các điểm landmark
            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                h, w, c = img.shape
                #Nhân cho kích thước của các cạnh camera để hiện tọa độ theo tỉ lệ
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                # print(id, cx, cy)
                self.lmList.append([id, cx, cy])
                if draw:
                    #Vẽ các vòng tròn màu vào các điểm có tọa độ cx cy các landmark
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
            #Tìm 4 tọa độ x y của 4 đỉnh hình chữ nhật biên
            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            #Add vào list bbox
            bbox = xmin, ymin, xmax, ymax
            #Vẽ hình chữ nhật biên nhưng rộng ra thêm 20 đơn vị
            if draw:
                cv2.rectangle(img, (bbox[0] - 20, bbox[1] - 20),
            (bbox[2] + 20, bbox[3] + 20), (0, 255, 0), 2)
        #Trả về list các tọa độ landmark và list đường biên
        return self.lmList,bbox
    #Hàm kiểm tra ngón tay up down
    def fingersUp(self):
        fingers = []
        # Ngón cái - Thì xét tọa độ x của đầu ngón cái có lớn hơn điểm số 3 không.
        if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        # 4 ngón còn lại - Tọa độ y của các đầu ngón tay nhỏ hơn 2 đốt dưới ko.
        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers
    #Hàm tìm khoảng cách
    def findDistance(self, p1, p2, img, draw=True):
        #Lấy tọa độ x,y của các điểm đầu ngón tay
        x1, y1 = self.lmList[p1][1], self.lmList[p1][2]
        x2, y2 = self.lmList[p2][1], self.lmList[p2][2]
        #Tìm tọa độ trung điểm và làm tròn.
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        
        if draw:
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
        #Hàm tính khoảng cách giữa 2 điểm trong math.
        length = math.hypot(x2 - x1, y2 - y1)
        return length, img, [x1, y1, x2, y2, cx, cy]

def main():
    pTime = 0
    cTime =0
    #Sử dụng camera của thiết bị laptop
    cap = cv2.VideoCapture(0)
    #Gọi class handetector
    detector = handDetector()
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[4])
        #Tính độ phân giải
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        #Hiển thị độ phân giải.
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
        (255, 0, 255), 3)
        
        cv2.imshow("Image", img)
        cv2.waitKey(1)
 

if __name__ == "__main__":
     main()