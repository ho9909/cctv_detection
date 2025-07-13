from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QPixmap, QImage
import PyQt5.QtCore as qtcore
from time import sleep
import threading
import cv2


class Ui_MainWindow(object):
    #기본적으로 창만드는 작업
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("ForSign")
        MainWindow.resize(660, 490) # 창 사이즈 - 가로+20 세로+10
        MainWindow.move(500,500)    # 창 뜰 때 위치

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.video_viewer_label = QtWidgets.QLabel(self.centralwidget)
        self.video_viewer_label.setGeometry(QtCore.QRect(10, 10, 640, 480)) #영상 크기 조절

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)




    def Video_to_frame(self, MainWindow):
        cap = cv2.VideoCapture('7010.mp4')  # 저장된 영상 가져오기 프레임별로 계속 가져오는 듯
        while True:
            self.ret, self.frame = cap.read()  # 영상의 정보 저장
            if self.ret:
                self.rgbImage = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)  # 프레임에 색입히기
                self.convertToQtFormat = QImage(self.rgbImage.data, self.rgbImage.shape[1], self.rgbImage.shape[0],
                                                QImage.Format_RGB888)
                self.pixmap = QPixmap(self.convertToQtFormat)
                self.p = self.pixmap.scaled(640, 480, QtCore.Qt.IgnoreAspectRatio)  # 프레임 크기 조정
                self.video_viewer_label.setPixmap(self.p)
                self.video_viewer_label.update()  # 프레임 띄우기
                sleep(0.015)  # 영상 1프레임당 0.01초로 이걸로 영상 재생속도 조절하면됨 0.02로하면 0.5배속인거임
            else:
                break

        cap.release()
        cv2.destroyAllWindows()



    # 창 이름 설정
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("vehicle detection", "vehicle detection"))

    # video_to_frame을 쓰레드로 사용
    #이게 영상 재생 쓰레드 돌리는거 얘를 조작하거나 함수를 생성해서 연속재생 관리해야할듯
    def video_thread(self, MainWindow):
        thread = threading.Thread(target=self.Video_to_frame, args=(self,))
        thread.daemon = True  # 프로그램 종료시 프로세스도 함께 종료 (백그라운드 재생 X)
        thread.start()



    def mousePressEvent(self, e):
        if e.buttons() & qtcore.Qt.LeftButton:      # 마우스 클릭했을 때
            if()                                    # 바운딩 박스 안을 클릭했을 때



if __name__ == "__main__":
    import sys

    #화면 만들려면 기본으로 있어야 하는 코드들 건들지않기
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)

    #영상 스레드 시작
    ui.video_thread(MainWindow)

    #창 띄우기
    MainWindow.show()

    sys.exit(app.exec_())