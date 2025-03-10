import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget
from PyQt5.QtCore import Qt
import sqlite3


class PatientWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('患者信息')
        self.setGeometry(100, 100, 300, 400)

        self.button = QPushButton('显示数据', self)
        self.button.clicked.connect(self.loadData)

        self.table = QTableWidget(self)
        self.table.setColumnCount(2)  # 设置表格列数
        self.table.setHorizontalHeaderLabels(['患者id', '类别'])

        layout = QVBoxLayout()
        layout.addWidget(self.button)
        layout.addWidget(self.table)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def loadData(self):
        # 连接到数据库并查询数据
        conn = sqlite3.connect('detection_database2.db')
        cursor = conn.cursor()

        # 查询所有数据
        cursor.execute('SELECT * FROM detections')
        rows = cursor.fetchall()
        print(rows)

        # 关闭数据库连接
        conn.close()

        # 清空表格中的数据
        self.table.setRowCount(0)

        # 将数据填充到表格中
        for index, row in enumerate(rows):
            self.table.insertRow(index)  # 添加行
            for col, value in enumerate(row):
                item = QTableWidgetItem(str(value))
                self.table.setItem(index, col, item)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = PatientWindow()
    window.show()
    sys.exit(app.exec_())
