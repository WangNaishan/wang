# -*- coding: utf-8 -*-

import sys
from datetime import datetime

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from id_utils import get_id_info_mysql, sava_id_info_mysql  # 账号信息工具函数
from lib.share import shareInfo  # 公共变量名

# 导入QT-Design生成的UI
from ui.login_ui import Login_Ui_Form
from ui.registe_ui import Ui_Dialog

# 导入设计好的检测界面
from detect_logical6_mp3 import UI_Logic_Window


# 界面登录
class win_Login(QMainWindow):
    def __init__(self, parent=None):
        super(win_Login, self).__init__(parent)
        self.ui_login = Login_Ui_Form()
        self.ui_login.setupUi(self)
        self.init_slots()
        self.hidden_pwd()

    # 密码输入框隐藏
    def hidden_pwd(self):
        self.ui_login.edit_password.setEchoMode(QLineEdit.Password)

    # 绑定信号槽
    def init_slots(self):
        self.ui_login.btn_login.clicked.connect(self.onSignIn)  # 点击按钮登录
        self.ui_login.edit_password.returnPressed.connect(self.onSignIn)  # 按下回车登录
        # 注释注册按钮
        self.ui_login.btn_regeist.clicked.connect(self.create_id)

    # 跳转到注册界面
    def create_id(self):
        shareInfo.createWin = win_Register()
        shareInfo.createWin.show()

    # 登录
    def onSignIn(self):
        print("You pressed sign in")
        # 从登陆界面获得输入账户名与密码
        username = self.ui_login.edit_username.text().strip()
        password = self.ui_login.edit_password.text().strip()

        # 从mysql数据库中获取账号信息
        USER_PWD = get_id_info_mysql()
        print(USER_PWD)

        if username not in USER_PWD.keys():
            replay = QMessageBox.warning(self, "登陆失败!", "账号或密码输入错误", QMessageBox.Yes)
        else:
            # 若登陆成功，则跳转主界面
            if USER_PWD.get(username) == password:
                print("Jump to main window")
                # 打开检测窗口
                shareInfo.mainWin = UI_Logic_Window(username)
                shareInfo.mainWin.show()
                # 关闭当前窗口
                self.close()
            else:
                replay = QMessageBox.warning(self, "!", "账号或密码输入错误", QMessageBox.Yes)


# 注册界面
class win_Register(QDialog):
    def __init__(self, parent=None):
        super(win_Register, self).__init__(parent)
        self.ui_register = Ui_Dialog()
        self.ui_register.setupUi(self)
        self.init_slots()

    # 绑定槽信号
    def init_slots(self):
        self.ui_register.pushButton_regiser.clicked.connect(self.new_account)
        self.ui_register.pushButton_cancer.clicked.connect(self.cancel)

    # 创建新账户
    def new_account(self):
        print("Create new account")
        USER_PWD = get_id_info_mysql()  # 获取已存在的用户信息
        new_username = self.ui_register.edit_username.text().strip()
        new_password = self.ui_register.edit_password.text().strip()
        # 判断用户名是否为空
        if new_username == "":
            replay = QMessageBox.warning(self, "!", "账号不准为空", QMessageBox.Yes)
        else:
            # 判断账号是否存在
            if new_username in USER_PWD.keys():
                replay = QMessageBox.warning(self, "!", "账号已存在", QMessageBox.Yes)
            else:
                # 判断密码是否为空
                if new_password == "":
                    replay = QMessageBox.warning(self, "!", "密码不能为空", QMessageBox.Yes)
                else:
                    # 注册成功
                    print("Successful!")
                    # 写入账户信息到mysql数据库
                    sava_id_info_mysql(new_username, new_password)
                    replay = QMessageBox.warning(self, "!", "注册成功！", QMessageBox.Yes)
                    # 关闭界面
                    self.close()

    # 取消注册
    def cancel(self):
        self.close()  # 关闭当前界面


if __name__ == "__main__":
    app = QApplication(sys.argv)
    # 利用共享变量名来实例化对象
    shareInfo.loginWin = win_Login()  # 登录界面作为主界面
    shareInfo.loginWin.show()
    sys.exit(app.exec_())
