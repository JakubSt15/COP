import csv
from PyQt5 import QtCore, QtWidgets
from DoctorMenu.DoctorMenuList import Ui_MainWindow

class Ui_LoginWindow(object):
    def setupUi(self, LoginWindow):
        LoginWindow.setObjectName("LoginWindow")
        LoginWindow.resize(315, 273)

        # Central widget
        self.centralwidget = QtWidgets.QWidget(LoginWindow)
        self.centralwidget.setObjectName("centralwidget")

        # Layout
        self.verticalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(30, 20, 251, 221))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")

        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")

        # Login
        self.loginTextBox = QtWidgets.QLineEdit(self.verticalLayoutWidget)
        self.loginTextBox.setObjectName("loginTextBox")
        self.loginTextBox.setText("Login")
        self.verticalLayout.addWidget(self.loginTextBox)

        # Password
        self.passwordTextBox = QtWidgets.QLineEdit(self.verticalLayoutWidget)
        self.passwordTextBox.setPlaceholderText("Hasło")
        self.passwordTextBox.setObjectName("passwordTextBox")
        self.passwordTextBox.setEchoMode(QtWidgets.QLineEdit.Password)
        self.verticalLayout.addWidget(self.passwordTextBox)

        # Login button
        self.LoginButton = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.LoginButton.setObjectName("LoginButton")
        self.LoginButton.setText("Login")
        self.verticalLayout.addWidget(self.LoginButton)

        # Status bar
        self.statusbar = QtWidgets.QStatusBar(LoginWindow)
        self.statusbar.setObjectName("statusbar")
        LoginWindow.setStatusBar(self.statusbar)

        LoginWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(LoginWindow)
        QtCore.QMetaObject.connectSlotsByName(LoginWindow)

        self.LoginButton.clicked.connect(self.onLoginClicked)

    def retranslateUi(self, LoginWindow):
        _translate = QtCore.QCoreApplication.translate
        LoginWindow.setWindowTitle(_translate("LoginWindow", "Login Window"))

    def onLoginClicked(self):
        login = self.loginTextBox.text()
        password = self.passwordTextBox.text()

        def isValidLogin(login, password):
            with open("./Users.csv", "r") as csvfile:
                reader = csv.reader(csvfile, delimiter=";")
                next(reader, None)
                users = list(reader)

            for user in users:
                if user[4] == login and user[5] == password:
                    print(f"Zalogowano użytkownika {user[1]} {user[2]} ({user[3]})")
                    return True

            print("Błędny login lub hasło")
            return False

        if not isValidLogin(login, password):
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Critical)
            msg.setText("Niepoprawny login lub hasło")
            msg.setWindowTitle("Błąd logowania")
            msg.exec_()
        else:
            self.MenuList()

    def MenuList(self):
        self.doctor_menu_list = QtWidgets.QMainWindow()
        self.second_ui = Ui_MainWindow()
        self.second_ui.setupUi(self.doctor_menu_list)
        self.doctor_menu_list.show()
        LoginWindow.close()


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    LoginWindow = QtWidgets.QMainWindow()
    ui = Ui_LoginWindow()
    ui.setupUi(LoginWindow)
    LoginWindow.show()
    sys.exit(app.exec())
