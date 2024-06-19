import csv
from PyQt5 import QtCore, QtWidgets
from CommonTools.CommonTools import show_popup, QMessageBox
class Ui_LoginWindow(object):
    def setupUi(self, LoginWindow):
        LoginWindow.setObjectName("LoginWindow")
        LoginWindow.resize(300, 300)
        
        # Central widget
        self.centralwidget = QtWidgets.QWidget(LoginWindow)
        self.centralwidget.setObjectName("centralwidget")

        # Layout
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setContentsMargins(25, 25, 25, 25)
        self.gridLayout.setObjectName("gridLayout")

        # Login
        self.loginTextBox = QtWidgets.QLineEdit(self.centralwidget)
        self.loginTextBox.setObjectName("loginTextBox")
        self.loginTextBox.setPlaceholderText("Login")
        self.gridLayout.addWidget(self.loginTextBox, 0, 0, 1, 1)

        # Password
        self.passwordTextBox = QtWidgets.QLineEdit(self.centralwidget)
        self.passwordTextBox.setPlaceholderText("Hasło")
        self.passwordTextBox.setObjectName("passwordTextBox")
        self.passwordTextBox.setEchoMode(QtWidgets.QLineEdit.Password)
        self.gridLayout.addWidget(self.passwordTextBox, 1, 0, 1, 1)

        # Login button
        self.LoginButton = QtWidgets.QPushButton(self.centralwidget)
        self.LoginButton.setObjectName("LoginButton")
        self.LoginButton.setText("Login")
        self.gridLayout.addWidget(self.LoginButton, 2, 0, 1, 1)

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
                    show_popup("Sukces", f"Zalogowano użytkownika {user[1]} {user[2]} ({user[3]})", QMessageBox.Information)
                    return True

            return False

        if not isValidLogin(login, password):
            show_popup("Błąd", "Nieprawidłowe Hasło lub Login!", QMessageBox.Warning)
        else:
            self.MenuList()

    def MenuList(self):
        from DoctorMenu.DoctorMenuList import Ui_MainWindow

        QtWidgets.qApp.closeAllWindows()
        self.doctor_menu_list = QtWidgets.QMainWindow()
        self.second_ui = Ui_MainWindow()
        self.second_ui.setupUi(self.doctor_menu_list)
        self.doctor_menu_list.show()


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    LoginWindow = QtWidgets.QMainWindow()
    ui = Ui_LoginWindow()
    ui.setupUi(LoginWindow)
    LoginWindow.show()
    sys.exit(app.exec())