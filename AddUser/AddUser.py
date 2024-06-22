import csv
from DoctorMenu import DoctorMenuList
from UserListWindow import UserListWindow
from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_AddUser(object):
    def setupUi(self, AddUser):
        AddUser.setObjectName("AddUser")
        AddUser.resize(805, 600)

        self.centralwidget = QtWidgets.QWidget(AddUser)
        self.centralwidget.setObjectName("centralwidget")

        # Create a grid layout for arranging the buttons
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setContentsMargins(25, 25, 25, 25)
        self.gridLayout.setObjectName("gridLayout")

        # Clear Button
        self.Clear = QtWidgets.QPushButton(self.centralwidget)
        self.Clear.setObjectName("Clear")
        self.gridLayout.addWidget(self.Clear, 4, 2, 1, 1)

        # Add User button
        self.AddUser_2 = QtWidgets.QPushButton(self.centralwidget)
        self.AddUser_2.setObjectName("AddUser_2")
        self.gridLayout.addWidget(self.AddUser_2, 4, 1, 1, 1)

        # Close Button
        self.Close = QtWidgets.QPushButton(self.centralwidget)
        self.Close.setObjectName("Close")
        self.gridLayout.addWidget(self.Close, 4, 0, 1, 1)

        self.UsenName = QtWidgets.QLineEdit(self.centralwidget)
        self.UsenName.setInputMask("")
        self.UsenName.setObjectName("UsenName")
        self.gridLayout.addWidget(self.UsenName, 0, 1, 1, 1)

        self.UserLastName = QtWidgets.QLineEdit(self.centralwidget)
        self.UserLastName.setInputMask("")
        self.UserLastName.setObjectName("UserLastName")
        self.gridLayout.addWidget(self.UserLastName, 1, 1, 1, 1)

        self.UserLogin = QtWidgets.QLineEdit(self.centralwidget)
        self.UserLogin.setInputMask("")
        self.UserLogin.setObjectName("UserLogin")
        self.gridLayout.addWidget(self.UserLogin, 2, 1, 1, 1)

        self.UserPassword = QtWidgets.QLineEdit(self.centralwidget)
        self.UserPassword.setInputMask("")
        self.UserPassword.setObjectName("UserPassword")
        self.gridLayout.addWidget(self.UserPassword, 3, 1, 1, 1)

        self.label = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(16)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)

        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(16)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 1, 0, 1, 1)

        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(16)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 2, 0, 1, 1)

        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(16)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.gridLayout.addWidget(self.label_4, 3, 0, 1, 1)

        AddUser.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(AddUser)
        self.menubar.setObjectName("menubar")
        AddUser.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(AddUser)
        self.statusbar.setObjectName("statusbar")
        AddUser.setStatusBar(self.statusbar)

        self.retranslateUi(AddUser)
        QtCore.QMetaObject.connectSlotsByName(AddUser)

        self.AddUser_2.clicked.connect(self.add_user)
        self.Close.clicked.connect(self.onCloseClicked)
        self.Clear.clicked.connect(self.clear_label)

        self.apply_styles()

    def apply_styles(self):
        button_style = """
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 10px 20px;
                text-align: center;
                text-decoration: none;
                font-size: 16px;
                margin: 4px 2px;
                border-radius: 12px;
            }
            QPushButton:hover {
                background-color: white;
                color: black;
                border: 2px solid #4CAF50;
            }
        """
        button_style_reversed = """
            QPushButton {
                background-color: #AF4CAB;
                color: white;
                border: none;
                padding: 10px 20px;
                text-align: center;
                text-decoration: none;
                font-size: 16px;
                margin: 4px 2px;
                border-radius: 12px;
            }
            QPushButton:hover {
                background-color: white;
                color: black;
                border: 2px solid #AF4CAB;
            }
        """
        self.Clear.setStyleSheet(button_style)
        self.AddUser_2.setStyleSheet(button_style)
        self.Close.setStyleSheet(button_style_reversed)

        line_edit_style = """
            QLineEdit {
                font-size: 16px;
                padding: 5px;
                border: 1px solid #ccc;
                border-radius: 5px;
            }
        """
        self.UsenName.setStyleSheet(line_edit_style)
        self.UserLastName.setStyleSheet(line_edit_style)
        self.UserLogin.setStyleSheet(line_edit_style)
        self.UserPassword.setStyleSheet(line_edit_style)

        label_style = """
            QLabel {
                font-size: 16px;
            }
        """
        self.label.setStyleSheet(label_style)
        self.label_2.setStyleSheet(label_style)
        self.label_3.setStyleSheet(label_style)
        self.label_4.setStyleSheet(label_style)

    def clear_label(self):
        self.UsenName.clear()
        self.UserLastName.clear()
        self.UserLogin.clear()
        self.UserPassword.clear()

    def add_user(self):
        # Pobieranie danych z pól tekstowych
        name = self.UsenName.text()
        surname = self.UserLastName.text()
        login = self.UserLogin.text()
        password = self.UserPassword.text()

        # Pobieranie następnego ID użytkownika
        user_id = self.get_next_user_id()

        # Tworzenie nowego użytkownika
        role = "1"  # Domyślna rola ustawiona na 1

        # Zapisywanie użytkownika do pliku CSV
        with open('Users.csv', mode='a', newline='') as file:
            writer = csv.writer(file, delimiter=';')
            writer.writerow([user_id, name, surname, role, login, password])

        # Czyszczenie pól tekstowych po dodaniu użytkownika
        self.UsenName.clear()
        self.UserLastName.clear()
        self.UserLogin.clear()
        self.UserPassword.clear()
        self.onCloseClicked()

    def get_next_user_id(self):
        existing_ids = set()

        try:
            with open('Users.csv', mode='r') as file:
                reader = csv.reader(file, delimiter=';')
                next(reader)
                for row in reader:
                    existing_ids.add(int(row[0]))
        except FileNotFoundError:
            return "1"
        except (ValueError, IndexError):
            return "1"

        next_id = max(existing_ids) + 1
        return str(next_id)

    def retranslateUi(self, AddUser):
        _translate = QtCore.QCoreApplication.translate
        AddUser.setWindowTitle(_translate("AddUser", "Add User"))
        self.Clear.setText(_translate("AddUser", "Clear"))
        self.AddUser_2.setText(_translate("AddUser", "Add"))
        self.Close.setText(_translate("AddUser", "Close"))
        self.label.setText(_translate("AddUser", "First Name:"))
        self.label_2.setText(_translate("AddUser", "Last Name:"))
        self.label_3.setText(_translate("AddUser", "Login:"))
        self.label_4.setText(_translate("AddUser", "Password:"))

    def onCloseClicked(self):
        try:
            QtWidgets.qApp.closeAllWindows()
            self.userListWindow = QtWidgets.QMainWindow()
            self.secondWindow = UserListWindow.Ui_UserListWindow()
            self.secondWindow.setupUi(self.userListWindow)
            self.userListWindow.show()
        except Exception as e:
            print(e)


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    AddUser = QtWidgets.QMainWindow()
    ui = Ui_AddUser()
    ui.setupUi(AddUser)
    AddUser.show()
    sys.exit(app.exec_())
