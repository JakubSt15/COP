import sys
import pandas as pd
from PyQt5 import QtCore, QtGui, QtWidgets
from DoctorMenu import DoctorMenuList

class Ui_UserListWindow(object):
    def setupUi(self, UserListWindow):
        UserListWindow.setObjectName("UserListWindow")
        UserListWindow.resize(800, 600)

        self.centralwidget = QtWidgets.QWidget(UserListWindow)
        self.centralwidget.setObjectName("centralwidget")

        # Create a grid layout for arranging the buttons
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setContentsMargins(25, 25, 25, 25)
        self.gridLayout.setObjectName("gridLayout")

        self.UserList = QtWidgets.QTableView(self.centralwidget)
        self.gridLayout.addWidget(self.UserList, 0, 0, 1, 1)
        self.UserList.setObjectName("UserList")

        self.CloseButton = QtWidgets.QPushButton(self.centralwidget)
        self.gridLayout.addWidget(self.CloseButton, 1, 0, 1, 1)
        self.CloseButton.setObjectName("CloseButton")

        UserListWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(UserListWindow)
        self.statusbar.setObjectName("statusbar")
        UserListWindow.setStatusBar(self.statusbar)

        self.retranslateUi(UserListWindow)
        QtCore.QMetaObject.connectSlotsByName(UserListWindow)

        self.loadData()

        # Connect the close button
        self.CloseButton.clicked.connect(self.close_window)

    def close_window(self):
        QtWidgets.qApp.closeAllWindows()
        self.doctorMenu = QtWidgets.QMainWindow()
        self.secondWindow = DoctorMenuList.Ui_MainWindow()
        self.secondWindow.setupUi(self.doctorMenu)
        self.doctorMenu.show()

    def retranslateUi(self, UserListWindow):
        _translate = QtCore.QCoreApplication.translate
        UserListWindow.setWindowTitle(_translate("UserListWindow", "User List"))
        self.CloseButton.setText(_translate("UserListWindow", "Close"))

    def loadData(self):
        # Load data from CSV file
        df = pd.read_csv('../Users.csv', delimiter=';')

        # Filter data where role is 1
        df_filtered = df[df['role'] == 1]

        # Create model and set it to the QTableView
        model = QtGui.QStandardItemModel()
        model.setHorizontalHeaderLabels(df_filtered.columns)

        for row in df_filtered.itertuples():
            items = [
                QtGui.QStandardItem(str(row.id)),
                QtGui.QStandardItem(str(row.name)),
                QtGui.QStandardItem(str(row.surname)),
                QtGui.QStandardItem(str(row.role)),
                QtGui.QStandardItem(str(row.login)),
                QtGui.QStandardItem(str(row.haslo)),
            ]
            model.appendRow(items)

        self.UserList.setModel(model)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    UserListWindow = QtWidgets.QMainWindow()
    ui = Ui_UserListWindow()
    ui.setupUi(UserListWindow)
    UserListWindow.show()
    sys.exit(app.exec_())
