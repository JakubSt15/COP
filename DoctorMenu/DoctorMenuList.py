# -*- coding: utf-8 -*-

# This line imports necessary modules from PyQt5 library
from PyQt5 import QtCore, QtWidgets

from ShowEDF import ShowEDFWindow
from ModelRun import ModelView
from UserListWindow import UserListWindow
import LoginWindow


# This class definition creates the user interface for the main window
class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        """
        This function initializes the main window's layout and widgets.

        Args:
            MainWindow (QtWidgets.QMainWindow): The main window object.
        """
        MainWindow.setObjectName("Doctor Menu List")

        # Set a minimum size for the window (optional)
        MainWindow.setMinimumSize(600, 400)  # Adjust values as needed

        # Create the central widget for the main window
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        # Create a grid layout for arranging the buttons
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setContentsMargins(25, 25, 25, 25)
        self.gridLayout.setObjectName("gridLayout")

        self.ShowEDFFile = QtWidgets.QPushButton(self.centralwidget)
        self.ShowEDFFile.setObjectName("ShowEDFFile")
        self.ShowEDFFile.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout.addWidget(self.ShowEDFFile, 1, 0, 1, 1)  # Add to grid layout
        self.ShowEDFFile.clicked.connect(self.onShowEDFCliced)

        self.UsersList = QtWidgets.QPushButton(self.centralwidget)
        self.UsersList.setObjectName("UsersList")
        self.UsersList.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout.addWidget(self.UsersList, 0, 0, 1, 1)  # Add to grid layout
        self.UsersList.clicked.connect(self.onShowUserList)

        self.RunModel = QtWidgets.QPushButton(self.centralwidget)
        self.RunModel.setObjectName("RunModel")
        self.RunModel.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout.addWidget(self.RunModel, 2, 0, 1, 1)  # Add to grid layout
        self.RunModel.clicked.connect(self.onShowRunModelCliced)

        self.Logout = QtWidgets.QPushButton(self.centralwidget)
        self.Logout.setObjectName("Logout")
        self.Logout.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout.addWidget(self.Logout, 3, 0, 1, 1)
        self.Logout.clicked.connect(self.onLogoutCliced)

        # Set the central widget of the main window
        MainWindow.setCentralWidget(self.centralwidget)

        # Create a status bar for the main window
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        """
        This function translates the button text into the desired language.

        Args:
            MainWindow (QtWidgets.QMainWindow): The main window object.
        """
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("Doctor Menu List", "Doctor Menu List"))
        self.UsersList.setText(_translate("Doctor Menu List", "See Users List"))
        self.RunModel.setText(_translate("Doctor Menu List", "Run Model"))
        self.ShowEDFFile.setText(_translate("Doctor Menu List", "Show EDF File"))
        self.Logout.setText(_translate("Doctor Menu List", "Logout"))

    def onShowUserList(self):
        QtWidgets.qApp.closeAllWindows()
        self.userList = QtWidgets.QMainWindow()
        self.secondWindow = UserListWindow.Ui_UserListWindow()
        self.secondWindow.setupUi(self.userList)
        self.userList.show()

    def onShowEDFCliced(self):
        QtWidgets.qApp.closeAllWindows()
        self.showEDF = QtWidgets.QMainWindow()
        self.secondWindow = ShowEDFWindow.Ui_MainWindow()
        self.secondWindow.setupUi(self.showEDF)
        self.showEDF.show()

    def onShowRunModelCliced(self):
        QtWidgets.qApp.closeAllWindows()
        self.runModel = QtWidgets.QMainWindow()
        self.secondWindow = ModelView.Ui_MainWindow(self.runModel)
        self.secondWindow.setupUi(self.runModel)
        self.runModel.show()

    def onLogoutCliced(self):

        QtWidgets.qApp.closeAllWindows()
        self.logoutWindow = QtWidgets.QMainWindow()
        self.secondWindow = LoginWindow.Ui_LoginWindow()
        self.secondWindow.setupUi(self.logoutWindow)
        self.logoutWindow.show()

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
