# -*- coding: utf-8 -*-

# This line imports necessary modules from PyQt5 library
import json
from PyQt5 import QtCore, QtWidgets

from ShowEDF import ShowEDFWindow
from ModelRun import ModelView
from UserListWindow import UserListWindow
from Options import Options
import LoginWindow
import csv


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
        _,_,_,self.userType = self.readLoggedUser()
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

        if(self.userType == '3'):
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

        self.Options = QtWidgets.QPushButton(self.centralwidget)
        self.Options.setObjectName("Options")
        self.Options.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout.addWidget(self.Options, 3, 0, 1, 1)
        self.Options.clicked.connect(self.onShowOptionsCliced)

        self.Logout = QtWidgets.QPushButton(self.centralwidget)
        self.Logout.setObjectName("Logout")
        self.Logout.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout.addWidget(self.Logout, 4, 0, 1, 1)
        self.Logout.clicked.connect(self.onLogoutCliced)

        # Set the central widget of the main window
        MainWindow.setCentralWidget(self.centralwidget)

        # Create a status bar for the main window
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.setStyles()
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def setStyles(self):
        # Set button style
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
        self.ShowEDFFile.setStyleSheet(button_style)
        if(self.userType == '3'):
            self.UsersList.setStyleSheet(button_style)
        self.RunModel.setStyleSheet(button_style)
        self.Options.setStyleSheet(button_style)
        self.Logout.setStyleSheet(button_style_reversed)

    def retranslateUi(self, MainWindow):
        """
        This function translates the button text into the desired language.

        Args:
            MainWindow (QtWidgets.QMainWindow): The main window object.
        """
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("Doctor Menu List", "Doctor Menu List"))
        if(self.userType == '3'):
            self.UsersList.setText(_translate("Doctor Menu List", "See Users List"))
        self.RunModel.setText(_translate("Doctor Menu List", "Run Model"))
        self.ShowEDFFile.setText(_translate("Doctor Menu List", "Show EDF File"))
        self.Options.setText(_translate("Doctor Menu List", "Options"))
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
    
    def onShowOptionsCliced(self):
        QtWidgets.qApp.closeAllWindows()
        self.options = QtWidgets.QMainWindow()
        self.secondWindow = Options.Ui_Options()
        self.secondWindow.setupUi(self.options)
        self.options.show()

    def onLogoutCliced(self):
        QtWidgets.qApp.closeAllWindows()
        self.logoutWindow = QtWidgets.QMainWindow()
        self.secondWindow = LoginWindow.Ui_LoginWindow()
        self.secondWindow.setupUi(self.logoutWindow)
        self.logoutWindow.show()
        
    def readLoggedUser(self):
        with open("log_temp.json", "r") as jsonfile:
            login_info = json.load(jsonfile)
        return login_info["login"], login_info["name"], login_info["surname"], login_info["userType"]
            
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
