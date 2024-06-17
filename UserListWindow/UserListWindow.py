import sys
import pandas as pd
from PyQt5 import QtCore, QtGui, QtWidgets
from DoctorMenu import DoctorMenuList


class Ui_UserListWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        # Connect resizeEvent to a function
        self.resizeEvent = self.on_resize

    def setupUi(self, UserListWindow):
        UserListWindow.setObjectName("UserListWindow")
        UserListWindow.resize(800, 600)


        self.centralwidget = QtWidgets.QWidget(UserListWindow)
        self.centralwidget.setObjectName("centralwidget")

        # Create a vertical layout for the central widget
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")

        # Create a QSplitter to divide the window into two parts
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal, self.centralwidget)
        self.splitter.setObjectName("splitter")

        # Create the list of users
        self.userListView = QtWidgets.QListView(self.splitter)
        self.userListView.setObjectName("userListView")

        # Create the detail view
        self.detailWidget = QtWidgets.QWidget(self.splitter)
        self.detailLayout = QtWidgets.QFormLayout(self.detailWidget)
        self.detailWidget.setObjectName("detailWidget")

        self.idLabel = QtWidgets.QLabel(self.detailWidget)
        self.detailLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.idLabel)
        self.nameLabel = QtWidgets.QLabel(self.detailWidget)
        self.detailLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.nameLabel)
        self.surnameLabel = QtWidgets.QLabel(self.detailWidget)
        self.detailLayout.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.surnameLabel)
        self.roleLabel = QtWidgets.QLabel(self.detailWidget)
        self.detailLayout.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.roleLabel)
        self.loginLabel = QtWidgets.QLabel(self.detailWidget)
        self.detailLayout.setWidget(4, QtWidgets.QFormLayout.LabelRole, self.loginLabel)

        # Add the splitter to the main layout
        self.verticalLayout.addWidget(self.splitter)

        # Create and add the close button
        self.CloseButton = QtWidgets.QPushButton(self.centralwidget)
        self.CloseButton.setObjectName("CloseButton")
        self.verticalLayout.addWidget(self.CloseButton)

        UserListWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(UserListWindow)
        self.statusbar.setObjectName("statusbar")
        UserListWindow.setStatusBar(self.statusbar)

        self.retranslateUi(UserListWindow)
        QtCore.QMetaObject.connectSlotsByName(UserListWindow)

        self.loadData()

        # Connect the close button
        self.CloseButton.clicked.connect(self.close_window)

        # Connect the user list view selection
        self.userListView.selectionModel().selectionChanged.connect(self.displayUserDetails)

        # Set initial splitter position and restrict maximum width for the left pane
        self.splitter.setSizes([UserListWindow.width() // 4, UserListWindow.width() * 3 // 4])
        self.splitter.splitterMoved.connect(self.adjust_splitter)

        # Connect resizeEvent to a function
        self.resizeEvent = self.on_resize

    def adjust_splitter(self, pos, index):
        if index == 1:
            max_width = self.splitter.width() // 4
            if pos > max_width:
                self.splitter.blockSignals(True)
                self.splitter.setSizes([max_width, self.splitter.width() - max_width])
                self.splitter.blockSignals(False)

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
        df = pd.read_csv('Users.csv', delimiter=';')

        # Filter data where role is 1
        self.df_filtered = df[df['role'] == 1]

        # Create a model for the QListView
        self.listModel = QtGui.QStandardItemModel()

        for row in self.df_filtered.itertuples():
            item = QtGui.QStandardItem(f"{row.name} {row.surname}")
            item.setData(row)
            self.listModel.appendRow(item)

        self.userListView.setModel(self.listModel)

    def displayUserDetails(self, selected, deselected):
        if selected.indexes():
            index = selected.indexes()[0]
            row = self.listModel.itemFromIndex(index).data()

            self.idLabel.setText(f"ID: {row.id}")
            self.nameLabel.setText(f"Name: {row.name}")
            self.surnameLabel.setText(f"Surname: {row.surname}")
            self.roleLabel.setText(f"Role: {row.role}")
            self.loginLabel.setText(f"Login: {row.login}")

            # Font size do momentu aż nie zacznie działać ResizeEvent
            geometry = self.frameGeometry()
            new_width = geometry.width()
            new_height = geometry.height()
            font_size = max(30, min(new_width, new_height) // 30)
            print(f"Window resized to {new_width}x{new_height}")
            print(f"Font size: {font_size}")

            font = QtGui.QFont()
            font.setPointSize(font_size)

            self.idLabel.setFont(font)
            self.nameLabel.setFont(font)
            self.surnameLabel.setFont(font)
            self.roleLabel.setFont(font)
            self.loginLabel.setFont(font)

    def on_resize(self, event: QtGui.QResizeEvent):
        super(Ui_UserListWindow, self).resize(event)
        new_width = self.width()
        new_height = self.height()
        font_size = max(10, min(new_width, new_height) // 30)
        print(f"Font size: {font_size}")

        font = QtGui.QFont()
        font.setPointSize(font_size)

        self.idLabel.setFont(font)
        self.nameLabel.setFont(font)
        self.surnameLabel.setFont(font)
        self.roleLabel.setFont(font)
        self.loginLabel.setFont(font)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    UserListWindow = Ui_UserListWindow()
    UserListWindow.show()
    sys.exit(app.exec_())
