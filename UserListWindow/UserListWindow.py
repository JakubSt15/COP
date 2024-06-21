import sys
import pandas as pd
from PyQt5 import QtCore, QtGui, QtWidgets
from DoctorMenu import DoctorMenuList
from AddUser import AddUser


class Ui_UserListWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

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

        # Create a horizontal layout for the buttons
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")

        # Create and add the close button
        self.CloseButton = QtWidgets.QPushButton(self.centralwidget)
        self.CloseButton.setObjectName("CloseButton")
        self.horizontalLayout.addWidget(self.CloseButton)

        # Create and add the remove button
        self.RemoveButton = QtWidgets.QPushButton(self.centralwidget)
        self.RemoveButton.setObjectName("RemoveButton")
        self.horizontalLayout.addWidget(self.RemoveButton)

        # Create and add the Add User button
        self.AddUserButton = QtWidgets.QPushButton(self.centralwidget)
        self.AddUserButton.setObjectName("AddUserButton")
        self.horizontalLayout.addWidget(self.AddUserButton)

        # Add the horizontal layout to the main vertical layout
        self.verticalLayout.addLayout(self.horizontalLayout)

        UserListWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(UserListWindow)
        self.statusbar.setObjectName("statusbar")
        UserListWindow.setStatusBar(self.statusbar)

        self.retranslateUi(UserListWindow)
        QtCore.QMetaObject.connectSlotsByName(UserListWindow)

        self.loadData()

        # Connect the close button
        self.CloseButton.clicked.connect(self.close_window)

        # Connect the remove button
        self.RemoveButton.clicked.connect(self.remove_selected_user)

        # Connect the add user button
        self.AddUserButton.clicked.connect(self.onAddUserCliced)

        # Connect the user list view selection
        self.userListView.selectionModel().selectionChanged.connect(self.displayUserDetails)

        # Set initial splitter position and restrict maximum width for the left pane
        self.splitter.setSizes([UserListWindow.width() // 4, UserListWindow.width() * 3 // 4])
        self.splitter.splitterMoved.connect(self.adjust_splitter)

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
        self.CloseButton.setStyleSheet(button_style_reversed)
        self.RemoveButton.setStyleSheet(button_style)
        self.AddUserButton.setStyleSheet(button_style)

        label_style = """
            font-size: 14px;
        """
        self.idLabel.setStyleSheet(label_style)
        self.nameLabel.setStyleSheet(label_style)
        self.surnameLabel.setStyleSheet(label_style)
        self.roleLabel.setStyleSheet(label_style)
        self.loginLabel.setStyleSheet(label_style)

        list_view_style = """
            background-color: #ecf0f1;
            border: 1px solid #bdc3c7;
            padding: 5px;
        """
        self.userListView.setStyleSheet(list_view_style)

        splitter_handle_style = """
            QSplitter::handle {
                background-color: #bdc3c7;
            }
        """
        self.splitter.setStyleSheet(splitter_handle_style)

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

    def onAddUserCliced(self):
        try:
            QtWidgets.qApp.closeAllWindows()
            self.addUser = QtWidgets.QMainWindow()
            self.secondWindow = AddUser.Ui_AddUser()
            self.secondWindow.setupUi(self.addUser)
            self.addUser.show()
        except Exception as e:
            print(e)

    def retranslateUi(self, UserListWindow):
        _translate = QtCore.QCoreApplication.translate
        UserListWindow.setWindowTitle(_translate("UserListWindow", "User List"))
        self.CloseButton.setText(_translate("UserListWindow", "Close"))
        self.RemoveButton.setText(_translate("UserListWindow", "Remove"))
        self.AddUserButton.setText(_translate("UserListWindow", "Add New User"))

    def loadData(self):
        # Load data from CSV file
        self.df = pd.read_csv('Users.csv', delimiter=';')

        # Filter data where role is 1
        self.df_filtered = self.df[self.df['role'] == 1]

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

    def remove_selected_user(self):
        selected_indexes = self.userListView.selectionModel().selectedIndexes()
        if not selected_indexes:
            return

        index = selected_indexes[0]
        row = self.listModel.itemFromIndex(index).data()

        # Remove from the list model
        self.listModel.removeRow(index.row())

        # Remove from the dataframe
        self.df = self.df[self.df['id'] != row.id]

        # Save the updated dataframe back to the CSV file
        self.df.to_csv('Users.csv', index=False, sep=';')

    def resizeEvent(self, event):
        super(Ui_UserListWindow, self).resizeEvent(event)
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
