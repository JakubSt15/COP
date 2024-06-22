import json
from DoctorMenu import DoctorMenuList
from UserListWindow import UserListWindow
from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Options(object):
    def setupUi(self, AddUser):
        AddUser.setObjectName("AddUser")
        AddUser.resize(805, 400)

        self.centralwidget = QtWidgets.QWidget(AddUser)
        self.centralwidget.setObjectName("centralwidget")

        # Create a grid layout for arranging the buttons
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setContentsMargins(25, 25, 25, 25)
        self.gridLayout.setObjectName("gridLayout")

        # Title
        self.label_title = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(20)
        self.label_title.setFont(font)
        self.label_title.setObjectName("label_title")
        self.label_title.setText("Threshold Alarm Settings")
        self.gridLayout.addWidget(self.label_title, 0, 0, 1, 3)
        
        # High Risk Threshold
        self.label_high = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(16)
        self.label_high.setFont(font)
        self.label_high.setObjectName("label_high")
        self.gridLayout.addWidget(self.label_high, 1, 0, 1, 1)

        self.slider_high = QtWidgets.QSlider(self.centralwidget)
        self.slider_high.setOrientation(QtCore.Qt.Horizontal)
        self.slider_high.setRange(0, 100)
        self.slider_high.setObjectName("slider_high")
        self.slider_high.valueChanged.connect(self.update_label_high_value)
        self.gridLayout.addWidget(self.slider_high, 1, 1, 1, 1)

        self.label_high_value = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(16)
        self.label_high_value.setFont(font)
        self.label_high_value.setObjectName("label_high_value")
        self.gridLayout.addWidget(self.label_high_value, 1, 2, 1, 1)

        # Medium High Risk Threshold
        self.label_medium_high = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(16)
        self.label_medium_high.setFont(font)
        self.label_medium_high.setObjectName("label_medium_high")
        self.gridLayout.addWidget(self.label_medium_high, 2, 0, 1, 1)

        self.slider_medium_high = QtWidgets.QSlider(self.centralwidget)
        self.slider_medium_high.setOrientation(QtCore.Qt.Horizontal)
        self.slider_medium_high.setRange(0, 100)
        self.slider_medium_high.setObjectName("slider_medium_high")
        self.slider_medium_high.valueChanged.connect(self.update_label_medium_high_value)
        self.gridLayout.addWidget(self.slider_medium_high, 2, 1, 1, 1)

        self.label_medium_high_value = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(16)
        self.label_medium_high_value.setFont(font)
        self.label_medium_high_value.setObjectName("label_medium_high_value")
        self.gridLayout.addWidget(self.label_medium_high_value, 2, 2, 1, 1)

        # Medium Risk Threshold
        self.label_medium = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(16)
        self.label_medium.setFont(font)
        self.label_medium.setObjectName("label_medium")
        self.gridLayout.addWidget(self.label_medium, 3, 0, 1, 1)

        self.slider_medium = QtWidgets.QSlider(self.centralwidget)
        self.slider_medium.setOrientation(QtCore.Qt.Horizontal)
        self.slider_medium.setRange(0, 100)
        self.slider_medium.setObjectName("slider_medium")
        self.slider_medium.valueChanged.connect(self.update_label_medium_value)
        self.gridLayout.addWidget(self.slider_medium, 3, 1, 1, 1)

        self.label_medium_value = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(16)
        self.label_medium_value.setFont(font)
        self.label_medium_value.setObjectName("label_medium_value")
        self.gridLayout.addWidget(self.label_medium_value, 3, 2, 1, 1)

        # Medium Low Risk Threshold
        self.label_medium_low = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(16)
        self.label_medium_low.setFont(font)
        self.label_medium_low.setObjectName("label_medium_low")
        self.gridLayout.addWidget(self.label_medium_low, 4, 0, 1, 1)

        self.slider_medium_low = QtWidgets.QSlider(self.centralwidget)
        self.slider_medium_low.setOrientation(QtCore.Qt.Horizontal)
        self.slider_medium_low.setRange(0, 100)
        self.slider_medium_low.setObjectName("slider_medium_low")
        self.slider_medium_low.valueChanged.connect(self.update_label_medium_low_value)
        self.gridLayout.addWidget(self.slider_medium_low, 4, 1, 1, 1)

        self.label_medium_low_value = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(16)
        self.label_medium_low_value.setFont(font)
        self.label_medium_low_value.setObjectName("label_medium_low_value")
        self.gridLayout.addWidget(self.label_medium_low_value, 4, 2, 1, 1)

        # Low Risk Threshold
        self.label_low = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(16)
        self.label_low.setFont(font)
        self.label_low.setObjectName("label_low")
        self.gridLayout.addWidget(self.label_low, 5, 0, 1, 1)

        self.slider_low = QtWidgets.QSlider(self.centralwidget)
        self.slider_low.setOrientation(QtCore.Qt.Horizontal)
        self.slider_low.setRange(0, 100)
        self.slider_low.setObjectName("slider_low")
        self.slider_low.valueChanged.connect(self.update_label_low_value)
        self.gridLayout.addWidget(self.slider_low, 5, 1, 1, 1)

        self.label_low_value = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(16)
        self.label_low_value.setFont(font)
        self.label_low_value.setObjectName("label_low_value")
        self.gridLayout.addWidget(self.label_low_value, 5, 2, 1, 1)

        # Very Low Risk Threshold
        self.label_very_low = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(16)
        self.label_very_low.setFont(font)
        self.label_very_low.setObjectName("label_very_low")
        self.gridLayout.addWidget(self.label_very_low, 6, 0, 1, 1)

        self.slider_very_low = QtWidgets.QSlider(self.centralwidget)
        self.slider_very_low.setOrientation(QtCore.Qt.Horizontal)
        self.slider_very_low.setRange(0, 100)
        self.slider_very_low.setObjectName("slider_very_low")
        self.slider_very_low.valueChanged.connect(self.update_label_very_low_value)
        self.gridLayout.addWidget(self.slider_very_low, 6, 1, 1, 1)

        self.label_very_low_value = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(16)
        self.label_very_low_value.setFont(font)
        self.label_very_low_value.setObjectName("label_very_low_value")
        self.gridLayout.addWidget(self.label_very_low_value, 6, 2, 1, 1)

        # Enable Warning Checkbox and its Input
        self.enable_warning_layout = QtWidgets.QHBoxLayout()
        self.enable_warning = QtWidgets.QCheckBox(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(16)
        self.enable_warning.setFont(font)
        self.enable_warning.setObjectName("enable_warning")
        self.enable_warning_layout.addWidget(self.enable_warning)

        self.warning_value = QtWidgets.QSlider(self.centralwidget)
        self.warning_value.setOrientation(QtCore.Qt.Horizontal)
        self.warning_value.setRange(0, 100)
        self.warning_value.setObjectName("warning_value")
        self.warning_value.valueChanged.connect(self.update_label_warning_value)
        self.enable_warning_layout.addWidget(self.warning_value)

        self.label_warning_value = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(16)
        self.label_warning_value.setFont(font)
        self.label_warning_value.setObjectName("label_warning_value")
        self.enable_warning_layout.addWidget(self.label_warning_value)

        self.gridLayout.addLayout(self.enable_warning_layout, 7, 0, 1, 3)

        # Save Button
        self.Save = QtWidgets.QPushButton(self.centralwidget)
        self.Save.setObjectName("Save")
        self.gridLayout.addWidget(self.Save, 8, 0, 1, 1)

        # Close Button
        self.Close = QtWidgets.QPushButton(self.centralwidget)
        self.Close.setObjectName("Close")
        self.gridLayout.addWidget(self.Close, 8, 1, 1, 1)

        # Set to default values
        self.SetToDefault = QtWidgets.QPushButton(self.centralwidget)
        self.SetToDefault.setObjectName("SetToDefault")
        self.gridLayout.addWidget(self.SetToDefault, 8, 2, 1, 1)

        AddUser.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(AddUser)
        self.menubar.setObjectName("menubar")
        AddUser.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(AddUser)
        self.statusbar.setObjectName("statusbar")
        AddUser.setStatusBar(self.statusbar)


        self.read_thresholds()
        self.retranslateUi(AddUser)
        QtCore.QMetaObject.connectSlotsByName(AddUser)

        self.Save.clicked.connect(self.onSaveClicked)
        self.Close.clicked.connect(self.onCloseClicked)
        self.SetToDefault.clicked.connect(self.onSetToDefaultClicked)
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
        self.Save.setStyleSheet(button_style)
        self.Close.setStyleSheet(button_style_reversed)
        self.SetToDefault.setStyleSheet(button_style)

        slider_style = """
            QSlider {
                min-height: 20px;
                max-height: 20px;
            }
            QSlider::groove:horizontal {
                border: 1px solid #bbb;
                background: white;
                height: 10px;
                border-radius: 4px;
            }
            QSlider::sub-page:horizontal {
                background: #4CAF50;
                border: 1px solid #777;
                height: 10px;
                border-radius: 4px;
            }
            QSlider::add-page:horizontal {
                background: #fff;
                border: 1px solid #777;
                height: 10px;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #4CAF50;
                border: 1px solid #777;
                width: 15px;
                margin-top: -5px;
                margin-bottom: -5px;
                border-radius: 4px;
            }
            QSlider::handle:horizontal:hover {
                background: #45a049;
            }
        """
        self.slider_high.setStyleSheet(slider_style)
        self.slider_medium_high.setStyleSheet(slider_style)
        self.slider_medium.setStyleSheet(slider_style)
        self.slider_medium_low.setStyleSheet(slider_style)
        self.slider_low.setStyleSheet(slider_style)
        self.slider_very_low.setStyleSheet(slider_style)
        self.warning_value.setStyleSheet(slider_style)

        label_style = """
            QLabel {
                font-size: 16px;
            }
        """
        self.label_high.setStyleSheet(label_style)
        self.label_medium_high.setStyleSheet(label_style)
        self.label_medium.setStyleSheet(label_style)
        self.label_medium_low.setStyleSheet(label_style)
        self.label_low.setStyleSheet(label_style)
        self.label_very_low.setStyleSheet(label_style)
        self.label_warning_value.setStyleSheet(label_style)
        checkbox_style = """
            QCheckBox {
                font-size: 16px;
            }
        """
        self.enable_warning.setStyleSheet(checkbox_style)

    def retranslateUi(self, AddUser):
        _translate = QtCore.QCoreApplication.translate
        AddUser.setWindowTitle(_translate("AddUser", "Threshold Settings"))
        self.label_high.setText(_translate("AddUser", "High Risk Threshold:"))
        self.label_medium_high.setText(_translate("AddUser", "Medium High Risk Threshold:"))
        self.label_medium.setText(_translate("AddUser", "Medium Risk Threshold:"))
        self.label_medium_low.setText(_translate("AddUser", "Medium Low Risk Threshold:"))
        self.label_low.setText(_translate("AddUser", "Low Risk Threshold:"))
        self.label_very_low.setText(_translate("AddUser", "Very Low Risk Threshold:"))
        self.enable_warning.setText(_translate("AddUser", "Enable Pop-up Warning"))
        self.Save.setText(_translate("AddUser", "Save"))
        self.Close.setText(_translate("AddUser", "Close"))
        self.SetToDefault.setText(_translate("AddUser", "Set to Default"))

    def update_label_high_value(self):
        self.label_high_value.setText(str(self.slider_high.value())+"%")

    def update_label_medium_high_value(self):
        self.label_medium_high_value.setText(str(self.slider_medium_high.value())+"%")

    def update_label_medium_value(self):
        self.label_medium_value.setText(str(self.slider_medium.value())+"%")

    def update_label_medium_low_value(self):
        self.label_medium_low_value.setText(str(self.slider_medium_low.value())+"%")

    def update_label_low_value(self):
        self.label_low_value.setText(str(self.slider_low.value())+"%")

    def update_label_very_low_value(self):
        self.label_very_low_value.setText(str(self.slider_very_low.value())+"%")

    def update_label_warning_value(self):
        self.label_warning_value.setText(str(self.warning_value.value())+"%")


    def onSaveClicked(self):
        thresholds = {
            "high": self.slider_high.value(),
            "medium_high": self.slider_medium_high.value(),
            "medium": self.slider_medium.value(),
            "medium_low": self.slider_medium_low.value(),
            "low": self.slider_low.value(),
            "very_low": self.slider_very_low.value(),
            "warning": self.warning_value.value() if self.enable_warning.isChecked() else "Disabled"
        }
        with open("thresholds.json", "w") as jsonfile:
            json.dump(thresholds, jsonfile, indent=4)
        self.onCloseClicked()

    def onCloseClicked(self):
        try:
            QtWidgets.qApp.closeAllWindows()
            self.doctorMenu = QtWidgets.QMainWindow()
            self.secondWindow = DoctorMenuList.Ui_MainWindow()
            self.secondWindow.setupUi(self.doctorMenu)
            self.doctorMenu.show()
        except Exception as e:
            print(e)
    
    def onSetToDefaultClicked(self):
        self.slider_high.setValue(90)
        self.slider_medium_high.setValue(80)
        self.slider_medium.setValue(70)
        self.slider_medium_low.setValue(60)
        self.slider_low.setValue(50)
        self.slider_very_low.setValue(25)
        self.warning_value.setValue(75)
        self.enable_warning.setChecked(True)
        self.update_label_high_value()
        self.update_label_medium_high_value()
        self.update_label_medium_value()
        self.update_label_medium_low_value()
        self.update_label_low_value()
        self.update_label_very_low_value()
        self.update_label_warning_value()
    
    def read_thresholds(self):
        try:
            with open("thresholds.json", "r") as jsonfile:
                thresholds = json.load(jsonfile)
                self.slider_high.setValue(thresholds["high"])
                self.slider_medium_high.setValue(thresholds["medium_high"])
                self.slider_medium.setValue(thresholds["medium"])
                self.slider_medium_low.setValue(thresholds["medium_low"])
                self.slider_low.setValue(thresholds["low"])
                self.slider_very_low.setValue(thresholds["very_low"])
                self.warning_value.setValue(thresholds["warning"])
                self.enable_warning.setChecked(thresholds["warning"] != "Disabled")
                self.update_label_high_value()
                self.update_label_medium_high_value()
                self.update_label_medium_value()
                self.update_label_medium_low_value()
                self.update_label_low_value()
                self.update_label_very_low_value()
                self.update_label_warning_value()
        except Exception as e:
            print(e)

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    AddUser = QtWidgets.QMainWindow()
    ui = Ui_Options()
    ui.setupUi(AddUser)
    AddUser.show()
    sys.exit(app.exec_())
