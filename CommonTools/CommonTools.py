from PyQt5.QtWidgets import QMessageBox, QApplication

def show_popup(title, message, icon=QMessageBox.Information):
    msg_box = QMessageBox()
    msg_box.setWindowTitle(title)
    msg_box.setText(message)
    msg_box.setIcon(icon)
    msg_box.exec_()