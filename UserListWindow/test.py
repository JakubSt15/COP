from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QLabel, QApplication
from PyQt5.QtCore import Qt

class ResizeWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Resize Event Example")

        # Create a central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Create layout and label
        layout = QVBoxLayout()
        self.status_label = QLabel("Window not resized yet")
        layout.addWidget(self.status_label)
        central_widget.setLayout(layout)

        # Connect resizeEvent to a function
        self.resizeEvent = self.on_resize

    def on_resize(self, event):
        # Get the new window size
        new_width = self.width()
        new_height = self.height()

        # Update the status label
        self.status_label.setText(f"Window resized to {new_width}x{new_height}")

if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    window = ResizeWindow()
    window.show()
    sys.exit(app.exec_())
