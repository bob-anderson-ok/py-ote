import pyqtgraph as pg
import pyqtgraph.examples
from PyQt5 import QtGui


def excercise():
    _ = QtGui.QApplication([])

    win = QtGui.QMainWindow()
    win.resize(800, 800)
    imv = pg.ImageView()
    imv.show()

    pyqtgraph.examples.run()


if __name__ == "__main__":
    excercise()
