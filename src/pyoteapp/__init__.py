# Set HiDPI attributes on the QApplication class BEFORE any module import
# elsewhere in this package creates a QCoreApplication. Qt emits a warning
# ("Attribute Qt::AA_EnableHighDpiScaling must be set before QCoreApplication
# is created") and silently drops these if the QApplication already exists.
# Module-level imports in pyote.py (matplotlib Qt5Agg backend, pyqtgraph,
# etc.) can instantiate one, so __init__.py is the only place guaranteed to
# run first.
import PyQt5.QtCore
import PyQt5.QtWidgets

PyQt5.QtWidgets.QApplication.setAttribute(PyQt5.QtCore.Qt.AA_EnableHighDpiScaling, True)
PyQt5.QtWidgets.QApplication.setAttribute(PyQt5.QtCore.Qt.AA_UseHighDpiPixmaps, True)
