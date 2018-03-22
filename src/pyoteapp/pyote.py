#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 20 15:32:13 2017

@author: Bob Anderson
"""

import datetime
import os
import sys

from math import trunc, floor
from copy import deepcopy

import numpy as np
import pyqtgraph as pg
import pyqtgraph.exporters as pex
import scipy.signal
from PyQt5 import QtCore
from PyQt5 import QtGui
from PyQt5.QtCore import QSettings, QPoint, QSize
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QDialog
from pyqtgraph import PlotWidget

from pyoteapp import version
from pyoteapp import fixedPrecision as fp
from pyoteapp import gui
from pyoteapp import timestampDialog
from pyoteapp.checkForNewerVersion import getMostRecentVersionOfPyote
from pyoteapp.checkForNewerVersion import upgradePyote
from pyoteapp.csvreader import readLightCurve
from pyoteapp.errorBarUtils import ciBars
from pyoteapp.errorBarUtils import createDurDistribution
from pyoteapp.errorBarUtils import edgeDistributionGenerator
from pyoteapp.noiseUtils import getCorCoefs
from pyoteapp.solverUtils import candidateCounter, solver, subFrameAdjusted
from pyoteapp.timestampUtils import convertTimeStringToTime
from pyoteapp.timestampUtils import convertTimeToTimeString
from pyoteapp.timestampUtils import getTimeStepAndOutliers
from pyoteapp.timestampUtils import manualTimeStampEntry
from pyoteapp.iterative_logl_functions import locate_event_from_d_and_r_ranges
from pyoteapp.iterative_logl_functions import find_best_event_from_min_max_size
from pyoteapp.iterative_logl_functions import find_best_r_only_from_min_max_size
from pyoteapp.iterative_logl_functions import find_best_d_only_from_min_max_size

cursorAlert = pyqtSignal()

# The gui module was created by typing
#    !pyuic5 simple_plot.ui -o gui.py
# in the IPython console while in pyoteapp directory

# The timestampDialog module was created by typing
#    !pyuic5 timestamp_dialog.ui -o timestampDialog.py
# in the IPython console while in pyoteapp directory

# Status of points and associated dot colors ---
SELECTED = 3  # big red
BASELINE = 2  # green
INCLUDED = 1  # blue
EXCLUDED = 0  # no dot

acfCoefThreshold = 0.05  # To match what is being done in R-OTE 4.5.4+

# There is a bug in pyqtgraph ImageExpoter, probably caused by new versions of PyQt5 returning
# float values for image rectangles.  Those floats were being given to numpy to create a matrix,
# and that was raising an exception.  Below is my 'cure', effected by overriding the internal
# methods of ImageExporter that manipulate width and height


class FixedImageExporter(pex.ImageExporter):
    def __init__(self, item):
        pex.ImageExporter.__init__(self, item)

    def makeWidthHeightInts(self):
        self.params['height'] = int(self.params['height'] + 1)  # The +1 is needed
        self.params['width'] = int(self.params['width'] + 1)

    def widthChanged(self):
        sr = self.getSourceRect()
        ar = float(sr.height()) / sr.width()
        self.params.param('height').setValue(int(self.params['width'] * ar),
                                             blockSignal=self.heightChanged)

    def heightChanged(self):
        sr = self.getSourceRect()
        ar = float(sr.width()) / sr.height()
        self.params.param('width').setValue(int(self.params['height'] * ar),
                                            blockSignal=self.widthChanged)


class Signal:
    def __init__(self):
        self.__subscribers = []

    def emit(self, *args, **kwargs):
        for subs in self.__subscribers:
            subs(*args, **kwargs)

    def connect(self, func):
        self.__subscribers.append(func)


mouseSignal = Signal()


class CustomViewBox(pg.ViewBox):
    def __init__(self, *args, **kwds):
        pg.ViewBox.__init__(self, *args, **kwds)
        self.setMouseMode(self.RectMode)
        
    # re-implement right-click to zoom out
    def mouseClickEvent(self, ev):
        if ev.button() == QtCore.Qt.RightButton:
            self.autoRange()
            mouseSignal.emit()

    def mouseDragEvent(self, ev, axis=None):
        if ev.button() == QtCore.Qt.RightButton:
            ev.ignore()
        else:
            pg.ViewBox.mouseDragEvent(self, ev, axis)
            mouseSignal.emit()


class TSdialog(QDialog, timestampDialog.Ui_manualTimestampDialog):
    def __init__(self):
        super(TSdialog, self).__init__()
        self.setupUi(self)


class SimplePlot(QtGui.QMainWindow, gui.Ui_MainWindow):
    def __init__(self):
        super(SimplePlot, self).__init__()

        # Change pyqtgraph plots to be black on white
        pg.setConfigOption('background', (255, 255, 255))  # Do before any widgets drawn
        pg.setConfigOption('foreground', 'k')  # Do before any widgets drawn
        
        self.setupUi(self)

        self.setWindowTitle('PYOTE  Version: ' + version.version())

        # Button: Info
        self.infoButton.clicked.connect(self.openHelpFile)

        # Button: Read light curve
        self.readData.clicked.connect(self.readDataFromFile)
        
        # CheckBox: Show secondary star
        self.showSecondaryCheckBox.clicked.connect(self.toggleDisplayOfSecondaryStar)

        # Checkbox: Show timestamp errors
        self.showTimestampErrors.clicked.connect(self.toggleDisplayOfTimestampErrors)

        # QSpinBox
        self.secondarySelector.valueChanged.connect(self.changeSecondary)

        # Button: Trim/Select data points
        self.setDataLimits.clicked.connect(self.doTrim)
        
        # Button: Smooth secondary
        self.smoothSecondaryButton.clicked.connect(self.smoothRefStar)

        # QLineEdit: window size for secondary smoothing
        self.numSmoothPointsEdit.editingFinished.connect(self.smoothRefStar)

        # Button: Normalize around selected point
        self.normalizeButton.clicked.connect(self.normalize)

        # Button: Do block integration
        self.doBlockIntegration.clicked.connect(self.doIntegration)
        
        # Button: Perform baseline noise analysis
        # self.doNoiseAnalysis.clicked.connect(self.processBaselineNoise)
        
        # Button: Perform event noise analysis (determine sigmaA only)
        # self.computeSigmaA.clicked.connect(self.processEventNoise)
        
        # Button: Mark D zone
        self.markDzone.clicked.connect(self.showDzone)
        
        # Button: Mark R zone
        self.markRzone.clicked.connect(self.showRzone)
        
        # Button: Locate event
        self.locateEvent.clicked.connect(self.findEvent)
        
        # Button: Cancel operation
        self.cancelButton.clicked.connect(self.requestCancel)
        
        # Button: Calculate error bars
        self.calcErrBars.clicked.connect(self.computeErrorBars)
        
        # Button: Write error bar plot to file
        self.writeBarPlots.clicked.connect(self.exportBarPlots)

        # Button: Write graphic to file
        self.writePlot.clicked.connect(self.exportGraphic)
        
        # Button: Start over
        self.startOver.clicked.connect(self.restart)
        
        # Set up handlers for clicks on table view of data
        self.table.cellClicked.connect(self.cellClick)
        self.table.verticalHeader().sectionClicked.connect(self.rowClick)
        
        # Re-instantiate mainPlot             Note: examine gui.py
        # to get this right after a re-layout !!!!  self.widget changes sometimes
        # as does horizontalLayout_?

        oldMainPlot = self.mainPlot
        self.mainPlot = PlotWidget(self.widget,
                                   viewBox=CustomViewBox(border=(255, 255, 255)),
                                   enableMenu=False)
        self.mainPlot.setObjectName("mainPlot")
        self.horizontalLayout_11.addWidget(self.mainPlot, stretch=1)

        oldMainPlot.setParent(None)

        self.mainPlot.scene().sigMouseMoved.connect(self.reportMouseMoved)
        self.verticalCursor = pg.InfiniteLine(angle=90, movable=False, pen=(0, 0, 0))
        self.mainPlot.addItem(self.verticalCursor)
        self.blankCursor = True
        self.mainPlot.viewport().setProperty("cursor", QtGui.QCursor(QtCore.Qt.BlankCursor))
        mouseSignal.connect(self.mouseEvent)

        # Set up handler for clicks on data plot
        self.mainPlot.scene().sigMouseClicked.connect(self.processClick)
        self.mainPlotViewBox = self.mainPlot.getViewBox()
        self.mainPlotViewBox.rbScaleBox.setPen(pg.mkPen((255, 0, 0), width=2))
        self.mainPlotViewBox.rbScaleBox.setBrush(pg.mkBrush(None))
        self.mainPlot.hideButtons()
        self.mainPlot.showGrid(y=True, alpha=1.0)

        self.initializeTableView()  # Mostly just establishes column headers
        
        # Open (or create) file for holding 'sticky' stuff
        self.settings = QSettings('simple-ote.ini', QSettings.IniFormat)
        self.settings.setFallbacksEnabled(False)
        
        # Use 'sticky' settings to size and position the main screen
        self.resize(self.settings.value('size', QSize(800, 800)))
        self.move(self.settings.value('pos', QPoint(50, 50)))
     
        self.outliers = []
        self.logFile = ''
        self.initializeVariablesThatDontDependOnAfile()

        self.checkForNewVersion()

        self.only_new_solver_wanted = True

    def openHelpFile(self):
        helpFilePath = os.path.join(os.path.split(__file__)[0], 'pyote-info.pdf')

        url = QtCore.QUrl.fromLocalFile(helpFilePath)
        fileOpened = QtGui.QDesktopServices.openUrl(url)

        if not fileOpened:
            self.showMsg('Failed to open pyote-info.pdf', bold=True, color='red', blankLine=False)
            self.showMsg('Location of pyote information file: ' + helpFilePath, bold=True, color='blue')

    def mouseEvent(self):
        if not self.blankCursor:
            # self.showMsg('Mouse event')
            self.blankCursor = True
            self.mainPlot.viewport().setProperty("cursor", QtGui.QCursor(QtCore.Qt.BlankCursor))

    def keyPressEvent(self, ev):
        if ev.key() == QtCore.Qt.Key_Shift:
            # self.showMsg('Shift key pressed')
            if self.blankCursor:
                self.mainPlot.viewport().setProperty("cursor", QtGui.QCursor(QtCore.Qt.ArrowCursor))
                self.blankCursor = False
            else:
                self.blankCursor = True
                self.mainPlot.viewport().setProperty("cursor", QtGui.QCursor(QtCore.Qt.BlankCursor))

    @staticmethod
    def timestampListIsEmpty(alist):
        ans = True
        for item in alist:
            # Limovie = '[::]'   Tangra = ''  R-OTE = '[NA]' or 'NA'
            if item == '' or item == '[::]' or item == '[NA]' or item == 'NA':
                pass
            else:
                ans = False
                break
        return ans

    def changeSecondary(self):
        selText = self.secondarySelector.text()
        self.showMsg('Secondary reference ' + selText + ' selected.')
        refNum = int(selText)
        refStar = None
        if refNum == 1:
            refStar = [float(item) for item in self.secondary]
        if refNum == 2:
            refStar = [float(item) for item in self.ref2]
        if refNum == 3:
            refStar = [float(item) for item in self.ref3]

        self.smoothSecondary = []
        self.yRefStar = np.array(refStar)
        self.yRefStarCopy = np.array(refStar)
        self.reDrawMainPlot()
        self.mainPlot.autoRange()

    def installLatestVersion(self):
        pipResult = upgradePyote()
        for line in pipResult:
            self.showMsg(line, blankLine=False)

        self.showMsg('', blankLine=False)
        self.showMsg('The new version is installed but not yet running.', color='red', bold=True)
        self.showMsg('Close and reopen pyote to start the new version running.', color='red', bold=True)

    def checkForNewVersion(self):
        gotVersion, latestVersion = getMostRecentVersionOfPyote()
        if gotVersion:
            if latestVersion <= version.version():
                self.showMsg('You are running the most recent version of pyote', color='red', bold=True)
            else:
                self.showMsg('Version ' + latestVersion + ' is available', color='red', bold=True)
                if self.queryWhetherNewVersionShouldBeInstalled() == QMessageBox.Yes:
                    self.showMsg('You have opted to install latest version of PYOTE')
                    self.installLatestVersion()
                else:
                    self.showMsg('You have declined the opportunity to install latest PYOTE')
        else:
            self.showMsg(latestVersion, color='red', bold=True)

    @staticmethod
    def queryWhetherNewVersionShouldBeInstalled():
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Question)
        msg.setText('A newer version of PYOTE is available. Do you wish to install it?')
        msg.setWindowTitle('Get latest version of PYOTE query')
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        retval = msg.exec_()
        return retval

    def reportMouseMoved(self, pos):
        # self.showMsg(str(pos.x()))
        mousePoint = self.mainPlotViewBox.mapSceneToView(pos)
        # self.showMsg(str(mousePoint.x()))
        self.verticalCursor.setPos(round(mousePoint.x()))

    def writeDefaultGraphicsPlots(self):
        self.graphicFile, _ = os.path.splitext(self.filename)

        exporter = FixedImageExporter(self.dBarPlotItem)
        exporter.makeWidthHeightInts()
        targetFileD = self.graphicFile + '.D.PYOTE.png'
        exporter.export(targetFileD)

        exporter = FixedImageExporter(self.durBarPlotItem)
        exporter.makeWidthHeightInts()
        targetFileDur = self.graphicFile + '.R-D.PYOTE.png'
        exporter.export(targetFileDur)

        exporter = FixedImageExporter(self.mainPlot.getPlotItem())
        exporter.makeWidthHeightInts()
        targetFile = self.graphicFile + '.PYOTE.png'
        exporter.export(targetFile)

    def exportBarPlots(self):
        if self.dBarPlotItem is None:
            self.showInfo('No error bar plots available yet')
            return

        _, name = os.path.split(self.filename)
        name = self.removeCsvExtension(name)

        myOptions = QFileDialog.Options()
        myOptions |= QFileDialog.DontConfirmOverwrite
        myOptions |= QFileDialog.DontUseNativeDialog
        myOptions |= QFileDialog.ShowDirsOnly

        self.graphicFile, _ = QFileDialog.getSaveFileName(
                self,                                      # parent
                "Select directory/modify filename (png will be appended for you)",     # title for dialog
                self.settings.value('lightcurvedir', "") + '/' + name,  # starting directory
                # "csv files (*.csv)", options=myOptions)
                "png files (*.png)", options=myOptions)
        
        if self.graphicFile:
            self.graphicFile = self.removeCsvExtension(self.graphicFile)
            exporter = FixedImageExporter(self.dBarPlotItem)
            exporter.makeWidthHeightInts()
            targetFileD = self.graphicFile + '.D.PYOTE.png'
            exporter.export(targetFileD)
            
            exporter = FixedImageExporter(self.durBarPlotItem)
            exporter.makeWidthHeightInts()
            targetFileDur = self.graphicFile + '.R-D.PYOTE.png'
            exporter.export(targetFileDur)
            
            self.showInfo('Wrote to: \r\r' + targetFileD + ' \r\r' + targetFileDur)

    @staticmethod
    def removeCsvExtension(path):
        base, ext = os.path.splitext(path)
        if ext == '.csv':
            return base
        else:
            return path

    def exportGraphic(self):

        _, name = os.path.split(self.filename)
        name = self.removeCsvExtension(name)

        myOptions = QFileDialog.Options()
        myOptions |= QFileDialog.DontConfirmOverwrite
        myOptions |= QFileDialog.DontUseNativeDialog
        myOptions |= QFileDialog.ShowDirsOnly

        self.graphicFile, _ = QFileDialog.getSaveFileName(
                self,                                      # parent
                "Select directory/modify filename (png will be appended for you)",    # title for dialog
                self.settings.value('lightcurvedir', "") + '/' + name,  # starting directory
                "png files (*.png)", options=myOptions)

        if self.graphicFile:
            self.graphicFile = self.removeCsvExtension(self.graphicFile)
            exporter = FixedImageExporter(self.mainPlot.getPlotItem())
            exporter.makeWidthHeightInts()
            targetFile = self.graphicFile + '.PYOTE.png'
            exporter.export(targetFile)
            self.showInfo('Wrote to: \r\r' + targetFile)
        
    def initializeVariablesThatDontDependOnAfile(self):

        self.normalized = False
        self.timesAreValid = False
        self.selectedPoints = {}  # Clear/declare 'selected points' dictionary
        self.baselineXvals = []
        self.baselineYvals = []
        self.blockSize = 1  # Contains the number of points used in block integration
        self.solution = None
        self.firstPassSolution = None
        self.secondPassSolution = None
        self.smoothSecondary = []
        self.corCoefs = []
        self.numPtsInCorCoefs = 0
        self.Doffset = 1  # Offset (in readings) between D and 'start of exposure'
        self.Roffset = 1  # Offset (in readings) between R and 'start of exposure'
        self.sigmaB = None
        self.sigmaA = None
        self.A = None
        self.B = None
        self.snrB = None
        self.snrA = None
        self.dRegion = None
        self.rRegion = None
        self.dLimits = []
        self.rLimits = []
        self.minEvent = None
        self.maxEvent = None
        self.solution = None
        self.eventType = 'none'
        self.cancelRequested = False
        self.deltaDlo68 = 0
        self.deltaDlo95 = 0
        self.deltaDhi68 = 0
        self.deltaDhi95 = 0
        self.deltaRlo68 = 0
        self.deltaRlo95 = 0
        self.deltaRhi68 = 0
        self.deltaRhi95 = 0
        self.deltaDurlo68 = 0
        self.deltaDurlo95 = 0
        self.deltaDurhi68 = 0
        self.deltaDurhi95 = 0
        self.plusD = None
        self.minusD = None
        self.plusR = None
        self.minusR = None
        self.dBarPlotItem = None
        self.durBarPlotItem = None
        self.errBarWin = None

    def requestCancel(self):
        self.cancelRequested = True
        # The following line was just used to test uncaught exception handling
        # raise Exception('The requestCancel devil made me do it')
        
    def showDzone(self):
        # If the user has not selected any points, we remove any dRegion that may
        # have been present
        if len(self.selectedPoints) == 0:
            self.dRegion = None
            self.dLimits = None
            self.reDrawMainPlot()
            return
        
        if len(self.selectedPoints) != 2:
            self.showInfo('Exactly two points must be selected for this operation.')
            return
        selIndices = [key for key, _ in self.selectedPoints.items()]
        selIndices.sort()
        
        leftEdge = int(min(selIndices))
        rightEdge = int(max(selIndices))

        if self.rLimits:
            if rightEdge > self.rLimits[0] - 2:  # Enforce at least 1 'a' point
                rightEdge = self.rLimits[0] - 2

            if self.rLimits[1] < self.right:  # At least 1 'b' point is present
                if leftEdge < self.left:
                    leftEdge = self.left
            else:
                if leftEdge < self.left + 1:
                    leftEdge = self.left + 1
        else:
            if rightEdge >= self.right - 1:
                rightEdge = self.right - 1    # Enforce at least 1 'a' point
            if leftEdge < self.left + 1:
                leftEdge = self.left + 1  # Enforce at least 1 'b' point

        if rightEdge < self.left or rightEdge <= leftEdge:
            self.removePointSelections()
            self.reDrawMainPlot()
            return

        self.setDataLimits.setEnabled(False)

        if self.only_new_solver_wanted:
            self.locateEvent.setEnabled(True)

        self.dLimits = [leftEdge, rightEdge]
        
        if self.rLimits:
            self.DandR.setChecked(True)
        else:
            self.Donly.setChecked(True)
            
        self.dRegion = pg.LinearRegionItem(
                [leftEdge, rightEdge], movable=False, brush=(0, 200, 0, 50))
        self.mainPlot.addItem(self.dRegion)

        self.showMsg('D zone selected: ' + str([leftEdge, rightEdge]))
        self.removePointSelections()
        self.reDrawMainPlot()
        
    def showRzone(self):
        # If the user has not selected any points, we remove any rRegion that may
        # have been present
        if len(self.selectedPoints) == 0:
            self.rRegion = None
            self.rLimits = None
            self.reDrawMainPlot()
            return
        
        if len(self.selectedPoints) != 2:
            self.showInfo('Exactly two points must be selected for this operation.')
            return
        selIndices = [key for key, _ in self.selectedPoints.items()]
        selIndices.sort()
        
        leftEdge = int(min(selIndices))
        rightEdge = int(max(selIndices))

        if self.dLimits:
            if leftEdge < self.dLimits[1] + 2:
                leftEdge = self.dLimits[1] + 2  # Enforce at least 1 'a' point
            if self.dLimits[0] == self.left:
                if rightEdge >= self.right:
                    rightEdge = self.right - 1  # Enforce at least 1 'b' point
            else:
                if rightEdge >= self.right:
                    rightEdge = self.right
        else:
            if rightEdge >= self.right - 1:
                rightEdge = self.right - 1  # Enforce 1 'a' (for r-only search)
            if leftEdge < self.left + 1:
                leftEdge = self.left + 1    # Enforce  1 'b' point

        if rightEdge <= leftEdge:
            self.removePointSelections()
            self.reDrawMainPlot()
            return

        self.setDataLimits.setEnabled(False)

        if self.only_new_solver_wanted:
            self.locateEvent.setEnabled(True)

        self.rLimits = [leftEdge, rightEdge]
        
        if self.dLimits:
            self.DandR.setChecked(True)
        else:
            self.Ronly.setChecked(True)
            
        self.rRegion = pg.LinearRegionItem(
                [leftEdge, rightEdge], movable=False, brush=(200, 0, 0, 50))
        self.mainPlot.addItem(self.rRegion)
        
        self.showMsg('R zone selected: ' + str([leftEdge, rightEdge]))
        self.removePointSelections()
        self.reDrawMainPlot()
        
    def normalize(self):
        if len(self.selectedPoints) != 1:
            self.showInfo('A single point must be selected for this operation.' +
                          'That point will retain its value while all other points ' +
                          'are scaled (normalized) around it.')
            return
        
        selIndices = [key for key, value in self.selectedPoints.items()]
        index = selIndices[0]
        # self.showMsg('Index: ' + str(index) )
        # Reminder: the smoothSecondary[] only cover self.left to self.right inclusive,
        # hence the index manipulation in the following code
        ref = self.smoothSecondary[int(index)-self.left]
        
        for i in range(self.left, self.right+1):
            try:
                self.yValues[i] = (ref * self.yValues[i]) / self.smoothSecondary[i-self.left]
            except Exception as e:
                self.showMsg(str(e))

        self.showMsg('Light curve normalized to secondary around point ' + str(index))
        self.normalized = True
        
        self.removePointSelections()
        self.normalizeButton.setEnabled(False)
        self.smoothSecondaryButton.setEnabled(False)
        self.numSmoothPointsEdit.setEnabled(False)
        self.setDataLimits.setEnabled(False)
        self.reDrawMainPlot()
        
    def smoothRefStar(self):
        if (self.right - self.left) < 4:
            self.showInfo('The smoothing algorithm requires a minimum selection of 5 points')
            return
        
        y = [self.yRefStar[i] for i in range(self.left, self.right+1)]

        userSpecedWindow = 101

        numPts = self.numSmoothPointsEdit.text().strip()
        if numPts:
            if not numPts.isnumeric():
                self.showInfo('Invalid entry for smoothing window size - defaulting to 101')
            else:
                userSpecedWindow = int(numPts)
                if userSpecedWindow < 5:
                    self.showInfo('smoothing window must be size 5 or greater - defaulting to 101')
                    userSpecedWindow = 101

        window = None
        try:
            if len(y) > userSpecedWindow:
                window = userSpecedWindow
            else:
                window = len(y)

            # Enforce the odd window size required by savgol_filter()
            if window % 2 == 0:
                window -= 1

            filteredY = scipy.signal.savgol_filter(np.array(y), window, 3)
            self.smoothSecondary = scipy.signal.savgol_filter(filteredY, window, 3)
            self.reDrawMainPlot()
        except Exception as e:
            self.showMsg(str(e))

        self.showMsg('Smoothing of secondary star light curve performed with window size: %i' % window)

        # self.smoothSecondaryButton.setEnabled(False)
        # self.numSmoothPointsEdit.setEnabled(False)
        self.normalizeButton.setEnabled(True)

    def toggleDisplayOfTimestampErrors(self):
        self.reDrawMainPlot()
        self.mainPlot.autoRange()

    def toggleDisplayOfSecondaryStar(self):
        self.reDrawMainPlot()
        self.mainPlot.autoRange()
        
    def showInfo(self, stuffToSay):
        QMessageBox.information(self, 'General information', stuffToSay)

    def showQuery(self, question, title=''):
        msgBox = QMessageBox(self)
        msgBox.setIcon(QMessageBox.Question)
        msgBox.setText(question)
        msgBox.setWindowTitle(title)
        msgBox.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msgBox.setDefaultButton(QMessageBox.Yes)
        self.queryRetVal = msgBox.exec_()
        
    def doIntegration(self):
        if len(self.selectedPoints) != 2:
            self.showInfo('Exactly two points must be selected for a block integration')
            return
        
        if self.outliers:
            self.showInfo('This data set contains some erroneous time steps, which have ' +
                          'been marked with red lines.  Best practice is to ' +
                          'choose an integration block that is ' +
                          'positioned in an unmarked region, hopefully containing ' +
                          'the "event".  Block integration ' +
                          'proceeds to the left and then to the right of the marked block.')
            
        selPts = [key for key in self.selectedPoints.keys()]
        self.removePointSelections()
        left = min(selPts)
        right = max(selPts)

        # Time to do the work
        p0 = left
        span = right - left + 1  # Number of points in integration block
        self.blockSize = span
        newFrame = []
        newTime = []
        newVal = []
        newRef = []
        
        p = p0 - span  # Start working toward the left
        while p > 0:
            avg = np.mean(self.yValues[p:(p+span)])
            newVal.insert(0, avg)

            if len(self.yRefStar) > 0:
                avg = np.mean(self.yRefStar[p:(p+span)])
                newRef.insert(0, avg)

            newFrame.insert(0, self.yFrame[p])
            newTime.insert(0, self.yTimes[p])
            p = p - span
            
        p = p0  # Start working toward the right
        while p < self.dataLen - span:
            avg = np.mean(self.yValues[p:(p+span)])
            newVal.append(avg)

            if len(self.yRefStar) > 0:
                avg = np.mean(self.yRefStar[p:(p+span)])
                newRef.append(avg)

            newFrame.append(self.yFrame[p])
            newTime.append(self.yTimes[p])
            p = p + span
            
        self.dataLen = len(newVal)
        
        # auto-select all points
        self.left = 0
        self.right = self.dataLen - 1
        
        self.yValues = np.array(newVal)
        if len(newRef) > 0:
            self.yRefStar = np.array(newRef)
        else:
            self.yRefStar = []

        self.yTimes = newTime[:]
        self.yFrame = newFrame[:]
        # noinspection PyUnusedLocal
        self.yStatus = [1 for _i in range(self.dataLen)]
        self.fillTableViewOfData()
        
        selPts.sort()
        self.showMsg('Block integration started at entry ' + str(selPts[0]) +
                     ' with block size of ' + str(selPts[1]-selPts[0]+1) + ' readings')
        
        self.timeDelta, self.outliers, self.errRate = getTimeStepAndOutliers(self.yTimes)
        self.showMsg('timeDelta: ' + fp.to_precision(self.timeDelta, 6) + ' seconds per block', blankLine=False)
        self.showMsg('timestamp error rate: ' + fp.to_precision(100 * self.errRate, 2) + '%')

        self.illustrateTimestampOutliers()
        
        self.doBlockIntegration.setEnabled(False)
        self.reDrawMainPlot()
        self.mainPlot.autoRange()

    def togglePointSelected(self, index):
        if self.yStatus[index] != 3:
            # Save current status for possible undo (a later click)
            self.selectedPoints[index] = self.yStatus[index]
            self.yStatus[index] = 3  # Set color to 'selected'
        else:
            # Restore previous status (when originally clicked)
            self.yStatus[index] = self.selectedPoints[index]
            del (self.selectedPoints[index])
        self.reDrawMainPlot()  # Redraw plot to show selection change

    def processClick(self, event):
        # Don't allow mouse clicks to select points unless the cursor is blank
        if self.blankCursor:
            # This try/except handles case where user clicks in plot area before a
            # plot has been drawn.
            try:
                mousePoint = self.mainPlotViewBox.mapSceneToView(event.scenePos())
                index = round(mousePoint.x())
                if index in range(self.dataLen):
                    if event.button() == 1:  # left button clicked?
                        if index < self.left:
                            index = self.left
                        if index > self.right:
                            index = self.right
                        self.togglePointSelected(index)
                    # Move the table view of data so that clicked point data is visible
                    self.table.setCurrentCell(index, 0)
                else:
                    pass  # Out of bounds clicks simply ignored
            except AttributeError:
                pass
        
    def initializeTableView(self):
        self.table.clear()
        self.table.setColumnCount(4)
        self.table.setRowCount(3)
        colLabels = ['entry num', 'Frame num', 'timestamp', 'value']
        self.table.setHorizontalHeaderLabels(colLabels)
        
    def closeEvent(self, event):
        # Capture the close request and update 'sticky' settings
        self.settings.setValue('size', self.size())
        self.settings.setValue('pos', self.pos())
        
        curDateTime = datetime.datetime.today().ctime()
        self.showMsg('')
        self.showMsg('#' * 20 + ' Session ended: ' + curDateTime + '  ' + '#' * 20)
        
        if self.errBarWin:
            self.errBarWin.close()
        
        event.accept()
    
    def rowClick(self, row):
        entry = self.table.item(row, 0)
        self.highlightReading(int(entry.text()))
        
    def cellClick(self, row):
        entry = self.table.item(row, 0)
        self.togglePointSelected(int(entry.text()))

    def highlightReading(self, rdgNum):
        x = [rdgNum]
        y = [self.yValues[x]]
        self.reDrawMainPlot()
        self.mainPlot.plot(x, y, pen=None, symbol='o', symbolPen=(255, 0, 0),
                           symbolBrush=(255, 255, 0), symbolSize=10)
        
    def showMsg(self, msg, color=None, bold=False, blankLine=True):
        """ show standard output message """
        htmlmsg = msg
        if color:
            htmlmsg = '<font color=' + color + '>' + htmlmsg + '</font>'
        if bold:
            htmlmsg = '<b>' + htmlmsg + '</b>'
        htmlmsg = htmlmsg + '<br>'
        self.textOut.moveCursor(QtGui.QTextCursor.End)
        self.textOut.insertHtml(htmlmsg)
        if blankLine:
            self.textOut.insertHtml('<br>')
        self.textOut.ensureCursorVisible()
        if self.logFile:
            fileObject = open(self.logFile, 'a')
            fileObject.write(msg + '\n')
            if blankLine:
                fileObject.write('\n')
            fileObject.close()

    def reportSpecialProceduredUsed(self):
        if self.blockSize == 1:
            self.showMsg('This light curve has not been block integrated.',
                         color='blue', bold=True, blankLine=False)
        else:
            self.showMsg('Block integration of size %d has been applied to '
                         'this light curve.' %
                         self.blockSize, color='blue', bold=True, blankLine=False)

        if self.normalized:
            self.showMsg('This light curve has been normalized against a '
                         'reference star.',
                         color='blue', bold=True, blankLine=False)

        if not (self.left == 0 and self.right == self.dataLen - 1):
            self.showMsg('This light curve has been trimmed.',
                         color='blue', bold=True, blankLine=False)

        self.showMsg('', blankLine=False)

    def Dreport(self, deltaDhi, deltaDlo):
        D, _ = self.solution

        intD = int(D)  # So that we can do lookup in the data table

        noiseAsymmetry = self.snrA / self.snrB
        if (noiseAsymmetry > 0.7) and (noiseAsymmetry < 1.3):
            plusD = (deltaDhi - deltaDlo) / 2
            minusD = plusD
        else:
            plusD = -deltaDlo   # Deliberate 'inversion'
            minusD = deltaDhi   # Deliberate 'inversion'
         
        # Save these for the 'envelope' plotter
        self.plusD = plusD
        self.minusD = minusD

        entryNum = intD
        frameNum = float(self.yFrame[intD])
        # Dframe = D + frameNum - entryNum
        Dframe = (D - intD) * self.blockSize + intD + frameNum - entryNum
        self.showMsg('D: %.2f {+%.2f,-%.2f} (frame number)' % (Dframe, plusD * self.blockSize, minusD * self.blockSize),
                     blankLine=False)
        ts = self.yTimes[int(D)]
        time = convertTimeStringToTime(ts)
        adjTime = time + (D - int(D)) * self.timeDelta
        ts = convertTimeToTimeString(adjTime)
        self.showMsg('D: %s  {+%.4f,-%.4f} seconds' % 
                     (ts, plusD * self.timeDelta, minusD * self.timeDelta)
                     )
        return adjTime
        
    def Rreport(self, deltaRhi, deltaRlo):
        _, R = self.solution
        # if R: R = R - self.Roffset
        noiseAsymmetry = self.snrA / self.snrB
        if (noiseAsymmetry > 0.7) and (noiseAsymmetry < 1.3):
            plusR = (deltaRhi - deltaRlo) / 2
            minusR = plusR
        else:
            plusR = -deltaRlo  # Deliberate 'inversion'
            minusR = deltaRhi  # Deliberate 'inversion'
        
        # Save these for the 'envelope' plotter
        self.plusR = plusR
        self.minusR = minusR

        intR = int(R)
        entryNum = intR
        frameNum = float(self.yFrame[intR])
        # Rframe = R + frameNum - entryNum
        Rframe = (R - intR) * self.blockSize + intR + frameNum - entryNum
        self.showMsg('R: %.2f {+%.2f,-%.2f} (frame number)' % (Rframe, plusR * self.blockSize, minusR * self.blockSize),
                     blankLine=False)

        ts = self.yTimes[int(R)]
        time = convertTimeStringToTime(ts)
        adjTime = time + (R - int(R)) * self.timeDelta
        ts = convertTimeToTimeString(adjTime)
        self.showMsg('R: %s  {+%.4f,-%.4f} seconds' % 
                     (ts, plusR * self.timeDelta, minusR * self.timeDelta)
                     )
        return adjTime

    def confidenceIntervalReport(self, numSigmas, deltaDurhi, deltaDurlo, deltaDhi, deltaDlo,
                                 deltaRhi, deltaRlo):

        D, R = self.solution

        self.showMsg('B: %0.2f  {+/- %0.2f}' % (self.B, numSigmas * self.sigmaB / np.sqrt(self.nBpts)))
        self.showMsg('A: %0.2f  {+/- %0.2f}' % (self.A, numSigmas * self.sigmaA / np.sqrt(self.nApts)))

        if self.A > 0:
            self.showMsg('nominal magDrop: %0.2f' % ((np.log10(self.B) - np.log10(self.A)) * 2.5))
        else:
            self.showMsg('magDrop calculation not possible because A is negative')
        self.showMsg('snr: %0.2f' % self.snrB)

        if self.eventType == 'Donly':
            self.Dreport(deltaDhi, deltaDlo)
        elif self.eventType == 'Ronly':
            self.Rreport(deltaRhi, deltaRlo)
        elif self.eventType == 'DandR':
            Dtime = self.Dreport(deltaDhi, deltaDlo)
            Rtime = self.Rreport(deltaDhi, deltaDlo)
            plusDur = ((deltaDurhi - deltaDurlo) / 2)
            minusDur = plusDur
            self.showMsg('Duration (R - D): %.4f {+%.4f,-%.4f} readings' %
                         ((R - D) * self.blockSize, plusDur * self.blockSize, minusDur * self.blockSize),
                         blankLine=False)
            plusDur = ((deltaDurhi - deltaDurlo) / 2) * self.timeDelta
            minusDur = plusDur
            self.showMsg('Duration (R - D): %.4f {+%.4f,-%.4f} seconds' %
                         (Rtime - Dtime, plusDur, minusDur))

    def finalReport(self):
        self.writeDefaultGraphicsPlots()

        # Grab the D and R values found and apply our timing convention
        D, R = self.solution

        if self.eventType == 'DandR':
            self.showMsg('Timestamp validity check ...')
            self.reportTimeValidity(D, R)

        self.calcNumBandApoints()

        self.showMsg('================= 0.68 confidence interval report =================')

        self.confidenceIntervalReport(1, self.deltaDurhi68, self.deltaDurlo68,
                                      self.deltaDhi68, self.deltaDlo68,
                                      self.deltaRhi68, self.deltaRlo68)

        self.showMsg('=============== end 0.68 confidence interval report ===============')

        self.showMsg('================= 0.95 confidence interval report =================')

        self.confidenceIntervalReport(2, self.deltaDurhi95, self.deltaDurlo95,
                                      self.deltaDhi95, self.deltaDlo95,
                                      self.deltaRhi95, self.deltaRlo95)

        envelopePlusR = self.plusR
        envelopePlusD = self.plusD
        envelopeMinusR = self.minusR
        envelopeMinusD = self.minusD

        self.showMsg('=============== end 0.95 confidence interval report ===============')

        self.showMsg('================= 0.9973 confidence interval report ===============')

        self.confidenceIntervalReport(3, self.deltaDurhi99, self.deltaDurlo99,
                                      self.deltaDhi99, self.deltaDlo99,
                                      self.deltaRhi99, self.deltaRlo99)

        self.showMsg('=============== end 0.9973 confidence interval report =============')

        # Set the values to be used for the envelope plot (saved during 0.95 ci calculations)
        self.plusR = envelopePlusR
        self.plusD = envelopePlusD
        self.minusR = envelopeMinusR
        self.minusD = envelopeMinusD

        self.doDframeReport()
        self.doRframeReport()
        self.doDurFrameReport()

        self.showMsg('=============== Summary report for Excel file =====================')

        self.reportSpecialProceduredUsed()

        if not self.timesAreValid:
            self.showMsg("Times are invalid due to corrupted timestamps!",
                         color='red', bold=True)

        if self.A > 0:
            self.showMsg('nominal magDrop: %0.2f' % ((np.log10(self.B) - np.log10(self.A)) * 2.5))
        else:
            self.showMsg('magDrop calculation not possible because A is negative')
        self.showMsg('snr: %0.2f' % self.snrB)

        self.doDtimeReport()
        self.doRtimeReport()
        self.doDurTimeReport()

        self.showMsg('Enter confidence intervals in Excel spreadsheet without + or - sign (assumed to be +/-)')

        self.showMsg('=========== end Summary report for Excel file =====================')

        self.showMsg("Solution 'envelope' in the main plot drawn using 0.95 confidence interval error bars")

    def doDframeReport(self):
        if self.eventType == 'DandR' or self.eventType == 'Donly':
            D, _ = self.solution
            entryNum = int(D)
            frameNum = float(self.yFrame[entryNum])
            # Dframe = D + frameNum - entryNum
            Dframe = (D - int(D)) * self.blockSize + int(D) + frameNum - entryNum
            self.showMsg('D frame number: {0:0.2f}'.format(Dframe), blankLine=False)
            errBar = max(abs(self.deltaDlo68), abs(self.deltaDhi68)) * self.blockSize
            self.showMsg('D: 0.6800 confidence intervals:  {{+/- {0:0.2f}}} (readings)'.format(errBar), blankLine=False)
            errBar = max(abs(self.deltaDlo95), abs(self.deltaDhi95)) * self.blockSize
            self.showMsg('D: 0.9500 confidence intervals:  {{+/- {0:0.2f}}} (readings)'.format(errBar), blankLine=False)
            errBar = max(abs(self.deltaDlo99), abs(self.deltaDhi99)) * self.blockSize
            self.showMsg('D: 0.9973 confidence intervals:  {{+/- {0:0.2f}}} (readings)'.format(errBar))

    def doRframeReport(self):
        if self.eventType == 'DandR' or self.eventType == 'Ronly':
            _, R = self.solution
            entryNum = int(R)
            frameNum = float(self.yFrame[entryNum])
            # Rframe = R + frameNum - entryNum
            Rframe = (R - int(R)) * self.blockSize + int(R) + frameNum - entryNum
            self.showMsg('R frame number: {0:0.2f}'.format(Rframe), blankLine=False)
            errBar = max(abs(self.deltaRlo68), abs(self.deltaRhi68)) * self.blockSize
            self.showMsg('R: 0.6800 confidence intervals:  {{+/- {0:0.2f}}} (readings)'.format(errBar), blankLine=False)
            errBar = max(abs(self.deltaRlo95), abs(self.deltaRhi95)) * self.blockSize
            self.showMsg('R: 0.9500 confidence intervals:  {{+/- {0:0.2f}}} (readings)'.format(errBar), blankLine=False)
            errBar = max(abs(self.deltaRlo99), abs(self.deltaRhi99)) * self.blockSize
            self.showMsg('R: 0.9973 confidence intervals:  {{+/- {0:0.2f}}} (readings)'.format(errBar))

    def doDurFrameReport(self):
        if self.eventType == 'DandR':
            D, R = self.solution
            self.showMsg('Duration (R - D): {0:0.4f} readings'.format((R - D) * self.blockSize), blankLine=False)
            errBar = ((self.deltaDurhi68 - self.deltaDurlo68) / 2) * self.blockSize
            self.showMsg('Duration: 0.6800 confidence intervals:  {{+/- {0:0.4f}}} (readings)'.format(errBar),
                         blankLine=False)
            errBar = ((self.deltaDurhi95 - self.deltaDurlo95) / 2) * self.blockSize
            self.showMsg('Duration: 0.9500 confidence intervals:  {{+/- {0:0.4f}}} (readings)'.format(errBar),
                         blankLine=False)
            errBar = ((self.deltaDurhi99 - self.deltaDurlo99) / 2) * self.blockSize
            self.showMsg('Duration: 0.9973 confidence intervals:  {{+/- {0:0.4f}}} (readings)'.format(errBar))

    def doDtimeReport(self):
        if self.eventType == 'DandR' or self.eventType == 'Donly':
            D, _ = self.solution
            ts = self.yTimes[int(D)]
            time = convertTimeStringToTime(ts)
            adjTime = time + (D - int(D)) * self.timeDelta
            self.Dtime = adjTime  # This is needed for the duration report (assumed to follow!!!)
            ts = convertTimeToTimeString(adjTime)
            self.showMsg('D time: %s' % ts, blankLine=False)
            errBar = max(abs(self.deltaDlo68), abs(self.deltaDhi68)) * self.timeDelta
            self.showMsg('D: 0.6800 confidence intervals:  {{+/- {0:0.4f}}} seconds'.format(errBar), blankLine=False)
            errBar = max(abs(self.deltaDlo95), abs(self.deltaDhi95)) * self.timeDelta
            self.showMsg('D: 0.9500 confidence intervals:  {{+/- {0:0.4f}}} seconds'.format(errBar), blankLine=False)
            errBar = max(abs(self.deltaDlo99), abs(self.deltaDhi99)) * self.timeDelta
            self.showMsg('D: 0.9973 confidence intervals:  {{+/- {0:0.4f}}} seconds'.format(errBar))

    def doRtimeReport(self):
        if self.eventType == 'DandR' or self.eventType == 'Ronly':
            _, R = self.solution
            ts = self.yTimes[int(R)]
            time = convertTimeStringToTime(ts)
            adjTime = time + (R - int(R)) * self.timeDelta
            self.Rtime = adjTime  # This is needed for the duration report (assumed to follow!!!)
            ts = convertTimeToTimeString(adjTime)
            self.showMsg('R time: %s' % ts, blankLine=False)
            errBar = max(abs(self.deltaRlo68), abs(self.deltaRhi68)) * self.timeDelta
            self.showMsg('R: 0.6800 confidence intervals:  {{+/- {0:0.4f}}} seconds'.format(errBar), blankLine=False)
            errBar = max(abs(self.deltaRlo95), abs(self.deltaRhi95)) * self.timeDelta
            self.showMsg('R: 0.9500 confidence intervals:  {{+/- {0:0.4f}}} seconds'.format(errBar), blankLine=False)
            errBar = max(abs(self.deltaRlo99), abs(self.deltaRhi99)) * self.timeDelta
            self.showMsg('R: 0.9973 confidence intervals:  {{+/- {0:0.4f}}} seconds'.format(errBar))

    def doDurTimeReport(self):
        if self.eventType == 'DandR':
            self.showMsg('Duration (R - D): {0:0.4f} seconds'.format(self.Rtime - self.Dtime), blankLine=False)
            errBar = ((self.deltaDurhi68 - self.deltaDurlo68) / 2) * self.timeDelta
            self.showMsg('Duration: 0.6800 confidence intervals:  {{+/- {0:0.4f}}} seconds'.format(errBar), blankLine=False)
            errBar = ((self.deltaDurhi95 - self.deltaDurlo95) / 2) * self.timeDelta
            self.showMsg('Duration: 0.9500 confidence intervals:  {{+/- {0:0.4f}}} seconds'.format(errBar), blankLine=False)
            errBar = ((self.deltaDurhi99 - self.deltaDurlo99) / 2) * self.timeDelta
            self.showMsg('Duration: 0.9973 confidence intervals:  {{+/- {0:0.4f}}} seconds'.format(errBar))

    def reportTimeValidity(self, D, R):
        intD = int(D)
        intR = int(R)
        dTime = convertTimeStringToTime(self.yTimes[intD])
        rTime = convertTimeStringToTime(self.yTimes[intR])
        
        # Here we check for a 'midnight transition'
        if rTime < dTime:
            rTime += 24 * 60 * 60
            self.showMsg('D and R enclose a transition through midnight')

        if self.timeDelta == 0:
            self.showMsg('Timestamps are corrupted in a manner that caused a '
                         'timeDelta of '
                         '0.0 to be estimated!', color='red', bold=True)
            self.showInfo(
                'Timestamps are corrupted in a manner that caused a '
                         'timeDelta of '
                         '0.0 to be estimated!')
            return

        numEnclosedReadings = int(round((rTime - dTime) / self.timeDelta))
        self.showMsg('From timestamps at D and R, calculated %d reading blocks.  From reading blocks, calculated %d blocks.' %
                     (numEnclosedReadings, intR - intD))
        if numEnclosedReadings == intR - intD:
            self.showMsg('Timestamps appear valid @ D and R')
            self.timesAreValid = True
        else:
            self.showMsg('! There is something wrong with timestamps at D '
                         'and/or R or frames have been dropped !', bold=True,
                         color='red')
        
    def computeErrorBars(self):

        self.snrB = (self.B - self.A) / self.sigmaB
        self.snrA = (self.B - self.A) / self.sigmaA
        snr = max(self.snrB, 0.2)  # A more reliable number

        D = int(round(80 / snr**2 + 0.5))
        
        D = max(10, D)
        if self.corCoefs.size > 1:
            D = round(1.5 * D)
        numPts = 2 * (D - 1) + 1
        posCoefs = []
        for entry in self.corCoefs:
            if entry < acfCoefThreshold:
                break
            posCoefs.append(entry)

        distGen = edgeDistributionGenerator(
                ntrials=100000, numPts=numPts, D=D, acfcoeffs=posCoefs,
                B=self.B, A=self.A, sigmaB=self.sigmaB, sigmaA=self.sigmaA)

        dist = None
        for dist in distGen:
            if type(dist) == float:
                if dist == -1.0:
                    self.showInfo(
                        'The Cholesky-Decomposition routine has failed.  This may be because the light curve ' +
                        'required some level of block integration.  Please examine the light curve for that possibility.')
                    return
                self.progressBar.setValue(dist * 100)
                QtGui.QApplication.processEvents()
                if self.cancelRequested:
                    self.cancelRequested = False
                    self.showMsg('Error bar calculation was cancelled')
                    self.progressBar.setValue(0)
                    # self.finalReport()
                    return
            else:
                # self.showMsg('Error bar calculation done')
                self.calcErrBars.setEnabled(False)
                self.progressBar.setValue(0)
        
        y, x = np.histogram(dist, bins=1000)
        self.loDbar95, _, self.hiDbar95, self.deltaDlo95, self.deltaDhi95 = ciBars(dist=dist, ci=0.95)
        self.loDbar99, _, self.hiDbar99, self.deltaDlo99, self.deltaDhi99 = ciBars(dist=dist, ci=0.9973)
        self.loDbar68, _, self.hiDbar68, self.deltaDlo68, self.deltaDhi68 = ciBars(dist=dist, ci=0.6827)

        self.deltaRlo95 = - self.deltaDhi95
        self.deltaRhi95 = - self.deltaDlo95

        self.deltaRlo99 = - self.deltaDhi99
        self.deltaRhi99 = - self.deltaDlo99

        self.deltaRlo68 = - self.deltaDhi68
        self.deltaRhi68 = - self.deltaDlo68
        
        if isinstance(dist, np.ndarray):
            durDist = createDurDistribution(dist)
        else:
            self.showInfo('Unexpected error: variable dist is not of type np.ndarray')
            return
        ydur, xdur = np.histogram(durDist, bins=1000)
        self.loDurbar95, _, self.hiDurbar95, self.deltaDurlo95, self.deltaDurhi95 = ciBars(dist=durDist, ci=0.95)
        self.loDurbar99, _, self.hiDurbar99, self.deltaDurlo99, self.deltaDurhi99 = ciBars(dist=durDist, ci=0.9973)
        self.loDurbar68, _, self.hiDurbar68, self.deltaDurlo68, self.deltaDurhi68 = ciBars(dist=durDist, ci=0.6827)

        pg.setConfigOptions(antialias=True)
        pen = pg.mkPen((0, 0, 0), width=2)
        
        self.errBarWin = pg.GraphicsWindow(
            title='Solution distributions with confidence intervals marked')
        self.errBarWin.resize(1200, 600)
        layout = QtGui.QGridLayout()
        self.errBarWin.setLayout(layout)
        
        pw = PlotWidget(viewBox=CustomViewBox(border=(0, 0, 0)),
                        enableMenu=False, title='Distribution of edge (D) errors due to noise',
                        labels={'bottom': 'Reading blocks'})
        self.dBarPlotItem = pw.getPlotItem()
        pw.hideButtons()
        
        pw2 = PlotWidget(viewBox=CustomViewBox(border=(0, 0, 0)),
                         enableMenu=False, title='Distribution of duration (R - D) errors due to noise',
                         labels={'bottom': 'Reading blocks'})
        self.durBarPlotItem = pw2.getPlotItem()
        pw2.hideButtons()
        
        vb = pw.getViewBox()
        vb.rbScaleBox.setPen(pg.mkPen((255, 0, 0), width=2))
        vb.rbScaleBox.setBrush(pg.mkBrush(None))
        
        vb = pw2.getViewBox()
        vb.rbScaleBox.setPen(pg.mkPen((255, 0, 0), width=2))
        vb.rbScaleBox.setBrush(pg.mkBrush(None))
        
        layout.addWidget(pw, 0, 0)
        layout.addWidget(pw2, 0, 1)
        
        pw.plot(x-D, y, stepMode=True, fillLevel=0, brush=(0, 0, 255, 150))
        pw.addLine(y=0, z=-10, pen=[0, 0, 255])
        pw.addLine(x=0, z=+10, pen=[255, 0, 0])
        
        yp = max(y) * 0.75
        x1 = self.loDbar68-D
        pw.plot(x=[x1, x1], y=[0, yp], pen=pen)
        
        x2 = self.hiDbar68-D
        pw.plot(x=[x2, x2], y=[0, yp], pen=pen)
        
        pw.addLegend()
        legend68 = '[%0.2f,%0.2f] @ 0.6827' % (x1, x2)
        pw.plot(name=legend68)

        self.showMsg("Error bar report based on 100,000 simulations (units are readings)...")

        self.showMsg('loDbar   @ .68 ci: %8.4f' % (x1 * self.blockSize), blankLine=False)
        self.showMsg('hiDbar   @ .68 ci: %8.4f' % (x2 * self.blockSize), blankLine=False)
        
        yp = max(y) * 0.25
        x1 = self.loDbar95-D
        pw.plot(x=[x1, x1], y=[0, yp], pen=pen)
        x2 = self.hiDbar95-D
        pw.plot(x=[x2, x2], y=[0, yp], pen=pen)
        
        self.showMsg('loDbar   @ .95 ci: %8.4f' % (x1 * self.blockSize), blankLine=False)
        self.showMsg('hiDbar   @ .95 ci: %8.4f' % (x2 * self.blockSize), blankLine=False)
        
        legend95 = '[%0.2f,%0.2f] @ 0.95' % (x1, x2)
        pw.plot(name=legend95)

        yp = max(y) * 0.15
        x1 = self.loDbar99 - D
        pw.plot(x=[x1, x1], y=[0, yp], pen=pen)
        x2 = self.hiDbar99 - D
        pw.plot(x=[x2, x2], y=[0, yp], pen=pen)

        self.showMsg('loDbar   @ .9973 ci: %8.4f' % (x1 * self.blockSize), blankLine=False)
        self.showMsg('hiDbar   @ .9973 ci: %8.4f' % (x2 * self.blockSize), blankLine=True)

        legend99 = '[%0.2f,%0.2f] @ 0.9973' % (x1, x2)
        pw.plot(name=legend99)
        
        pw.hideAxis('left')
        
        pw2.plot(xdur, ydur, stepMode=True, fillLevel=0, brush=(0, 0, 255, 150))
        pw2.addLine(y=0, z=-10, pen=[0, 0, 255])
        pw2.addLine(x=0, z=+10, pen=[255, 0, 0])

        yp = max(ydur) * 0.75
        x1 = self.loDurbar68
        pw2.plot(x=[x1, x1], y=[0, yp], pen=pen)
        x2 = self.hiDurbar68
        pw2.plot(x=[x2, x2], y=[0, yp], pen=pen)
        
        pw2.addLegend()
        legend68 = '[%0.2f,%0.2f] @ 0.6827' % (x1, x2)
        pw2.plot(name=legend68)
        
        self.showMsg('loDurBar @ .68 ci: %8.4f' % (x1 * self.blockSize), blankLine=False)
        self.showMsg('hiDurBar @ .68 ci: %8.4f' % (x2 * self.blockSize), blankLine=False)
        
        yp = max(ydur) * 0.25
        x1 = self.loDurbar95
        pw2.plot(x=[x1, x1], y=[0, yp], pen=pen)
        x2 = self.hiDurbar95
        pw2.plot(x=[x2, x2], y=[0, yp], pen=pen)
        
        self.showMsg('loDurBar @ .95 ci: %8.4f' % (x1 * self.blockSize), blankLine=False)
        self.showMsg('hiDurBar @ .95 ci: %8.4f' % (x2 * self.blockSize), blankLine=False)
        
        legend95 = '[%0.2f,%0.2f] @ 0.95' % (x1, x2)
        pw2.plot(name=legend95)

        yp = max(ydur) * 0.15
        x1 = self.loDurbar99
        pw2.plot(x=[x1, x1], y=[0, yp], pen=pen)
        x2 = self.hiDurbar99
        pw2.plot(x=[x2, x2], y=[0, yp], pen=pen)

        self.showMsg('loDurBar @ .9973 ci: %8.4f' % (x1 * self.blockSize), blankLine=False)
        self.showMsg('hiDurBar @ .9973 ci: %8.4f' % (x2 * self.blockSize), blankLine=True)

        legend99 = '[%0.2f,%0.2f] @ 0.9973' % (x1, x2)
        pw2.plot(name=legend99)
        
        pw2.hideAxis('left')
        
        self.writeBarPlots.setEnabled(True)

        if self.timestampListIsEmpty(self.yTimes):
            self.showMsg('Cannot produce final report because timestamps are missing.', bold=True, color='red')
        else:
            self.finalReport()

        self.reDrawMainPlot()  # To add envelope to solution

    def displaySolution(self, subframe=True):
        D, R = self.solution

        # D and R are floats and may be fractional because of sub-frame timing.
        # We have to remove the effects of sub-frame timing to calulate the D
        # and R transition points as integers.

        if D and R:
            Dtransition = trunc(floor(self.solution[0]))
            Rtransition = trunc(floor(self.solution[1]))
            if subframe:
                solMsg = ('D: %d  R: %d  D(subframe): %0.2f  R(subframe): %0.2f' %
                          (Dtransition, Rtransition, D, R))
            else:
                solMsg = ('D: %d  R: %d' % (D, R))
            self.showMsg('solution indices found: ' + solMsg)
        elif D:
            Dtransition = trunc(floor(self.solution[0]))
            if subframe:
                solMsg = ('D: %d  D(subframe): %0.2f' % (Dtransition, D))
            else:
                solMsg = ('D: %d' % (D))
            self.showMsg('solution indices found: ' + solMsg)
        else:
            Rtransition = trunc(floor(self.solution[1]))
            if subframe:
                solMsg = ('R: %d  R(subframe): %0.2f' % (Rtransition, R))
            else:
                solMsg = ('R: %d' % (R))

            self.showMsg('solution indices found: ' + solMsg)

    def update_noise_parameters_from_solution(self):
        D, R = self.solution
        # D and R are floats and may be fractional because of sub-frame timing.
        # Here we remove the effects of sub-frame timing to calulate the D and
        # and R transition points as integers.
        if D:
            D = trunc(floor(D))
        if R:
            R = trunc(floor(R))

        self.showMsg('Recalculating noise parameters to include all points '
                     'based on first pass solution ====',
                     color='red', bold=True)
        if D and R:
            self.sigmaA = None
            self.corCoefs = []
            self.selectedPoints = {}

            self.togglePointSelected(self.left)
            self.togglePointSelected(D-1)
            self.processBaselineNoise(secondPass=True)

            self.selectedPoints = {}
            self.togglePointSelected(R+1)
            self.togglePointSelected(self.right)
            self.processBaselineNoise(secondPass=True)

            self.selectedPoints = {}
            self.togglePointSelected(D+1)
            self.togglePointSelected(R-1)
            self.processEventNoise(secondPass=True)
        elif D:
            self.sigmaA = None
            self.corCoefs = []
            self.selectedPoints = {}

            self.togglePointSelected(self.left)
            self.togglePointSelected(D - 1)
            self.processBaselineNoise(secondPass=True)

            self.selectedPoints = {}
            self.togglePointSelected(D + 1)
            self.togglePointSelected(self.right)
            self.processEventNoise(secondPass=True)
        else:
            self.sigmaA = None
            self.corCoefs = []
            self.selectedPoints = {}

            self.togglePointSelected(R + 1)
            self.togglePointSelected(self.right)
            self.processBaselineNoise(secondPass=True)

            self.selectedPoints = {}
            self.togglePointSelected(self.left)
            self.togglePointSelected(R - 1)
            self.processEventNoise(secondPass=True)

        return

    def extract_noise_parameters_from_iterative_solution(self):

        D, R = self.solution
        # D and R are floats and may be fractional because of sub-frame timing.
        # Here we remove the effects of sub-frame timing to calulate the D and
        # and R transition points as integers.
        if D:
            D = trunc(floor(D))
        if R:
            R = trunc(floor(R))

        if D and R:
            self.sigmaA = None
            self.corCoefs = []

            self.processBaselineNoiseFromIterativeSolution(self.left, D - 1)

            self.processBaselineNoiseFromIterativeSolution(R + 1, self.right)

            self.processEventNoiseFromIterativeSolution(D + 1, R - 1)

            # Try to warn user about the possible need for block integration by testing the lag 1
            # and lag 2 correlation coefficients.  The tests are just guesses on my part, so only
            # warnings are given.  Later, the Cholesky-Decomposition may fail because block integration
            # was really needed.  That is a fatal error but is trapped and the user alerted to the problem

            if len(self.corCoefs) > 1:
                if self.corCoefs[1] >= 0.7:
                    self.showInfo(
                        'The auto-correlation coefficient at lag 1 is suspiciously large. '
                        'This may be because the light curve needs some degree of block integration. '
                        'Failure to do a needed block integration allows point-to-point correlations caused by '
                        'the camera integration to artificially induce non-physical correlated noise.')
                elif len(self.corCoefs) > 2:
                    if self.corCoefs[2] >= 0.3:
                        self.showInfo(
                            'The auto-correlation coefficient at lag 2 is suspiciously large. '
                            'This may be because the light curve needs some degree of block integration. '
                            'Failure to do a needed block integration allows point-to-point correlations caused by '
                            'the camera integration to artificially induce non-physical correlated noise.')

            if self.sigmaA is None:
                self.sigmaA = self.sigmaB
        elif D:
            self.sigmaA = None
            self.corCoefs = []

            self.processBaselineNoiseFromIterativeSolution(self.left, D - 1)

            self.processEventNoiseFromIterativeSolution(D + 1, self.right)
            if self.sigmaA is None:
                self.sigmaA = self.sigmaB
        else:  # R only
            self.sigmaA = None
            self.corCoefs = []

            self.processBaselineNoiseFromIterativeSolution(R + 1, self.right)

            self.processEventNoiseFromIterativeSolution(self.left, R - 1)
            if self.sigmaA is None:
                self.sigmaA = self.sigmaB

        self.prettyPrintCorCoefs()

        return

    def try_to_get_solution(self):
        self.solution = None
        self.reDrawMainPlot()
        solverGen = solver(
            eventType=self.eventType, yValues=self.yValues,
            left=self.left, right=self.right,
            sigmaB=self.sigmaB, sigmaA=self.sigmaA,
            dLimits=self.dLimits, rLimits=self.rLimits,
            minSize=self.minEvent, maxSize=self.maxEvent)
        self.cancelRequested = False
        for item in solverGen:
            if item[0] == 'fractionDone':
                pass
                # Here we should update progress bar and check for cancellation
                # self.progressBar.setValue(item[1] * 100)
                # QtGui.QApplication.processEvents()
                # if self.cancelRequested:
                #     self.cancelRequested = False
                #     self.runSolver = False
                #     self.showMsg('Solution search was cancelled')
                #     self.progressBar.setValue(0)
                #     break
            elif item[0] == 'no event present':
                self.showMsg('No event fitting search criteria could be found.')
                # self.progressBar.setValue(0)
                self.runSolver = False
                break
            else:
                # self.progressBar.setValue(0)
                self.solution = item[0]
                self.B = item[1]
                self.A = item[2]

    def compareFirstAndSecondPassResults(self):
        D1, R1 = self.firstPassSolution
        D2, R2 = self.secondPassSolution
        if D1:
            D1 = trunc(floor(D1))
        if D2:
            D2 = trunc(floor(D2))
        if R1:
            R1 = trunc(floor(R1))
        if R2:
            R2 = trunc(floor(R2))

        if D1 == D2 and R1 == R2:
            return

        # There is a difference in the D and/or R transition points identified
        # in the first and second passes --- alert the user.
        self.showInfo('The D and/or R transition points identified in pass 1 '
                      'are different from those found in pass 2 (after '
                      'automatic noise analysis).  '
                      'It is recommended that you '
                      'rerun the light curve using the D and R values found in '
                      'this second pass to more accurately select points for '
                      'the initial noise analysis.')

    def findEvent(self):
        if self.DandR.isChecked():
            self.eventType = 'DandR'
            self.showMsg('Locate a "D and R" event triggered')
        elif self.Donly.isChecked():
            self.eventType = 'Donly'
            self.showMsg('Locate a "D only" event triggered')
        else:
            self.eventType = 'Ronly'
            self.showMsg('Locate an "R only" event triggered')
        
        minText = self.minEventEdit.text().strip()
        maxText = self.maxEventEdit.text().strip()
        
        self.minEvent = None
        self.maxEvent = None
        
        if minText and not maxText:
            self.showInfo('If minEvent is filled in, so must be maxEvent')
            return
            
        if maxText and not minText:
            self.showInfo('If maxEvent is filled in, so must be minEvent')
            return
            
        if minText:
            if not minText.isnumeric():
                self.showInfo('Invalid entry for min event (rdgs)')
            else:
                self.minEvent = int(minText)
                if self.minEvent < 1:
                    self.showInfo('minEvent must be greater than 0')
                    return
        
        if maxText:
            if not maxText.isnumeric():
                self.showInfo('Invalid entry for max event (rdgs)')
            else:
                self.maxEvent = int(maxText)
                if self.maxEvent < self.minEvent:
                    self.showInfo('maxEvent must be >= minEvent')
                    return
                if self.maxEvent > self.right - self.left - 1:
                    self.showInfo('maxEvent is too large for selected points')
                    return
        if minText == '':
            minText = '<blank>'
        if maxText == '':
            maxText = '<blank>'
        self.showMsg('minEvent: ' + minText + '  maxEvent: ' + maxText)
        
        candFrom, numCandidates = candidateCounter(eventType=self.eventType,
                                                   dLimits=self.dLimits, rLimits=self.rLimits,
                                                   left=self.left, right=self.right,
                                                   numPts=self.right - self.left + 1,
                                                   minSize=self.minEvent, maxSize=self.maxEvent)
        if numCandidates < 0:
            self.showInfo('Search parameters are not properly specified')
            return

        if candFrom == 'usedSize':
            self.showMsg('Number of candidate solutions: ' + str(numCandidates) +
                     ' (using event min/max entries)')
        else:
            self.showMsg(
                'Number of candidate solutions: ' + str(numCandidates) +
                ' (using D/R region selections)')

        # self.runSolver = True
        # if numCandidates > 10000:
        #     msg = 'There are ' + str(numCandidates) + ' candidates in the solution set. '
        #     msg = msg + 'Do you wish to continue?'
        #     self.showQuery(msg, 'Your chance to narrow the potential candidates')
        #
        #     if self.queryRetVal == QMessageBox.Yes:
        #         self.showMsg('Yes was clicked --- starting solution search...')
        #         self.solution = None
        #         self.reDrawMainPlot()
        #     else:
        #         self.showMsg('"No" was clicked, so solver will be skipped')
        #         self.runSolver = False

        self.runSolver = True
        solverGen = None

        if self.runSolver:
            if self.eventType == 'DandR':
                self.showMsg('New solver results...', color='blue', bold=True)

                if candFrom == 'usedSize':
                    solverGen = find_best_event_from_min_max_size(
                        self.yValues, self.left, self.right,
                        self.minEvent, self.maxEvent)
                else:
                    solverGen = locate_event_from_d_and_r_ranges(
                            self.yValues, self.left, self.right, self.dLimits[0],
                            self.dLimits[1], self.rLimits[0], self.rLimits[1])

            elif self.eventType == 'Ronly':
                self.showMsg('New solver results...', color='blue', bold=True)
                if candFrom == 'usedSize':
                    pass
                else:
                    self.minEvent = self.rLimits[0] - self.left
                    self.maxEvent = self.rLimits[1] - self.left
                solverGen = \
                    find_best_r_only_from_min_max_size(
                        self.yValues, self.left, self.right, self.minEvent,
                        self.maxEvent)

            else:  # Donly
                self.showMsg('New solver results...', color='blue', bold=True)
                if candFrom == 'usedSize':
                    pass
                else:
                    self.minEvent = self.right - self.dLimits[1]
                    self.maxEvent = self.right - self.dLimits[0] - 1

                solverGen = \
                    find_best_d_only_from_min_max_size(
                        self.yValues, self.left, self.right, self.minEvent,
                        self.maxEvent)

            if solverGen is None:
                self.showInfo('Generator version not yet implemented')
                return

            self.cancelRequested = False

            for item in solverGen:
                if item[0] == 'fractionDone':
                    self.progressBar.setValue(item[1] * 100)
                    QtGui.QApplication.processEvents()
                    if self.cancelRequested:
                        self.cancelRequested = False
                        self.runSolver = False
                        self.showMsg('Solution search was cancelled')
                        self.progressBar.setValue(0)
                        return
                elif item[0] == 'no event present':
                    self.showMsg(
                        'No event fitting search criteria could be found.')
                    self.progressBar.setValue(0)
                    self.runSolver = False
                    return
                else:
                    d, r, b, a, sigmaB, sigmaA, metric = item
                    self.solution = (d, r)
                    self.progressBar.setValue(0)

            self.showMsg('Integer (non-subframe) solution...', blankLine=False)
            self.showMsg(
                'sigB:%.2f  sigA:%.2f B:%.2f A:%.2f' %
                (sigmaB, sigmaA, b, a),
                blankLine=False)
            self.displaySolution(subframe=False)  # First solution

            # This fills in self.sigmaB and self.sigmaA
            self.extract_noise_parameters_from_iterative_solution()

            subDandR, new_b, new_a = subFrameAdjusted(
                eventType=self.eventType, cand=(d, r), B=b, A=a,
                sigmaB=self.sigmaB, sigmaA=self.sigmaA, yValues=self.yValues,
                left=self.left, right=self.right)

            self.solution = subDandR

            self.showMsg('Subframe adjusted solution...', blankLine=False)
            self.showMsg(
                'sigB:%.2f  sigA:%.2f B:%.2f A:%.2f' %
                (self.sigmaB, self.sigmaA, new_b, new_a),
                blankLine=False)
            self.displaySolution()  # Adjusted solution

            self.B = new_b
            self.A = new_a

            # Activate this code if not using old solver following this.
            if self.only_new_solver_wanted:
                self.dRegion = None
                self.rRegion = None
                self.dLimits = None
                self.rLimits = None

            self.showMsg('... end New solver results', color='blue', bold=True)

            if not self.only_new_solver_wanted:
                # Proceed with old dual-pass 'solver'
                self.solution = (None, None)
                self.try_to_get_solution()

                if self.solution:
                    self.showMsg(
                        'sigB:%.2f  sigA:%.2f B:%.2f A:%.2f' %
                        (self.sigmaB, self.sigmaA, self.B, self.A),
                        blankLine=False)
                    self.displaySolution()
                    # self.firstPassSolution = deepcopy(self.solution)
                    # # Now we will use the solution for D and/or R to update
                    # # noise calculation and then make a second pass.
                    # self.update_noise_parameters_from_solution()
                    # self.showMsg('Second pass solution using updated noise '
                    #              'parameters =====', color='red', bold=True)
                    # self.solution = (None, None)
                    # self.try_to_get_solution()
                    # if self.solution:
                    #     self.displaySolution()
                    self.dRegion = None
                    self.rRegion = None
                    self.dLimits = None
                    self.rLimits = None

        if self.runSolver and self.solution:
            D, R = self.solution
            if D is not None:
                D = round(D, 2)
            if R is not None:
                R = round(R, 2)
            self.solution = (D, R)
            if self.eventType == 'DandR':
                # ans = '(%.2f,%.2f) B: %.2f  A: %.2f' % (D, R, self.B, self.A)
                pass
            elif self.eventType == 'Donly':
                # ans = '(%.2f,None) B: %.2f  A: %.2f' % (D, self.B, self.A)
                pass
            elif self.eventType == 'Ronly':
                # ans = '(None,%.2f) B: %.2f  A: %.2f' % (R, self.B, self.A)
                pass
            else:
                raise Exception('Undefined event type')
            # self.showMsg('Raw solution (debug output): ' + ans)
        elif self.runSolver:
            self.showMsg('Event could not be found')
            
        self.reDrawMainPlot()

        self.calcErrBars.setEnabled(True)
        
    def fillTableViewOfData(self):
        
        self.table.setRowCount(self.dataLen)
        self.table.setVerticalHeaderLabels([str(i) for i in range(self.dataLen)])

        for i in range(self.dataLen):
            newitem = QtGui.QTableWidgetItem(str(i))
            self.table.setItem(i, 0, newitem)
            neatStr = fp.to_precision(self.yValues[i], 6)
            newitem = QtGui.QTableWidgetItem(str(neatStr))
            self.table.setItem(i, 3, newitem)
            newitem = QtGui.QTableWidgetItem(str(self.yTimes[i]))
            self.table.setItem(i, 2, newitem)
            newitem = QtGui.QTableWidgetItem(str(self.yFrame[i]))
            self.table.setItem(i, 1, newitem)
            
        self.table.resizeColumnsToContents()   
        
    def readDataFromFile(self):
        
        self.initializeVariablesThatDontDependOnAfile()
        
        self.disableAllButtons()
        self.mainPlot.clear()
        self.textOut.clear()
        self.initializeTableView()
        
        # Open a file select dialog
        self.filename, _ = QFileDialog.getOpenFileName(
                self,                                      # parent
                "Select light curve csv file",             # title for dialog
                self.settings.value('lightcurvedir', ""),  # starting directory
                "Csv files (*.csv)")
        if self.filename:
            self.setWindowTitle('PYOTE Version: ' + version.version() + '  File in process: ' + self.filename)
            dirpath, _ = os.path.split(self.filename)
            self.logFile, _ = os.path.splitext(self.filename)
            self.logFile = self.logFile + '.PYOTE.log'
            
            curDateTime = datetime.datetime.today().ctime()
            self.showMsg('')
            self.showMsg('#' * 20 + ' PYOTE ' + version.version() + '  session started: ' + curDateTime + '  ' + '#' * 20)
        
            # Make the directory 'sticky'
            self.settings.setValue('lightcurvedir', dirpath)
            self.showMsg('filename: ' + self.filename, bold=True, color="red")

            try:
                self.outliers = []
                frame, time, value, self.secondary, self.ref2, self.ref3, headers = readLightCurve(self.filename)
                values = [float(item) for item in value]

                if not frame:
                    # This is a raw data file, imported for test purposes
                    # raw = values
                    self.showInfo('raw data file read')
                    return

                refStar = [float(item) for item in self.secondary]

                self.secondarySelector.setValue(1)
                if self.secondary:
                    self.showSecondaryCheckBox.setEnabled(True)
                    self.showSecondaryCheckBox.setChecked(True)
                    if self.ref2:
                        self.secondarySelector.setEnabled(True)
                        self.secondarySelector.setMaximum(2)
                        if self.ref3:
                            self.secondarySelector.setMaximum(3)
                    else:
                        self.secondarySelector.setEnabled(False)
                        self.secondarySelector.setMaximum(1)

                # If no timestamps were found in the input file, prompt for manual entry
                if self.timestampListIsEmpty(time):
                    self.showMsg('Manual entry of timestamps requested as the file contained no timestamp entries', bold=True)
                    self.showInfo('This file does not contain timestamp entries. The manual entry of either two timestamps' +
                                  ' OR one timestamp and a frame delta time will need to be done.')
                    errmsg = ''
                    while errmsg != 'ok':
                        errmsg, manualTime, dataEntered = manualTimeStampEntry(frame, TSdialog())
                        if errmsg != 'ok':
                            self.showInfo(errmsg)
                        else:
                            self.showMsg(dataEntered, bold=True)
                            if manualTime:  # Needed only during development
                                time = manualTime[:]

                self.showMsg('=' * 20 + ' file header lines ' + '=' * 20, bold=True, blankLine=False)
                for item in headers:
                    self.showMsg(item, blankLine=False)
                self.showMsg('=' * 20 + ' end header lines ' + '=' * 20, bold=True)

                self.yTimes = time[:]
                self.yTimesCopy = time[:]
                self.yValues = np.array(values)
                self.yValCopy = np.ndarray(shape=(len(self.yValues),))
                np.copyto(self.yValCopy, self.yValues)
                self.yRefStar = np.array(refStar)
                self.yRefStarCopy = np.array(refStar)
                
                if self.yRefStar.size > 0:
                    self.smoothSecondaryButton.setEnabled(True)
                    self.numSmoothPointsEdit.setEnabled(True)
                    
                self.dataLen = len(self.yValues)
                self.yFrame = frame[:]
                self.yFrameCopy = frame[:]
                
                # Automatically select all points
                # noinspection PyUnusedLocal
                self.yStatus = [INCLUDED for _i in range(self.dataLen)]
                self.left = 0
                self.right = self.dataLen - 1

                self.timeDelta, self.outliers, self.errRate = getTimeStepAndOutliers(self.yTimes)

                self.mainPlot.autoRange()

                self.setDataLimits.setEnabled(True)
                self.writePlot.setEnabled(True)

                if self.only_new_solver_wanted:
                    self.markDzone.setEnabled(True)
                    self.markRzone.setEnabled(True)
                    self.minEventEdit.setEnabled(True)
                    self.maxEventEdit.setEnabled(True)
                    self.locateEvent.setEnabled(True)
                else:
                    self.markDzone.setEnabled(True)
                    self.markRzone.setEnabled(True)
                    self.minEventEdit.setEnabled(True)
                    self.maxEventEdit.setEnabled(True)
                    self.locateEvent.setEnabled(True)
                    # self.doNoiseAnalysis.setEnabled(True)
                    # self.computeSigmaA.setEnabled(True)

                self.doBlockIntegration.setEnabled(True)
                self.startOver.setEnabled(True)
                self.fillTableViewOfData()
                self.timeDelta, self.outliers, self.errRate = getTimeStepAndOutliers(self.yTimes)
                self.showMsg('timeDelta: ' + fp.to_precision(self.timeDelta, 6) + ' seconds per reading', blankLine=False)
                self.showMsg('timestamp error rate: ' + fp.to_precision(100 * self.errRate, 2) + '%')

                if self.outliers:
                    self.showTimestampErrors.setEnabled(True)
                    self.showTimestampErrors.setChecked(True)
                self.reDrawMainPlot()
                self.mainPlot.autoRange()
            except Exception as e:
                self.showMsg(str(e))
    
    def illustrateTimestampOutliers(self):
        for pos in self.outliers:
            vLine = pg.InfiniteLine(pos=pos+0.5, pen=(255, 0, 0))
            self.mainPlot.addItem(vLine)

    def prettyPrintCorCoefs(self):
        outStr = 'noise corr coefs: ['
        
        posCoefs = []
        for coef in self.corCoefs:
            if coef < acfCoefThreshold:
                break
            posCoefs.append(coef)

        for i in range(len(posCoefs)-1):
            outStr = outStr + fp.to_precision(posCoefs[i], 3) + ', '
        outStr = outStr + fp.to_precision(posCoefs[-1], 3)
        outStr = outStr + ']  (based on ' + str(self.numPtsInCorCoefs) + ' points)'
        outStr = outStr + '  sigmaB: ' + fp.to_precision(self.sigmaB, 4)
        self.showMsg(outStr)
    
    def processEventNoise(self, secondPass=False):
        if len(self.selectedPoints) != 2:
            self.showInfo('Exactly two points must be selected for this operation')
            return
        selPts = self.selectedPoints.keys()
        left = int(min(selPts))
        right = int(max(selPts))
        if (right - left) < 9:
            if secondPass:
                self.removePointSelections()
                self.sigmaA = self.sigmaB
                return
            else:
                self.showInfo('At least 10 points must be included.')
                return
        if left < self.left or right > self.right:
            self.showInfo('Selection point(s) outside of included data points')
            self.removePointSelections()
            return
        else:
            self.eventXvals = []
            self.eventYvals = []
            for i in range(left, right+1):
                self.eventXvals.append(i)
                self.eventYvals.append(self.yValues[i])
            self.showSelectedPoints('Points selected for event noise '
                                    'analysis: ')
            # self.doNoiseAnalysis.setEnabled(True)
            # self.computeSigmaA.setEnabled(True)
        
        self.removePointSelections()
        _, self.numNApts, self.sigmaA = getCorCoefs(self.eventXvals, self.eventYvals)
        self.showMsg('Event noise analysis done using ' + str(self.numNApts) + 
                     ' points ---  sigmaA: ' + fp.to_precision(self.sigmaA, 4))
        
        self.reDrawMainPlot()

    def processEventNoiseFromIterativeSolution(self, left, right):

        if (right - left) < 9:
            return

        assert left >= self.left
        assert right <= self.right

        self.eventXvals = []
        self.eventYvals = []
        for i in range(left, right + 1):
            self.eventXvals.append(i)
            self.eventYvals.append(self.yValues[i])

        _, self.numNApts, self.sigmaA = getCorCoefs(self.eventXvals,
                                                    self.eventYvals)

    def processBaselineNoise(self, secondPass=False):

        if len(self.selectedPoints) != 2:
            self.showInfo('Exactly two points must be selected for this operation')
            return
        selPts = self.selectedPoints.keys()
        left = int(min(selPts))
        right = int(max(selPts))
        if (right - left) < 14:
            if secondPass:
                self.removePointSelections()
                return
            else:
                self.showInfo('At least 15 points must be included.')
                return
        if left < self.left or right > self.right:
            self.showInfo('Selection point(s) outside of included data points')
            return
        else:
            self.baselineXvals = []
            self.baselineYvals = []
            for i in range(left, right+1):
                self.baselineXvals.append(i)
                self.baselineYvals.append(self.yValues[i])
            self.showSelectedPoints('Points selected for baseline noise '
                                    'analysis: ')
            # self.doNoiseAnalysis.setEnabled(True)
            # self.computeSigmaA.setEnabled(True)
        
        self.removePointSelections()
        
        self.newCorCoefs, self.numNApts, sigB = getCorCoefs(self.baselineXvals, self.baselineYvals)
        self.showMsg('Baseline noise analysis done using ' + str(self.numNApts) + 
                     ' baseline points')
        if len(self.corCoefs) == 0:
            self.corCoefs = np.ndarray(shape=(len(self.newCorCoefs),))
            np.copyto(self.corCoefs, self.newCorCoefs)
            self.numPtsInCorCoefs = self.numNApts
            self.sigmaB = sigB
        else:
            totalPoints = self.numNApts + self.numPtsInCorCoefs
            self.corCoefs = (self.corCoefs * self.numPtsInCorCoefs +
                             self.newCorCoefs * self.numNApts) / totalPoints
            self.sigmaB = (self.sigmaB * self.numPtsInCorCoefs +
                           sigB * self.numNApts) / totalPoints
            self.numPtsInCorCoefs = totalPoints
        
        self.prettyPrintCorCoefs()

        # Try to warn user about the possible need for block integration by testing the lag 1
        # and lag 2 correlation coefficients.  The tests are just guesses on my part, so only
        # warnings are given.  Later, the Cholesky-Decomposition may fail because block integration
        # was really needed.  That is a fatal error but is trapped and the user alerted to the problem

        if len(self.corCoefs) > 1:
            if self.corCoefs[1] >= 0.7:
                self.showInfo('The auto-correlation coefficient at lag 1 is suspiciously large. '
                              'This may be because the light curve needs some degree of block integration. '
                              'Failure to do a needed block integration allows point-to-point correlations caused by '
                              'the camera integration to artificially induce non-physical correlated noise.')
            elif len(self.corCoefs) > 2:
                if self.corCoefs[2] >= 0.3:
                    self.showInfo('The auto-correlation coefficient at lag 2 is suspiciously large. '
                                  'This may be because the light curve needs some degree of block integration. '
                                  'Failure to do a needed block integration allows point-to-point correlations caused by '
                                  'the camera integration to artificially induce non-physical correlated noise.')
        
        if self.sigmaA is None:
            self.sigmaA = self.sigmaB

        self.reDrawMainPlot()
                
        self.locateEvent.setEnabled(True)
        self.markDzone.setEnabled(True)
        self.markRzone.setEnabled(True)
        self.minEventEdit.setEnabled(True)
        self.maxEventEdit.setEnabled(True)

    def processBaselineNoiseFromIterativeSolution(self, left, right):

        # if (right - left) < 14:
        #     return 'Failed'

        assert left >= self.left
        assert right <= self.right

        self.baselineXvals = []
        self.baselineYvals = []
        for i in range(left, right + 1):
            self.baselineXvals.append(i)
            self.baselineYvals.append(self.yValues[i])

        self.newCorCoefs, self.numNApts, sigB = getCorCoefs(self.baselineXvals,
                                                            self.baselineYvals)

        if len(self.corCoefs) == 0:
            self.corCoefs = np.ndarray(shape=(len(self.newCorCoefs),))
            np.copyto(self.corCoefs, self.newCorCoefs)
            self.numPtsInCorCoefs = self.numNApts
            self.sigmaB = sigB
        else:
            totalPoints = self.numNApts + self.numPtsInCorCoefs
            self.corCoefs = (self.corCoefs * self.numPtsInCorCoefs +
                             self.newCorCoefs * self.numNApts) / totalPoints
            self.sigmaB = (self.sigmaB * self.numPtsInCorCoefs +
                           sigB * self.numNApts) / totalPoints
            self.numPtsInCorCoefs = totalPoints

    def removePointSelections(self):
        for i, oldStatus in self.selectedPoints.items():
            self.yStatus[i] = oldStatus
        self.selectedPoints = {}
        
    def disableAllButtons(self):
        self.showSecondaryCheckBox.setEnabled(False)
        self.secondarySelector.setEnabled(False)
        self.normalizeButton.setEnabled(False)
        self.smoothSecondaryButton.setEnabled(False)
        self.numSmoothPointsEdit.setEnabled(False)
        self.setDataLimits.setEnabled(False)      
        self.doBlockIntegration.setEnabled(False)    
        # self.doNoiseAnalysis.setEnabled(False)
        self.locateEvent.setEnabled(False)
        self.calcErrBars.setEnabled(False)
        self.startOver.setEnabled(False)
        self.markDzone.setEnabled(False)
        self.markRzone.setEnabled(False)
        self.numSmoothPointsEdit.setEnabled(False)
        self.minEventEdit.setEnabled(False)
        self.maxEventEdit.setEnabled(False)
        self.writeBarPlots.setEnabled(False)

    # noinspection PyUnusedLocal
    def restart(self):
        
        self.initializeVariablesThatDontDependOnAfile()
        self.disableAllButtons()
        
        if self.errBarWin:
            self.errBarWin.close()
                
        self.yValues = np.ndarray(shape=(len(self.yValCopy),))
        np.copyto(self.yValues, self.yValCopy)
        self.yRefStar = np.ndarray(shape=(len(self.yRefStarCopy),))
        np.copyto(self.yRefStar, self.yRefStarCopy)
        self.yTimes = self.yTimesCopy[:]
        self.yFrame = self.yFrameCopy[:]
        self.dataLen = len(self.yTimes)
        self.timeDelta, self.outliers, self.errRate = getTimeStepAndOutliers(self.yTimes)
        self.fillTableViewOfData()
        
        if len(self.yRefStar) > 0:
            self.showSecondaryCheckBox.setEnabled(True)
            self.smoothSecondaryButton.setEnabled(True)
            self.numSmoothPointsEdit.setEnabled(True)
            self.secondarySelector.setEnabled(True)
        
        # Enable the initial set of buttons (allowed operations)
        self.startOver.setEnabled(True)
        self.doBlockIntegration.setEnabled(True)
        self.setDataLimits.setEnabled(True)

        if self.only_new_solver_wanted:
            self.markDzone.setEnabled(True)
            self.markRzone.setEnabled(True)
            self.locateEvent.setEnabled(True)
            self.minEventEdit.setEnabled(True)
            self.maxEventEdit.setEnabled(True)
        else:
            self.markDzone.setEnabled(True)
            self.markRzone.setEnabled(True)
            self.minEventEdit.setEnabled(True)
            self.maxEventEdit.setEnabled(True)
            self.locateEvent.setEnabled(True)
            # self.doNoiseAnalysis.setEnabled(True)
            # self.computeSigmaA.setEnabled(True)

        # Reset the data plot so that all points are visible
        self.mainPlot.autoRange()
        
        # Show all data points as INCLUDED
        self.yStatus = [INCLUDED for _i in range(self.dataLen)]
        
        # Set the 'left' and 'right' edges of 'included' data to 'all'
        self.left = 0
        self.right = self.dataLen - 1
        
        self.minEventEdit.clear()
        self.maxEventEdit.clear()
        
        self.reDrawMainPlot()
        self.mainPlot.autoRange()
        self.showMsg('*' * 20 + ' starting over ' + '*' * 20, color='blue')
    
    def drawSolution(self):
        def plot(x, y):
            self.mainPlot.plot(x, y, pen=pg.mkPen((150, 100, 100), width=3), symbol=None)
        
        B = self.B
        A = self.A
        
        if self.eventType == 'DandR':
            D = self.solution[0] - self.Doffset
            R = self.solution[1] - self.Roffset
        
            plot([self.left, D], [B, B])
            plot([D, D], [B, A])
            plot([D, R], [A, A])
            plot([R, R], [A, B])
            plot([R, self.right], [B, B])
        elif self.eventType == 'Donly':
            D = self.solution[0] - self.Doffset
            plot([self.left, D], [B, B])
            plot([D, D], [B, A])
            plot([D, self.right], [A, A])
        elif self.eventType == 'Ronly':
            R = self.solution[1] - self.Roffset
            plot([self.left, R], [A, A])
            plot([R, R], [A, B])
            plot([R, self.right], [B, B])
        else:
            raise Exception('Unrecognized event type')

    def calcNumBandApoints(self):
        if self.eventType == 'Donly':
            self.nBpts = self.solution[0] - self.left
            self.nApts = self.right - self.solution[0] - 1

        if self.eventType == 'Ronly':
            self.nBpts = self.right - self.solution[1]
            self.nApts = self.solution[1] - self.left

        if self.eventType == 'DandR':
            self.nBpts = self.right - self.solution[1] + self.solution[0] - self.left
            self.nApts = self.solution[1] - self.solution[0] - 1

        if self.nBpts < 1:
            self.nBpts = 1

        if self.nApts < 1:
                self.nApts = 1

    def drawEnvelope(self):
        def plot(x, y):
            self.mainPlot.plot(x, y, pen=pg.mkPen((150, 100, 100), width=2), symbol=None)
        
        if self.solution is None:
            return

        self.calcNumBandApoints()

        if self.eventType == 'Donly':
            D = self.solution[0] - self.Doffset
            Dright = D + self.plusD
            Dleft = D - self.minusD
            Bup = self.B + 2 * self.sigmaB / np.sqrt(self.nBpts)
            Bdown = self.B - 2 * self.sigmaB / np.sqrt(self.nBpts)
            Aup = self.A + 2 * self.sigmaA / np.sqrt(self.nApts)
            Adown = self.A - 2 * self.sigmaA / np.sqrt(self.nApts)
            
            plot([self.left, Dright], [Bup, Bup])
            plot([Dright, Dright], [Bup, Aup])
            plot([Dright, self.right], [Aup, Aup])
            
            plot([self.left, Dleft], [Bdown, Bdown])
            plot([Dleft, Dleft], [Bdown, Adown])
            plot([Dleft, self.right], [Adown, Adown])
            return
            
        if self.eventType == 'Ronly':
            R = self.solution[1] - self.Roffset
            Rright = R + self.plusR
            Rleft = R - self.minusR
            Bup = self.B + 2 * self.sigmaB / np.sqrt(self.nBpts)
            Bdown = self.B - 2 * self.sigmaB / np.sqrt(self.nBpts)
            Aup = self.A + 2 * self.sigmaA / np.sqrt(self.nApts)
            Adown = self.A - 2 * self.sigmaA / np.sqrt(self.nApts)
            
            plot([self.left, Rleft], [Aup, Aup])
            plot([Rleft, Rleft], [Aup, Bup])
            plot([Rleft, self.right], [Bup, Bup])
            
            plot([self.left, Rright], [Adown, Adown])
            plot([Rright, Rright], [Adown, Bdown])
            plot([Rright, self.right], [Bdown, Bdown])
            return
        
        if self.eventType == 'DandR':
            R = self.solution[1] - self.Roffset
            D = self.solution[0] - self.Doffset
            
            Rright = R + self.plusR
            Rleft = R - self.minusR
            Dright = D + self.plusD
            Dleft = D - self.minusD
            Bup = self.B + 2 * self.sigmaB / np.sqrt(self.nBpts)
            Bdown = self.B - 2 * self.sigmaB / np.sqrt(self.nBpts)
            Aup = self.A + 2 * self.sigmaA / np.sqrt(self.nApts)
            Adown = self.A - 2 * self.sigmaA / np.sqrt(self.nApts)
            
            plot([self.left, Dright], [Bup, Bup])
            plot([Dright, Dright], [Bup, Aup])
            plot([Dright, Rleft], [Aup, Aup])
            plot([Rleft, Rleft], [Aup, Bup])
            plot([Rleft, self.right], [Bup, Bup])
            
            plot([self.left, Dleft], [Bdown, Bdown])
            plot([Dleft, Dleft], [Bdown, Adown])
            plot([Dleft, Rright], [Adown, Adown])
            plot([Rright, Rright], [Adown, Bdown])
            plot([Rright, self.right], [Bdown, Bdown])
            return
    
    def reDrawMainPlot(self):
        self.mainPlot.clear()

        if self.showTimestampErrors.checkState():
            self.illustrateTimestampOutliers()

        # # Automatically show timestamp errors at final report
        # if self.minusD is not None or self.minusR is not None:
        #     self.illustrateTimestampOutliers()

        self.mainPlot.addItem(self.verticalCursor)

        self.mainPlot.plot(self.yValues)
            
        x = [i for i in range(self.dataLen) if self.yStatus[i] == INCLUDED]
        y = [self.yValues[i] for i in range(self.dataLen) if self.yStatus[i] == INCLUDED]
        self.mainPlot.plot(x, y, pen=None, symbol='o', 
                           symbolBrush=(0, 0, 255), symbolSize=6)
        
        x = [i for i in range(self.dataLen) if self.yStatus[i] == BASELINE]
        y = [self.yValues[i] for i in range(self.dataLen) if self.yStatus[i] == BASELINE]
        self.mainPlot.plot(x, y, pen=None, symbol='o', 
                           symbolBrush=(0, 200, 200), symbolSize=6)
        
        x = [i for i in range(self.dataLen) if self.yStatus[i] == SELECTED]
        y = [self.yValues[i] for i in range(self.dataLen) if self.yStatus[i] == SELECTED]
        self.mainPlot.plot(x, y, pen=None, symbol='o', 
                           symbolBrush=(255, 0, 0), symbolSize=10)
        
        if self.showSecondaryCheckBox.isChecked() and len(self.yRefStar) == self.dataLen:
            self.mainPlot.plot(self.yRefStar)
            right = min(self.dataLen, self.right+1)
            x = [i for i in range(self.left, right)]
            y = [self.yRefStar[i]for i in range(self.left, right)]            
            self.mainPlot.plot(x, y, pen=None, symbol='o', 
                               symbolBrush=(0, 255, 0), symbolSize=6)
            if len(self.smoothSecondary) > 0:
                self.mainPlot.plot(x, self.smoothSecondary, 
                                   pen=pg.mkPen((100, 100, 100), width=4), symbol=None)
                 
        if self.dRegion is not None:
            self.mainPlot.addItem(self.dRegion)
        if self.rRegion is not None:
            self.mainPlot.addItem(self.rRegion)

        if self.solution:
            self.drawSolution()

        if self.minusD is not None or self.minusR is not None:
            # We have data for drawing an envelope
            self.drawEnvelope()
        
    def showSelectedPoints(self, header):
        selPts = list(self.selectedPoints.keys())
        selPts.sort()
        self.showMsg(header + str(selPts))
     
    def doTrim(self):
        if len(self.selectedPoints) != 0:
            if len(self.selectedPoints) != 2:
                self.showInfo('Exactly two points must be selected for a trim operation')
                return
            self.showSelectedPoints('Data trimmed/selected using points: ')
            selPts = list(self.selectedPoints.keys())
            selPts.sort()
            self.left = selPts[0]
            self.right = selPts[1]
        else:
            # self.showInfo('All points will be selected (because no trim points specified)')
            self.showMsg('All data points were selected')
            self.left = 0
            self.right = self.dataLen - 1

        self.smoothSecondary = []
        
        if len(self.yRefStar) > 0:
            self.smoothSecondaryButton.setEnabled(True)
            self.numSmoothPointsEdit.setEnabled(True)
        
        for i in range(0, self.left):
            self.yStatus[i] = EXCLUDED
        for i in range(min(self.dataLen, self.right+1), self.dataLen):
            self.yStatus[i] = EXCLUDED
        for i in range(self.left, min(self.dataLen, self.right+1)):
            self.yStatus[i] = INCLUDED
        
        self.selectedPoints = {}
        self.reDrawMainPlot()
        self.doBlockIntegration.setEnabled(False)
        self.mainPlot.autoRange()        


def main():
    import traceback
    QtGui.QApplication.setStyle('fusion')
    app = QtGui.QApplication(sys.argv)
    
    # Save the current/proper sys.excepthook object
    # sys._excepthook = sys.excepthook
    saved_excepthook = sys.excepthook

    def exception_hook(exctype, value, tb):
        print('')
        print('=' * 30)
        print(value)
        print('=' * 30)
        print('')

        traceback.print_tb(tb)
        # Call the usual exception processor
        # sys._excepthook(exctype, value, tb)
        saved_excepthook(exctype, value, tb)
        # Exit if you prefer...
        # sys.exit(1)
        
    sys.excepthook = exception_hook
    
    form = SimplePlot()
    form.show()
    app.exec_()
    

if __name__ == '__main__':
    main()
