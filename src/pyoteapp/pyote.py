#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 20 15:32:13 2017

@author: Bob Anderson
"""

import datetime
import os
import sys

import numpy as np
import pyqtgraph as pg
import pyqtgraph.exporters as pex
import scipy.signal
from PyQt5 import QtCore
from PyQt5 import QtGui
from PyQt5.QtCore import QSettings, QPoint, QSize
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from pyqtgraph import PlotWidget

from pyoteapp import version
from pyoteapp import fixedPrecision as fp
from pyoteapp import gui
from pyoteapp.csvreader import readLightCurve
from pyoteapp.errorBarUtils import ciBars
from pyoteapp.errorBarUtils import createDurDistribution
from pyoteapp.errorBarUtils import edgeDistributionGenerator
from pyoteapp.noiseUtils import getCorCoefs
from pyoteapp.solverUtils import candidateCounter, solver
from pyoteapp.timestampUtils import convertTimeStringToTime
from pyoteapp.timestampUtils import convertTimeToTimeString
from pyoteapp.timestampUtils import getTimeStepAndOutliers

# The following module was created by typing
#    !pyuic5 simple-plot.ui -o gui.py
# in the IPython console

# Status of points and associated dot colors ---
SELECTED = 3  # big red
BASELINE = 2  # green
INCLUDED = 1  # blue
EXCLUDED = 0  # no dot

acfCoefThreshold = 0.05  # To match what is being done in R-OTE 4.5.4+

# There is a bug in pyqtgraph ImageExpoter, probably caused by new versions of PyQt5 returning
# float values for image rectangles.  Those floats were being given to numpy to create a matrix,
# and that was raising an exception.  Below is my 'cure', effected by overriding the internal
# methods of ImageExporter the manipulate width and height

class FixedImageExporter(pex.ImageExporter):
    def __init__(self, item):
        pex.ImageExporter.__init__(self, item)

    def makeWidthHeightInts(self):
        self.params['height'] = int(self.params['height'] + 1)  # The +1 is needed
        self.params['width'] = int(self.params['width'] + 1)

    def widthChanged(self):
        sr = self.getSourceRect()
        ar = float(sr.height()) / sr.width()
        self.params.param('height').setValue(int(self.params['width'] * ar))

    def heightChanged(self):
        sr = self.getSourceRect()
        ar = float(sr.width()) / sr.height()
        self.params.param('width').setValue(int(self.params['height'] * ar))

class CustomViewBox(pg.ViewBox):
    def __init__(self, *args, **kwds):
        pg.ViewBox.__init__(self, *args, **kwds)
        self.setMouseMode(self.RectMode)
        
    # re-implement right-click to zoom out
    def mouseClickEvent(self, ev):
        if ev.button() == QtCore.Qt.RightButton:
            self.autoRange()
            
    def mouseDragEvent(self, ev):
        if ev.button() == QtCore.Qt.RightButton:
            ev.ignore()
        else:
            pg.ViewBox.mouseDragEvent(self, ev)


class SimplePlot(QtGui.QMainWindow, gui.Ui_MainWindow):
    def __init__(self):
        super(SimplePlot, self).__init__()

        # Change pyqtgraph plots to be black on white
        pg.setConfigOption('background', (255, 255, 255))  # Do before any widgets drawn
        pg.setConfigOption('foreground', 'k')  # Do before any widgets drawn
        
        self.setupUi(self)

        self.setWindowTitle('PYOTE  Version: ' + version.version())

        # Button: Read light curve
        self.readData.clicked.connect(self.readDataFromFile)
        
        # CheckBox: Show secondary star
        self.showSecondaryCheckBox.clicked.connect(self.toggleDisplayOfSecondaryStar)
        
        # Button: Trim/Select data points
        self.setDataLimits.clicked.connect(self.doTrim)
        
        # Button: Smooth secondary
        self.smoothSecondaryButton.clicked.connect(self.smoothRefStar)
        
        # Button: Normalize around selected point
        self.normalizeButton.clicked.connect(self.normalize)
        
        # Button: Do block integration
        self.doBlockIntegration.clicked.connect(self.doIntegration)
        
        # Button: Perform baseline noise analysis
        self.doNoiseAnalysis.clicked.connect(self.processBaselineNoise)
        
        # Button: Perform event noise analysis (determine sigmaA only)
        self.computeSigmaA.clicked.connect(self.processEventNoise)
        
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
        
        # Experimental --- try to re-instantiate mainPlot Note: examine gui.py
        # to get this right after a relayout !!!!
        oldMainPlot = self.mainPlot
        self.mainPlot = PlotWidget(self.layoutWidget, 
                                   viewBox=CustomViewBox(border=(255, 255, 255)),
                                   enableMenu=False)
        self.mainPlot.setObjectName("mainPlot")
        self.horizontalLayout_5.addWidget(self.mainPlot)
        oldMainPlot.setParent(None)
        
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

    def exportBarPlots(self):
        if self.dBarPlotItem is None:
            self.showInfo('No error bar plots available yet')
            return
        # self.showInfo('Export bar plots requested')
        self.graphicFile, _ = QFileDialog.getSaveFileName(
                self,                                      # parent
                "Select filename for error bar plots",     # title for dialog
                self.settings.value('lightcurvedir', ""),  # starting directory
                "png files (*.png)")
        
        if self.graphicFile:
            self.graphicFile, _ = os.path.splitext(self.graphicFile)
            exporter = FixedImageExporter(self.dBarPlotItem)
            exporter.makeWidthHeightInts()
            exporter.export(self.graphicFile + '.D.png')
            
            exporter = FixedImageExporter(self.durBarPlotItem)
            exporter.makeWidthHeightInts()
            exporter.export(self.graphicFile + '.R-D.png')
            
            self.showInfo('Wrote to: ' + self.graphicFile)
        
    def exportGraphic(self):
        self.graphicFile, _ = QFileDialog.getSaveFileName(
                self,                                      # parent
                "Select filename for main plot",           # title for dialog
                self.settings.value('lightcurvedir', ""),  # starting directory
                "png files (*.png)")

        # self.showInfo('User selected: ' + self.graphicFile)

        if self.graphicFile:
            # exporter = pyqtgraph.exporters.ImageExporter(self.mainPlot.getPlotItem())
            exporter = FixedImageExporter(self.mainPlot.getPlotItem())
            exporter.makeWidthHeightInts()
            exporter.export(self.graphicFile)
            self.showInfo('Wrote to: ' + self.graphicFile)
        
    def initializeVariablesThatDontDependOnAfile(self):
        
        self.selectedPoints = {}  # Clear/declare 'selected points' dictionary
        self.baselineXvals = []
        self.baselineYvals = []

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
        
        if leftEdge < self.left or rightEdge > self.right:
            self.showInfo('The D region must be positioned within the trimmed data!')
            self.removePointSelections()
            self.reDrawMainPlot()
            return
        
        if self.rLimits:
            if rightEdge >= self.rLimits[0]:
                self.showInfo('The D region may not overlap (or come after) R region!')
                self.removePointSelections()
                self.reDrawMainPlot()
                return
            
        self.setDataLimits.setEnabled(False)
        
        self.dLimits = [leftEdge, rightEdge]
        
        if self.rLimits:
            self.DandR.setChecked(True)
        else:
            self.Donly.setChecked(True)
            
        self.dRegion = pg.LinearRegionItem(
                [leftEdge, rightEdge], movable=False, brush=(0, 200, 0, 50))
        self.dRegion.setZValue(-10)
        self.mainPlot.addItem(self.dRegion)
        
        self.showMsg('D zone selected: ' + str(selIndices))
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
        
        if leftEdge < self.left or rightEdge > self.right:
            self.showInfo('The R region must be positioned within the trimmed data!')
            self.removePointSelections()
            self.reDrawMainPlot()
            return
        
        if self.dLimits:
            if leftEdge <= self.dLimits[1]:
                self.showInfo('The R region may not overlap (or come before) D region!')
                self.removePointSelections()
                self.reDrawMainPlot()
                return
        
        self.setDataLimits.setEnabled(False)
        
        self.rLimits = [leftEdge, rightEdge]
        
        if self.dLimits:
            self.DandR.setChecked(True)
        else:
            self.Ronly.setChecked(True)
            
        self.rRegion = pg.LinearRegionItem(
                [leftEdge, rightEdge], movable=False, brush=(200, 0, 0, 50))
        self.rRegion.setZValue(-10)
        self.mainPlot.addItem(self.rRegion)
        
        self.showMsg('R zone selected: ' + str(selIndices))
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
        
        self.removePointSelections()
        self.normalizeButton.setEnabled(False)
        self.setDataLimits.setEnabled(False)
        self.reDrawMainPlot()
        
    def smoothRefStar(self):
        if (self.right - self.left) < 4:
            self.showInfo('The smoothing algorithm requires a minimum selection of 5 points')
            return
        
        self.showMsg('Smoothing of secondary star light curve performed')
        y = [self.yRefStar[i] for i in range(self.left, self.right+1)]
           
        try:
            if len(y) > 100:
                window = 101
            else:
                window = len(y)
                if window % 2 == 0:
                    window -= 1
            filteredY = scipy.signal.savgol_filter(np.array(y), window, 3)
            # filteredY = scipy.signal.savgol_filter(np.array(y), 25, 3)
            self.smoothSecondary = scipy.signal.savgol_filter(filteredY, window, 3)
            # self.smoothSecondary = scipy.signal.savgol_filter(filteredY, 25, 3)
            self.reDrawMainPlot()            
        except Exception as e:
            self.showMsg(str(e))
               
        self.smoothSecondaryButton.setEnabled(False)
        self.normalizeButton.setEnabled(True)
        
    def toggleDisplayOfSecondaryStar(self):
        self.reDrawMainPlot()
        self.mainPlot.autoRange()
        
    def showInfo(self, stuffToSay):
        QMessageBox.information(self, 'This will not be seen', stuffToSay)

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
        newFrame = []
        newTime = []
        newVal = []
        newRef = []
        
        p = p0 - span  # Start working toward the left
        while p > 0:
            avg = np.mean(self.yValues[p:(p+span)])
            newVal.insert(0, avg)
            avg = np.mean(self.yRefStar[p:(p+span)])
            newRef.insert(0, avg)
            newFrame.insert(0, self.yFrame[p])
            newTime.insert(0, self.yTimes[p])
            p = p - span
            
        p = p0  # Start working toward the right
        while p < self.dataLen - span:
            avg = np.mean(self.yValues[p:(p+span)])
            newVal.append(avg)
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
        self.yRefStar = np.array(newRef)
        self.yTimes = newTime[:]
        self.yFrame = newFrame[:]
        self.yStatus = [1 for _i in range(self.dataLen)]
        self.fillTableViewOfData()
        
        selPts.sort()
        self.showMsg('Block integration started at entry ' + str(selPts[0]) +
                     ' with block size of ' + str(selPts[1]-selPts[0]+1))
        
        self.timeDelta, self.outliers, self.errRate = getTimeStepAndOutliers(self.yTimes)
        self.showMsg('timeDelta: ' + fp.to_precision(self.timeDelta, 6) + ' seconds', blankLine=False)
        self.showMsg('timestamp error rate: ' + fp.to_precision(100 * self.errRate, 2) + '%')

        self.illustrateTimestampOutliers()
        
        self.doBlockIntegration.setEnabled(False)
        self.reDrawMainPlot()
        self.mainPlot.autoRange()
        
    def processClick(self, event):
        # This try/except handles case where user clicks in plot area before a
        # plot has been drawn.
        try:
            mousePoint = self.mainPlotViewBox.mapSceneToView(event.scenePos())
            index = round(mousePoint.x())
            if index in range(self.dataLen):                
                if event.button() == 1:  # left button clicked?
                    if self.yStatus[index] != 3:
                        # Save current status for possible undo (a later click)
                        self.selectedPoints[index] = self.yStatus[index]
                        self.yStatus[index] = 3  # Set color to 'selected'
                    else:
                        # Restore previous status (when originally clicked)
                        self.yStatus[index] = self.selectedPoints[index]
                        del(self.selectedPoints[index])
                self.reDrawMainPlot()  # Redraw plot to show selection change
                # Move the table view of data so that clicked point data is visible
                self.table.setCurrentCell(index, 0)
            else:
                pass  # Out of bounds clicks simply ignored
        except:
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
        
    def cellClick(self, row, column):
        entry = self.table.item(row, 0)
        self.highlightReading(int(entry.text()))
        
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

    def Dreport(self):
        D, _ = self.solution

        intD = int(D)  # So that we can do lookup in the data table

        noiseAsymmetry = self.snrA / self.snrB
        if (noiseAsymmetry > 0.7) and (noiseAsymmetry < 1.3):
            plusD = (self.deltaDhi95 - self.deltaDlo95) / 2
            minusD = plusD
        else:
            plusD = -self.deltaDlo95   # Deliberate 'inversion'
            minusD = self.deltaDhi95   # Deliberate 'inversion'
         
        # Save these for the 'envelope' plotter
        self.plusD = plusD
        self.minusD = minusD

        entryNum = intD
        frameNum = float(self.yFrame[intD])
        Dframe = D + frameNum - entryNum
        self.showMsg('D: %.2f {+%.2f,-%.2f} (frame number)' % (Dframe, plusD, minusD))
        ts = self.yTimes[int(D)]
        time = convertTimeStringToTime(ts)
        adjTime = time + (D - int(D)) * self.timeDelta
        ts = convertTimeToTimeString(adjTime)
        self.showMsg('D: %s  {+%.4f,-%.4f} seconds' % 
                     (ts, plusD * self.timeDelta, minusD * self.timeDelta)
                     )
        return adjTime
        
    def Rreport(self):
        _, R = self.solution
        # if R: R = R - self.Roffset
        noiseAsymmetry = self.snrA / self.snrB
        if (noiseAsymmetry > 0.7) and (noiseAsymmetry < 1.3):
            plusR = (self.deltaRhi95 - self.deltaRlo95) / 2
            minusR = plusR
        else:
            plusR = -self.deltaRlo95  # Deliberate 'inversion'
            minusR = self.deltaRhi95  # Deliberate 'inversion'
        
        # Save these for the 'envelope' plotter
        self.plusR = plusR
        self.minusR = minusR

        intR = int(R)
        entryNum = intR
        frameNum = float(self.yFrame[intR])
        Rframe = R + frameNum - entryNum
        self.showMsg('R: %.2f {+%.2f,-%.2f} (frame number)' % (Rframe, plusR, minusR))

        ts = self.yTimes[int(R)]
        time = convertTimeStringToTime(ts)
        adjTime = time + (R - int(R)) * self.timeDelta
        ts = convertTimeToTimeString(adjTime)
        self.showMsg('R: %s  {+%.4f,-%.4f} seconds' % 
                     (ts, plusR * self.timeDelta, minusR * self.timeDelta)
                     )
        return adjTime
        
    def finalReport(self):
        # self.showMsg('! ! The final report goes here ! !')
        # Grab the D and R values found and apply our timing convention
        D, R = self.solution
        # if D: D = D - self.Doffset
        # if R: R = R - self.Roffset
        
        self.showMsg('In the following report, 0.95 confidence interval error bars are used.')
        self.showMsg('B: %0.2f  {+/- %0.2f}' % (self.B, 2 * self.sigmaB))
        self.showMsg('A: %0.2f  {+/- %0.2f}' % (self.A, 2 * self.sigmaA))
        if self.A > 0:
            self.showMsg('nominal magDrop: %0.2f' % ((np.log10(self.B) - np.log10(self.A)) * 2.5))
        else:
            self.showMsg('magDrop calculation not possible because A is negative')
        self.showMsg('snr: %0.2f' % self.snrB)
        
        if self.eventType == 'Donly':
            self.Dreport()
            return

        if self.eventType == 'Ronly':
            self.Rreport()
            return
        
        if self.eventType == 'DandR':
            Dtime = self.Dreport()
            Rtime = self.Rreport()
            self.reportTimeValidity(D, R)
            plusDur = ((self.deltaDurhi95 - self.deltaDurlo95) / 2)
            minusDur = plusDur
            self.showMsg('Duration (R - D): %.4f {+%.4f,-%.4f} readings' % 
                         (R - D, plusDur, minusDur))
            plusDur = ((self.deltaDurhi95 - self.deltaDurlo95) / 2) * self.timeDelta
            minusDur = plusDur
            self.showMsg('Duration (R - D): %.4f {+%.4f,-%.4f} seconds' % 
                         (Rtime - Dtime, plusDur, minusDur))
            return
        
    def reportTimeValidity(self, D, R):
        intD = int(D)
        intR = int(R)
        dTime = convertTimeStringToTime(self.yTimes[intD])
        rTime = convertTimeStringToTime(self.yTimes[intR])
        
        # Here we check for a 'midnight transition'
        if rTime < dTime:
            rTime += 24 * 60 * 60
            self.showMsg('D and R enclose a transition through midnight')
            
        numEnclosedReadings = int(round((rTime - dTime) / self.timeDelta))
        self.showMsg('From timestamps at D and R, calculated %d readings.  From reading numbers, calculated %d readings.' % 
                     (numEnclosedReadings, intR - intD))
        if numEnclosedReadings == intR - intD:
            self.showMsg('Timestamps appear valid @ D and R')
        else:
            self.showMsg('! There is something wrong with timestamps at D and/or R or frames have been dropped !')
        
    def computeErrorBars(self):
        global dist
        
        # self.showMsg('Error calculation requested')
        
        self.snrB = (self.B - self.A) / self.sigmaB
        self.snrA = (self.B - self.A) / self.sigmaA
        # snr = min(snrB, snrA)
        snr = self.snrB  # A more reliable number
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
        for dist in distGen:
            if type(dist) == float:
                self.progressBar.setValue(dist * 100)
                QtGui.QApplication.processEvents()
                if self.cancelRequested:
                    self.cancelRequested = False
                    self.showMsg('Error bar calculation was cancelled')
                    self.progressBar.setValue(0)
                    self.finalReport()
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
        
        # global durDist
        durDist = createDurDistribution(dist)
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
                        labels={'bottom': 'Readings'})
        self.dBarPlotItem = pw.getPlotItem()
        pw.hideButtons()
        
        pw2 = PlotWidget(viewBox=CustomViewBox(border=(0, 0, 0)),
                         enableMenu=False, title='Distribution of duration (R - D) errors due to noise',
                         labels={'bottom': 'Readings'})
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
        
        self.showMsg('loDbar   @ .68 ci: %8.4f' % x1, blankLine=False)
        self.showMsg('hiDbar   @ .68 ci: %8.4f' % x2, blankLine=False)
        
        yp = max(y) * 0.25
        x1 = self.loDbar95-D
        pw.plot(x=[x1, x1], y=[0, yp], pen=pen)
        x2 = self.hiDbar95-D
        pw.plot(x=[x2, x2], y=[0, yp], pen=pen)
        
        self.showMsg('loDbar   @ .95 ci: %8.4f' % x1, blankLine=False)
        self.showMsg('hiDbar   @ .95 ci: %8.4f' % x2, blankLine=False)
        
        legend95 = '[%0.2f,%0.2f] @ 0.95' % (x1, x2)
        pw.plot(name=legend95)

        yp = max(y) * 0.15
        x1 = self.loDbar99 - D
        pw.plot(x=[x1, x1], y=[0, yp], pen=pen)
        x2 = self.hiDbar99 - D
        pw.plot(x=[x2, x2], y=[0, yp], pen=pen)

        self.showMsg('loDbar   @ .9973 ci: %8.4f' % x1, blankLine=False)
        self.showMsg('hiDbar   @ .9973 ci: %8.4f' % x2, blankLine=True)

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
        
        self.showMsg('loDurBar @ .68 ci: %8.4f' % x1, blankLine=False)
        self.showMsg('hiDurBar @ .68 ci: %8.4f' % x2, blankLine=False)
        
        yp = max(ydur) * 0.25
        x1 = self.loDurbar95
        pw2.plot(x=[x1, x1], y=[0, yp], pen=pen)
        x2 = self.hiDurbar95
        pw2.plot(x=[x2, x2], y=[0, yp], pen=pen)
        
        self.showMsg('loDurBar @ .95 ci: %8.4f' % x1, blankLine=False)
        self.showMsg('hiDurBar @ .95 ci: %8.4f' % x2, blankLine=False)
        
        legend95 = '[%0.2f,%0.2f] @ 0.95' % (x1, x2)
        pw2.plot(name=legend95)

        yp = max(ydur) * 0.15
        x1 = self.loDurbar99
        pw2.plot(x=[x1, x1], y=[0, yp], pen=pen)
        x2 = self.hiDurbar99
        pw2.plot(x=[x2, x2], y=[0, yp], pen=pen)

        self.showMsg('loDurBar @ .9973 ci: %8.4f' % x1, blankLine=False)
        self.showMsg('hiDurBar @ .9973 ci: %8.4f' % x2, blankLine=True)

        legend99 = '[%0.2f,%0.2f] @ 0.9973' % (x1, x2)
        pw2.plot(name=legend99)
        
        pw2.hideAxis('left')
        
        self.writeBarPlots.setEnabled(True)
        self.finalReport()
        self.reDrawMainPlot()  # To add envelope to solution
        
    def findEvent(self):
        global yValues, left, right, sigmaB, sigmaA  # For debugging and experiments
        yValues = self.yValues[0:self.dataLen]
        left = self.left
        right = self.right
        sigmaB = self.sigmaB
        sigmaA = self.sigmaA
        
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
        
        self.showMsg('Number of candidate solutions: ' + str(numCandidates) +
                     ' (' + candFrom + ')')
    
        runSolver = True
        if numCandidates > 10000:
            msg = 'There are ' + str(numCandidates) + ' candidates in the solution set. '
            msg = msg + 'Do you wish to continue?'
            self.showQuery(msg, 'Your chance to narrow the potential candidates')
        
            if self.queryRetVal == QMessageBox.Yes:
                self.showMsg('Yes was clicked --- starting solution search...')
                self.solution = None
                self.reDrawMainPlot()
            else:
                self.showMsg('"No" was clicked, so solver will be skipped')
                runSolver = False
        
        if runSolver:
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
                    # Here we should update progress bar and check for cancellation
                    self.progressBar.setValue(item[1] * 100)
                    QtGui.QApplication.processEvents()
                    if self.cancelRequested:
                        self.cancelRequested = False
                        runSolver = False
                        self.showMsg('Solution search was cancelled')
                        self.progressBar.setValue(0)
                        break
                elif item[0] == 'no event present':
                    self.showMsg('No event fitting search criteria could be found.')
                    self.progressBar.setValue(0)
                    runSolver = False
                    break
                else:
                    self.progressBar.setValue(0)
                    self.solution = item[0]
                    self.B = item[1]
                    self.A = item[2]
                    self.dRegion = None
                    self.rRegion = None
                    self.dLimits = None
                    self.rLimits = None
            
        if runSolver and self.solution:
            D, R = self.solution
            if D is not None:
                D = round(D, 2)
            if R is not None:
                R = round(R, 2)
            self.solution = (D, R)
            if self.eventType == 'DandR':
                ans = '(%.2f,%.2f) B: %.2f  A: %.2f' % (D, R, self.B, self.A)
            elif self.eventType == 'Donly':
                ans = '(%.2f,None) B: %.2f  A: %.2f' % (D, self.B, self.A)
            elif self.eventType == 'Ronly':
                ans = '(None,%.2f) B: %.2f  A: %.2f' % (R, self.B, self.A)
            else:
                raise Exception('Undefined event type')
            self.showMsg('Raw solution (debug output): ' + ans)
        elif runSolver:
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
        
        global timestamps  # debug
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
            dirpath, _ = os.path.split(self.filename)
            self.logFile, _ = os.path.splitext(self.filename)
            self.logFile = self.logFile + '.log'
            
            curDateTime = datetime.datetime.today().ctime()
            self.showMsg('')
            self.showMsg('#' * 20 + ' Session started: ' + curDateTime + '  ' + '#' * 20)
        
            # Make the directory 'sticky'
            self.settings.setValue('lightcurvedir', dirpath)
            self.showMsg('filename: ' + self.filename, bold=True, color="red")

            try:
                self.outliers = []
                frame, time, value, secondary, headers = readLightCurve(self.filename)
                values = [float(item) for item in value]
                if frame == []:
                    # This is a raw data file, imported for test purposes
                    global raw
                    raw = values
                    self.showInfo('raw data file read')
                    return
                refStar = [float(item) for item in secondary]
                if secondary:
                    self.showSecondaryCheckBox.setEnabled(True)
                    self.showSecondaryCheckBox.setChecked(True)
                timestamps = time[:]  # debug
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
                    
                self.dataLen = len(self.yValues)
                self.yFrame = frame[:]
                self.yFrameCopy = frame[:]
                
                # Automatically select all points
                self.yStatus = [1 for _i in range(self.dataLen)]  # 1 means included
                self.left = 0
                self.right = self.dataLen - 1
                
                self.reDrawMainPlot()
                self.mainPlot.autoRange()
                self.setDataLimits.setEnabled(True)
                self.writePlot.setEnabled(True)
                self.doNoiseAnalysis.setEnabled(True)
                self.computeSigmaA.setEnabled(True)
                self.doBlockIntegration.setEnabled(True)
                self.startOver.setEnabled(True)
                self.fillTableViewOfData()
                self.timeDelta, self.outliers, self.errRate = getTimeStepAndOutliers(self.yTimes)
                self.showMsg('timeDelta: ' + fp.to_precision(self.timeDelta, 6) + ' seconds', blankLine=False)
                self.showMsg('timestamp error rate: ' + fp.to_precision(100 * self.errRate, 2) + '%')

                self.illustrateTimestampOutliers()
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
    
    def processEventNoise(self):
        if len(self.selectedPoints) != 2:
            self.showInfo('Exactly two points must be selected for this operation')
            return
        selPts = self.selectedPoints.keys()
        left = int(min(selPts))
        right = int(max(selPts))
        if (right - left) < 9:
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
            self.showSelectedPoints('Points selected for noise analysis: ')
            self.doNoiseAnalysis.setEnabled(True)
            self.computeSigmaA.setEnabled(True)
        
        self.removePointSelections()
        _, self.numNApts, self.sigmaA = getCorCoefs(self.eventXvals, self.eventYvals)
        self.showMsg('Event noise analysis done using ' + str(self.numNApts) + 
                     ' points ---  sigmaA: ' + fp.to_precision(self.sigmaA, 4))
        
        self.reDrawMainPlot()
        
    def processBaselineNoise(self):
        
        if len(self.selectedPoints) != 2:
            self.showInfo('Exactly two points must be selected for this operation')
            return
        selPts = self.selectedPoints.keys()
        left = int(min(selPts))
        right = int(max(selPts))
        if (right - left) < 14:
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
            self.showSelectedPoints('Points selected for noise analysis: ')
            self.doNoiseAnalysis.setEnabled(True)
            self.computeSigmaA.setEnabled(True)
        
        self.removePointSelections()
        
        # For experiments in console, write data into the global namespace
        global baseX, baseY
        baseX = self.baselineXvals
        baseY = self.baselineYvals
        
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
        
        if self.sigmaA is None:
            self.sigmaA = self.sigmaB
            
        self.reDrawMainPlot()
                
        self.locateEvent.setEnabled(True)
        self.markDzone.setEnabled(True)
        self.markRzone.setEnabled(True)
        self.minEventEdit.setEnabled(True)
        self.maxEventEdit.setEnabled(True)
    
    def removePointSelections(self):
        for i, oldStatus in self.selectedPoints.items():
            self.yStatus[i] = oldStatus
        self.selectedPoints = {}
        
    def disableAllButtons(self):
        self.showSecondaryCheckBox.setEnabled(False)
        self.normalizeButton.setEnabled(False)
        self.smoothSecondaryButton.setEnabled(False)
        self.setDataLimits.setEnabled(False)      
        self.doBlockIntegration.setEnabled(False)    
        self.doNoiseAnalysis.setEnabled(False)
        self.locateEvent.setEnabled(False)
        self.calcErrBars.setEnabled(False)
        self.startOver.setEnabled(False)
        self.markDzone.setEnabled(False)
        self.markRzone.setEnabled(False)
        self.minEventEdit.setEnabled(False)
        self.maxEventEdit.setEnabled(False)
        self.writeBarPlots.setEnabled(False)
       
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
        
        # Enable the initial set of buttons (allowed operations)
        self.startOver.setEnabled(True)
        self.doBlockIntegration.setEnabled(True)
        self.setDataLimits.setEnabled(True)
        self.doNoiseAnalysis.setEnabled(True)
        self.computeSigmaA.setEnabled(True)
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
        self.illustrateTimestampOutliers()
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
            
    def drawEnvelope(self):
        def plot(x, y):
            self.mainPlot.plot(x, y, pen=pg.mkPen((150, 100, 100), width=2), symbol=None)
        
        if self.solution is None:
            return
        
        if self.eventType == 'Donly':
            nBpts = self.solution[0] - self.left
            if nBpts < 1:
                nBpts = 1
            
            nApts = self.right - self.solution[0] - 1
            if nApts < 1:
                nApts = 1
            
            D = self.solution[0] - self.Doffset
            Dright = D + self.plusD
            Dleft = D - self.minusD
            Bup = self.B + 2 * self.sigmaB / np.sqrt(nBpts)
            Bdown = self.B - 2 * self.sigmaB / np.sqrt(nBpts)
            Aup = self.A + 2 * self.sigmaA / np.sqrt(nApts)
            Adown = self.A - 2 * self.sigmaA / np.sqrt(nApts)
            
            plot([self.left, Dright], [Bup, Bup])
            plot([Dright, Dright], [Bup, Aup])
            plot([Dright, self.right], [Aup, Aup])
            
            plot([self.left, Dleft], [Bdown, Bdown])
            plot([Dleft, Dleft], [Bdown, Adown])
            plot([Dleft, self.right], [Adown, Adown])
            return
            
        if self.eventType == 'Ronly':
            nBpts = self.right - self.solution[1]
            if nBpts < 1:
                nBpts = 1
            
            nApts = self.solution[1] - self.left
            if nApts < 1:
                nApts = 1
            
            R = self.solution[1] - self.Roffset
            Rright = R + self.plusR
            Rleft = R - self.minusR
            Bup = self.B + 2 * self.sigmaB / np.sqrt(nBpts)
            Bdown = self.B - 2 * self.sigmaB / np.sqrt(nBpts)
            Aup = self.A + 2 * self.sigmaA / np.sqrt(nApts)
            Adown = self.A - 2 * self.sigmaA / np.sqrt(nApts)
            
            plot([self.left, Rleft], [Aup, Aup])
            plot([Rleft, Rleft], [Aup, Bup])
            plot([Rleft, self.right], [Bup, Bup])
            
            plot([self.left, Rright], [Adown, Adown])
            plot([Rright, Rright], [Adown, Bdown])
            plot([Rright, self.right], [Bdown, Bdown])
            return
        
        if self.eventType == 'DandR':
            nBpts = self.right - self.solution[1] + self.solution[0] - self.left
            if nBpts < 1:
                nBpts = 1
            
            nApts = self.solution[1] - self.left + self.right - self.solution[0] - 1
            if nApts < 1:
                nApts = 1
            
            R = self.solution[1] - self.Roffset
            D = self.solution[0] - self.Doffset
            
            Rright = R + self.plusR
            Rleft = R - self.minusR
            Dright = D + self.plusD
            Dleft = D - self.minusD
            Bup = self.B + 2 * self.sigmaB / np.sqrt(nBpts)
            Bdown = self.B - 2 * self.sigmaB / np.sqrt(nBpts)
            Aup = self.A + 2 * self.sigmaA / np.sqrt(nApts)
            Adown = self.A - 2 * self.sigmaA / np.sqrt(nApts)
            
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
        self.mainPlot.plot(self.yValues)
        
        if self.solution:
            # self.showInfo('TBD --- solution plot')
            self.drawSolution()
            
        if self.minusD is not None or self.minusR is not None:
            # We have data for drawing an envelope
            self.drawEnvelope()
            
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
            # right = self.right
            x = [i for i in range(self.left, right)]
            y = [self.yRefStar[i]for i in range(self.left, right)]            
            self.mainPlot.plot(x, y, pen=None, symbol='o', 
                               symbolBrush=(0, 255, 0), symbolSize=6)
            if len(self.smoothSecondary) > 0:
                self.mainPlot.plot(x, self.smoothSecondary, 
                                   pen=pg.mkPen((100, 100, 100), width=4), symbol=None)
                 
        self.illustrateTimestampOutliers()
        
        if self.dRegion is not None:
            self.mainPlot.addItem(self.dRegion)
        if self.rRegion is not None:
            self.mainPlot.addItem(self.rRegion)            
        
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
    sys._excepthook = sys.excepthook

    def exception_hook(exctype, value, tb):
        print('')
        print('=' * 30)
        print(value)
        print('=' * 30)
        print('')

        traceback.print_tb(tb)
        # Call the usual exception processor
        sys._excepthook(exctype, value, tb)
        # Exit if you prefer...
        # sys.exit(1)
        
    sys.excepthook = exception_hook
    
    form = SimplePlot()
    form.show()
    app.exec_()
    
if __name__ == '__main__':
    main()
