"""
Created on Sat May 20 15:32:13 2017

@author: Bob Anderson
"""
# import pickle
# import math
import subprocess
from pathlib import Path

MIN_SIGMA = 0.1

import datetime
import os
import sys
import platform

from openpyxl import load_workbook

from math import trunc, floor
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Qt5Agg')

from pyoteapp.showVideoFrames import readAviFile
from pyoteapp.showVideoFrames import readSerFile
from pyoteapp.showVideoFrames import readFitsFile
from pyoteapp.showVideoFrames import readAavFile

from pyoteapp.false_positive import compute_drops, noise_gen_jit, simple_convolve
import numpy as np
import pyqtgraph as pg
import pyqtgraph.exporters as pex
import scipy.signal
# from scipy.stats import pearsonr as pearson
import PyQt5
from PyQt5 import QtCore, QtWidgets
from PyQt5 import QtGui
from PyQt5.QtCore import QSettings, QPoint, QSize
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QDialog
from pyqtgraph import PlotWidget

from pyoteapp import version
from pyoteapp import fixedPrecision as fp
from pyoteapp import gui
from pyoteapp import timestampDialog
from pyoteapp import helpDialog
from pyoteapp import exponentialEdgeUtilities as ex
from pyoteapp.checkForNewerVersion import getMostRecentVersionOfPyOTEViaJson
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
from pyoteapp.blockIntegrateUtils import mean_std_versus_offset
from pyoteapp.iterative_logl_functions import locate_event_from_d_and_r_ranges
from pyoteapp.iterative_logl_functions import find_best_event_from_min_max_size
from pyoteapp.iterative_logl_functions import find_best_r_only_from_min_max_size
from pyoteapp.iterative_logl_functions import find_best_d_only_from_min_max_size
from pyoteapp.subframe_timing_utilities import generate_underlying_lightcurve_plots, fresnel_length_km
from pyoteapp.subframe_timing_utilities import time_correction, intensity_at_time

cursorAlert = pyqtSignal()

# The gui module was created by typing
#    !pyuic5 pyote.ui -o gui.py
# in the IPython console while in pyoteapp directory

# The timestampDialog module was created by typing
#    !pyuic5 timestamp_dialog_alt.ui -o timestampDialog.py
# in the IPython console while in pyoteapp directory

# The help-dialog module was created by typing
#    !pyuic5 helpDialog.ui -o helpDialog.py
# in the IPython console while in pyoteapp directory

# Status of points and associated dot colors ---
EVENT = 4  # Same as baseline
SELECTED = 3  # big red
BASELINE = 2  # orangish
INCLUDED = 1  # blue
EXCLUDED = 0  # no dot

LINESIZE = 2

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


class TimestampAxis(pg.AxisItem):

    def tickStrings(self, values, scale, spacing):
        return [self.getTimestampString(val) for val in values]

    def setFetcher(self, timestampFetch):
        self.getTimestampString = timestampFetch


class CustomViewBox(pg.ViewBox):
    def __init__(self, *args, **kwds):
        pg.ViewBox.__init__(self, *args, **kwds)
        self.setMouseMode(self.RectMode)

    # re-implement right-click to zoom out
    def mouseClickEvent(self, ev):
        if ev.button() == QtCore.Qt.MouseButton.RightButton:
            self.autoRange()
            mouseSignal.emit()

    def mouseDragEvent(self, ev, axis=None):
        if ev.button() == QtCore.Qt.MouseButton.RightButton:
            ev.ignore()
        else:
            pg.ViewBox.mouseDragEvent(self, ev, axis)
            mouseSignal.emit()


class TSdialog(QDialog, timestampDialog.Ui_manualTimestampDialog):
    def __init__(self):
        super(TSdialog, self).__init__()
        self.setupUi(self)


class HelpDialog(QDialog, helpDialog.Ui_Dialog):
    def __init__(self):
        super(HelpDialog, self).__init__()
        self.setupUi(self)


# class SimplePlot(QtGui.QMainWindow, gui.Ui_MainWindow):
class SimplePlot(PyQt5.QtWidgets.QMainWindow, gui.Ui_MainWindow):
    def __init__(self, csv_file):
        super(SimplePlot, self).__init__()

        self.yTimes = []

        self.dataLen = None
        self.left = None
        self.right = None
        self.yValues = None

        # This is an externally supplied csv file path (probably from PyMovie)
        self.externalCsvFilePath = csv_file

        self.homeDir = os.path.split(__file__)[0]

        # Change pyqtgraph plots to be black on white
        pg.setConfigOption('background', (255, 255, 255))  # Do before any widgets drawn
        pg.setConfigOption('foreground', 'k')  # Do before any widgets drawn
        pg.setConfigOptions(imageAxisOrder='row-major')

        self.setupUi(self)

        self.setWindowTitle('PYOTE  Version: ' + version.version())

        self.skipNormalization = False  # A flag used to prevent inifinite recursion in self.refrawMainPlot

        self.suppressNormalization = False

        self.targetIndex = 0

        self.targetCheckBoxes = [self.targetCheckBox_1, self.targetCheckBox_2, self.targetCheckBox_3,
                                 self.targetCheckBox_4, self.targetCheckBox_5, self.targetCheckBox_6,
                                 self.targetCheckBox_7, self.targetCheckBox_8, self.targetCheckBox_9,
                                 self.targetCheckBox_10]

        self.showCheckBoxes = [self.showCheckBox_1, self.showCheckBox_2, self.showCheckBox_3,
                               self.showCheckBox_4, self.showCheckBox_5, self.showCheckBox_6,
                               self.showCheckBox_7, self.showCheckBox_8, self.showCheckBox_9,
                               self.showCheckBox_10]

        self.lightcurveTitles = [self.lightcurveTitle_1, self.lightcurveTitle_2, self.lightcurveTitle_3,
                                 self.lightcurveTitle_4, self.lightcurveTitle_5, self.lightcurveTitle_6,
                                 self.lightcurveTitle_7, self.lightcurveTitle_8, self.lightcurveTitle_9,
                                 self.lightcurveTitle_10]

        self.yOffsetSpinBoxes = [self.yOffsetSpinBox_1, self.yOffsetSpinBox_2, self.yOffsetSpinBox_3,
                                 self.yOffsetSpinBox_4, self.yOffsetSpinBox_5, self.yOffsetSpinBox_6,
                                 self.yOffsetSpinBox_7, self.yOffsetSpinBox_8, self.yOffsetSpinBox_9,
                                 self.yOffsetSpinBox_10]

        self.referenceCheckBoxes = [self.referenceCheckBox_1, self.referenceCheckBox_2, self.referenceCheckBox_3,
                                    self.referenceCheckBox_4, self.referenceCheckBox_5, self.referenceCheckBox_6,
                                    self.referenceCheckBox_7, self.referenceCheckBox_8, self.referenceCheckBox_9,
                                    self.referenceCheckBox_10]

        self.colorStyle = ['color: rgb(255,0,0)', 'color: rgb(160,32,255)', 'color: rgb(80,208,255)',
                           'color: rgb(96,255,128)', 'color: rgb(255,224,32)', 'color: rgb(255,160,16)',
                           'color: rgb(160,128,96)', 'color: rgb(64,64,64)', 'color: rgb(255,208,160)',
                           'color: rgb(0,128,0)']

        self.colorBlobs = [self.colorBlob0, self.colorBlob1, self.colorBlob2,
                           self.colorBlob3, self.colorBlob4, self.colorBlob5,
                           self.colorBlob6, self.colorBlob7, self.colorBlob8,
                           self.colorBlob9]

        self.xOffsetSpinBoxes = [self.xOffsetSpinBox_1, self.xOffsetSpinBox_2, self.xOffsetSpinBox_3,
                                 self.xOffsetSpinBox_4, self.xOffsetSpinBox_5, self.xOffsetSpinBox_6,
                                 self.xOffsetSpinBox_7, self.xOffsetSpinBox_8, self.xOffsetSpinBox_9,
                                 self.xOffsetSpinBox_10]

        self.initializeLightcurvePanel()

        self.smoothingLabel.installEventFilter(self)

        self.yOffsetLabel1.installEventFilter(self)
        self.yOffsetLabel2.installEventFilter(self)
        self.yOffsetLabel3.installEventFilter(self)
        self.yOffsetLabel4.installEventFilter(self)
        self.yOffsetLabel5.installEventFilter(self)
        self.yOffsetLabel6.installEventFilter(self)
        self.yOffsetLabel7.installEventFilter(self)
        self.yOffsetLabel8.installEventFilter(self)
        self.yOffsetLabel9.installEventFilter(self)
        self.yOffsetLabel10.installEventFilter(self)

        self.xOffsetLabel1.installEventFilter(self)
        self.xOffsetLabel2.installEventFilter(self)
        self.xOffsetLabel3.installEventFilter(self)
        self.xOffsetLabel4.installEventFilter(self)
        self.xOffsetLabel5.installEventFilter(self)
        self.xOffsetLabel6.installEventFilter(self)
        self.xOffsetLabel7.installEventFilter(self)
        self.xOffsetLabel8.installEventFilter(self)
        self.xOffsetLabel9.installEventFilter(self)
        self.xOffsetLabel10.installEventFilter(self)

        self.stepBy2radioButton.clicked.connect(self.processStepBy2)
        self.stepBy10radioButton.clicked.connect(self.processStepBy10)
        self.stepBy100radioButton.clicked.connect(self.processStepBy100)

        self.stepByButtonGroup = QtWidgets.QButtonGroup()
        self.stepByButtonGroup.addButton(self.stepBy2radioButton)
        self.stepByButtonGroup.addButton(self.stepBy10radioButton)
        self.stepByButtonGroup.addButton(self.stepBy100radioButton)

        self.yOffsetStep10radioButton.clicked.connect(self.processYoffsetStepBy10)
        self.yOffsetStep100radioButton.clicked.connect(self.processYoffsetStepBy100)
        self.yOffsetStep1000radioButton.clicked.connect(self.processYoffsetStepBy1000)

        self.offsetStepButtonGroup = QtWidgets.QButtonGroup()
        self.offsetStepButtonGroup.addButton(self.yOffsetStep10radioButton)
        self.offsetStepButtonGroup.addButton(self.yOffsetStep100radioButton)
        self.offsetStepButtonGroup.addButton(self.yOffsetStep1000radioButton)

        self.yOffsetStep10radioButton.setChecked(True)

        self.targetCheckBox_1.clicked.connect(self.processTargetSelection1)
        self.targetCheckBox_2.clicked.connect(self.processTargetSelection2)
        self.targetCheckBox_3.clicked.connect(self.processTargetSelection3)
        self.targetCheckBox_4.clicked.connect(self.processTargetSelection4)
        self.targetCheckBox_5.clicked.connect(self.processTargetSelection5)
        self.targetCheckBox_6.clicked.connect(self.processTargetSelection6)
        self.targetCheckBox_7.clicked.connect(self.processTargetSelection7)
        self.targetCheckBox_8.clicked.connect(self.processTargetSelection8)
        self.targetCheckBox_9.clicked.connect(self.processTargetSelection9)
        self.targetCheckBox_10.clicked.connect(self.processTargetSelection10)

        self.showCheckBox_1.clicked.connect(self.processShowSelection)
        self.showCheckBox_2.clicked.connect(self.processShowSelection)
        self.showCheckBox_3.clicked.connect(self.processShowSelection)
        self.showCheckBox_4.clicked.connect(self.processShowSelection)
        self.showCheckBox_5.clicked.connect(self.processShowSelection)
        self.showCheckBox_6.clicked.connect(self.processShowSelection)
        self.showCheckBox_7.clicked.connect(self.processShowSelection)
        self.showCheckBox_8.clicked.connect(self.processShowSelection)
        self.showCheckBox_9.clicked.connect(self.processShowSelection)
        self.showCheckBox_10.clicked.connect(self.processShowSelection)

        self.referenceCheckBox_1.clicked.connect(self.processReferenceSelection1)
        self.referenceCheckBox_2.clicked.connect(self.processReferenceSelection2)
        self.referenceCheckBox_3.clicked.connect(self.processReferenceSelection3)
        self.referenceCheckBox_4.clicked.connect(self.processReferenceSelection4)
        self.referenceCheckBox_5.clicked.connect(self.processReferenceSelection5)
        self.referenceCheckBox_6.clicked.connect(self.processReferenceSelection6)
        self.referenceCheckBox_7.clicked.connect(self.processReferenceSelection7)
        self.referenceCheckBox_8.clicked.connect(self.processReferenceSelection8)
        self.referenceCheckBox_9.clicked.connect(self.processReferenceSelection9)
        self.referenceCheckBox_10.clicked.connect(self.processReferenceSelection10)

        self.yOffsetSpinBox_1.editingFinished.connect(self.processYoffsetChange)
        self.yOffsetSpinBox_2.editingFinished.connect(self.processYoffsetChange)
        self.yOffsetSpinBox_3.editingFinished.connect(self.processYoffsetChange)
        self.yOffsetSpinBox_4.editingFinished.connect(self.processYoffsetChange)
        self.yOffsetSpinBox_5.editingFinished.connect(self.processYoffsetChange)
        self.yOffsetSpinBox_6.editingFinished.connect(self.processYoffsetChange)
        self.yOffsetSpinBox_7.editingFinished.connect(self.processYoffsetChange)
        self.yOffsetSpinBox_8.editingFinished.connect(self.processYoffsetChange)
        self.yOffsetSpinBox_9.editingFinished.connect(self.processYoffsetChange)
        self.yOffsetSpinBox_10.editingFinished.connect(self.processYoffsetChange)

        self.xOffsetSpinBox_1.editingFinished.connect(self.processXoffsetChange)
        self.xOffsetSpinBox_2.editingFinished.connect(self.processXoffsetChange)
        self.xOffsetSpinBox_3.editingFinished.connect(self.processXoffsetChange)
        self.xOffsetSpinBox_4.editingFinished.connect(self.processXoffsetChange)
        self.xOffsetSpinBox_5.editingFinished.connect(self.processXoffsetChange)
        self.xOffsetSpinBox_6.editingFinished.connect(self.processXoffsetChange)
        self.xOffsetSpinBox_7.editingFinished.connect(self.processXoffsetChange)
        self.xOffsetSpinBox_8.editingFinished.connect(self.processXoffsetChange)
        self.xOffsetSpinBox_9.editingFinished.connect(self.processXoffsetChange)
        self.xOffsetSpinBox_10.editingFinished.connect(self.processXoffsetChange)

        self.smoothingIntervalSpinBox.editingFinished.connect(self.newRedrawMainPlot)

        self.LC1 = []
        self.LC2 = []
        self.LC3 = []
        self.LC4 = []
        self.yRefStar = []

        self.dotSizeSpinner.valueChanged.connect(self.newRedrawMainPlot)

        # This object is used to display tooltip help in a separate
        # modeless dialog box.
        self.helperThing = HelpDialog()

        self.helpButton.clicked.connect(self.helpButtonClicked)
        self.helpButton.installEventFilter(self)

        self.tutorialButton.clicked.connect(self.tutorialButtonClicked)
        self.tutorialButton.installEventFilter(self)

        self.lightcurvesHelpButton.clicked.connect(self.lightcurvesHelpButtonClicked)
        self.lightcurvesHelpButton.installEventFilter(self)

        self.minEventLabel.installEventFilter(self)
        self.maxEventLabel.installEventFilter(self)

        self.bkgndRegionLimits = []
        self.bkgndRegions = []

        self.ne3ExplanationButton.clicked.connect(self.ne3ExplanationClicked)
        self.ne3ExplanationButton.installEventFilter(self)

        self.yPositionLabel.installEventFilter(self)

        self.dnrOffRadioButton.installEventFilter(self)
        self.dnrLowRadioButton.installEventFilter(self)
        self.dnrMiddleRadioButton.installEventFilter(self)
        self.dnrHighRadioButton.installEventFilter(self)

        self.dnrLowDtcLabel.installEventFilter(self)
        self.dnrLowRtcLabel.installEventFilter(self)
        self.dnrMiddleDtcLabel.installEventFilter(self)
        self.dnrMiddleRtcLabel.installEventFilter(self)
        self.dnrHighDtcLabel.installEventFilter(self)
        self.dnrHighRtcLabel.installEventFilter(self)

        self.exponentialDtheoryPts = None
        self.exponentialRtheoryPts = None

        self.exponentialDinitialX = 0
        self.exponentialRinitialX = 0

        self.exponentialDedge = 0
        self.exponentialRedge = 0

        self.userDeterminedBaselineStats = False
        self.userDeterminedEventStats = False
        self.userTrimInEffect = False

        self.markBaselineRegionButton.clicked.connect(self.markBaselineRegion)
        self.markBaselineRegionButton.installEventFilter(self)

        self.normMarkBaselineRegionButton.clicked.connect(self.markBaselineRegion)
        self.normMarkBaselineRegionButton.installEventFilter(self)

        self.pymovieDataColumnPrefixComboBox.currentTextChanged.connect(self.handlePymovieColumnChange)

        self.clearBaselineRegionsButton.clicked.connect(self.clearBaselineRegions)
        self.clearBaselineRegionsButton.installEventFilter(self)

        self.clearMetricPointsButton.clicked.connect(self.clearBaselineRegions)
        self.clearMetricPointsButton.installEventFilter(self)

        self.calcStatsFromBaselineRegionsButton.clicked.connect(self.calcBaselineStatisticsFromMarkedRegions)
        self.calcStatsFromBaselineRegionsButton.installEventFilter(self)

        self.calcDetectabilityButton.clicked.connect(self.calcDetectability)
        self.calcDetectabilityButton.installEventFilter(self)

        self.eventDurLabel.installEventFilter(self)
        self.obsDurLabel.installEventFilter(self)
        self.durStepLabel.installEventFilter(self)
        self.detectMagDropLabel.installEventFilter(self)

        # Checkbox: Use manual timestamp entry
        self.manualTimestampCheckBox.clicked.connect(self.toggleManualEntryButton)
        self.manualTimestampCheckBox.installEventFilter(self)

        # Button: Manual timestamp entry
        self.manualEntryPushButton.clicked.connect(self.doManualTimestampEntry)
        self.manualEntryPushButton.installEventFilter(self)

        # Button: Info
        self.infoButton.clicked.connect(self.openHelpFile)
        self.infoButton.installEventFilter(self)

        # Button: Read light curve
        self.readData.clicked.connect(self.readDataFromFile)
        self.readData.installEventFilter(self)

        # Checkbox: Show timestamp errors
        self.showTimestampErrors.clicked.connect(self.toggleDisplayOfTimestampErrors)
        self.showTimestampErrors.installEventFilter(self)

        self.showTimestampsCheckBox.installEventFilter(self)

        # Checkbox: Show underlying lightcurve
        self.showUnderlyingLightcurveCheckBox.installEventFilter(self)
        self.showUnderlyingLightcurveCheckBox.clicked.connect(self.newRedrawMainPlot)

        self.showCameraResponseCheckBox.installEventFilter(self)
        self.showCameraResponseCheckBox.clicked.connect(self.newRedrawMainPlot)

        # Checkbox: Show error bars
        self.showErrBarsCheckBox.installEventFilter(self)
        self.showErrBarsCheckBox.clicked.connect(self.newRedrawMainPlot)

        # Checkbox: Show D and R edges
        self.showEdgesCheckBox.installEventFilter(self)
        self.showEdgesCheckBox.clicked.connect(self.newRedrawMainPlot)

        # Checkbox: Do OCR check
        self.showOCRcheckFramesCheckBox.installEventFilter(self)

        # line size
        self.lineWidthLabel.installEventFilter(self)
        self.lineWidthSpinner.valueChanged.connect(self.newRedrawMainPlot)

        # plotHelpButton
        self.plotHelpButton.clicked.connect(self.plotHelpButtonClicked)
        self.plotHelpButton.installEventFilter(self)

        # detectabilityDelpButton
        self.detectabilityHelpButton.clicked.connect(self.detectabilityHelpButtonClicked)
        self.detectabilityHelpButton.installEventFilter(self)

        # Button: Trim/Select data points
        self.setDataLimits.clicked.connect(self.doTrim)
        self.setDataLimits.installEventFilter(self)

        # Button: Do block integration
        self.doBlockIntegration.clicked.connect(self.doIntegration)
        self.doBlockIntegration.installEventFilter(self)

        # Button: Accept integration
        self.acceptBlockIntegration.clicked.connect(self.applyIntegration)
        self.acceptBlockIntegration.installEventFilter(self)

        # Button: Validate a potential single point event
        self.singlePointDropButton.clicked.connect(self.validateSinglePointDrop)
        self.singlePointDropButton.installEventFilter(self)

        # Button: Mark D zone
        self.markDzone.clicked.connect(self.showDzone)
        self.markDzone.installEventFilter(self)

        # Button: Mark R zone
        self.markRzone.clicked.connect(self.showRzone)
        self.markRzone.installEventFilter(self)

        # Button: Mark E zone
        self.markEzone.clicked.connect(self.markEventRegion)
        self.markEzone.installEventFilter(self)

        # Button: Calc flash edge
        self.calcFlashEdge.clicked.connect(self.calculateFlashREdge)
        self.calcFlashEdge.installEventFilter(self)

        # Edit box: min event
        self.minEventEdit.installEventFilter(self)

        # Edit box: max event
        self.maxEventEdit.installEventFilter(self)

        # Button: Locate event
        self.locateEvent.clicked.connect(self.findEvent)
        self.penumbralFitCheckBox.installEventFilter(self)

        # Button: Cancel operation
        self.cancelButton.clicked.connect(self.requestCancel)

        # Button: Calculate error bars  (... write report)
        self.calcErrBars.clicked.connect(self.computeErrorBars)
        self.calcErrBars.installEventFilter(self)

        # Button: Copy results to Asteroid Occultation Report Form (... fill Excel report)
        self.fillExcelReportButton.installEventFilter(self)
        self.fillExcelReportButton.clicked.connect(self.fillExcelReport)

        # Button: View frame
        self.viewFrameButton.clicked.connect(self.viewFrame)
        self.viewFrameButton.installEventFilter(self)
        self.frameNumSpinBox.installEventFilter(self)
        self.fieldViewCheckBox.installEventFilter(self)
        self.flipYaxisCheckBox.installEventFilter(self)
        self.flipXaxisCheckBox.installEventFilter(self)

        # Underlying lightcurve controls
        self.underlyingLightcurveLabel.installEventFilter(self)
        self.enableDiffractionCalculationBox.installEventFilter(self)
        self.demoUnderlyingLighturvesButton.installEventFilter(self)
        self.demoUnderlyingLighturvesButton.clicked.connect(self.demoClickedUnderlyingLightcurves)
        self.exposureTimeLabel.installEventFilter(self)
        self.asteroidDistanceLabel.installEventFilter(self)
        self.shadowSpeedLabel.installEventFilter(self)
        self.asteroidSizeLabel.installEventFilter(self)
        self.pathOffsetLabel.installEventFilter(self)
        self.starDiameterLabel.installEventFilter(self)
        self.dLimbAngleLabel.installEventFilter(self)
        self.rLimbAngleLabel.installEventFilter(self)

        # Button: Write error bar plot to file
        self.writeBarPlots.clicked.connect(self.exportBarPlots)
        self.writeBarPlots.installEventFilter(self)

        # Button: Write graphic to file
        self.writePlot.clicked.connect(self.exportGraphic)
        self.writePlot.installEventFilter(self)

        # Button: Write csv file
        self.writeCSVButton.clicked.connect(self.writeCSVfile)
        self.writeCSVButton.installEventFilter(self)

        # Button: Start over
        self.startOver.clicked.connect(self.restart)
        self.startOver.installEventFilter(self)

        # Set up handlers for clicks on table view of data
        self.table.cellClicked.connect(self.cellClick)
        self.table.verticalHeader().sectionClicked.connect(self.rowClick)
        self.table.installEventFilter(self)
        self.helpLabelForDataGrid.installEventFilter(self)

        # Re-instantiate mainPlot             Note: examine gui.py
        # to get this right after a re-layout !!!!  self.widget changes sometimes
        # as does horizontalLayout_?
        axesPen = pg.mkPen((0, 0, 0), width=3)

        timeAxis = TimestampAxis(orientation='bottom', pen=axesPen)
        timeAxis.setFetcher(self.getTimestampFromRdgNum)

        toptimeAxis = TimestampAxis(orientation='top', pen=axesPen)
        toptimeAxis.setFetcher(self.getTimestampFromRdgNum)

        leftAxis = pg.AxisItem(orientation='left', pen=axesPen)

        oldMainPlot = self.mainPlot
        self.mainPlot = PlotWidget(self.splitterTwo,
                                   viewBox=CustomViewBox(border=(255, 255, 255)),
                                   axisItems={'bottom': timeAxis, 'top': toptimeAxis, 'left': leftAxis},
                                   enableMenu=False, stretch=1)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.mainPlot.sizePolicy().hasHeightForWidth())
        self.mainPlot.setSizePolicy(sizePolicy)
        self.mainPlot.setObjectName("mainPlot")
        self.mainPlot.getPlotItem().showAxis('bottom', True)
        self.mainPlot.getPlotItem().showAxis('top', True)
        self.mainPlot.getPlotItem().showAxis('left', True)
        self.splitterTwo.addWidget(self.mainPlot)

        oldMainPlot.setParent(None)

        self.mainPlot.scene().sigMouseMoved.connect(self.reportMouseMoved)
        self.verticalCursor = pg.InfiniteLine(angle=90, movable=False, pen=(0, 0, 0))
        self.mainPlot.addItem(self.verticalCursor)
        self.blankCursor = True
        self.mainPlot.viewport().setProperty("cursor", QtGui.QCursor(QtCore.Qt.CursorShape.BlankCursor))
        mouseSignal.connect(self.mouseEvent)

        # Set up handler for clicks on data plot
        self.mainPlot.scene().sigMouseClicked.connect(self.processClick)
        self.mainPlotViewBox = self.mainPlot.getViewBox()
        self.mainPlotViewBox.rbScaleBox.setPen(pg.mkPen((255, 0, 0), width=2))
        self.mainPlotViewBox.rbScaleBox.setBrush(pg.mkBrush(None))
        self.mainPlot.hideButtons()
        self.mainPlot.showGrid(y=True, alpha=.5)

        self.extra = []
        self.demoLightCurve = []  # Used for detectability demonstration
        self.minDetectableDurationRdgs = None  # Used for detectability demonstration
        self.minDetectableDurationSecs = None  # Used for detectability demonstration
        self.aperture_names = []
        self.initializeTableView()  # Mostly just establishes column headers

        # Open (or create) file for holding 'sticky' stuff
        self.settings = QSettings('pyote.ini', QSettings.IniFormat)
        self.settings.setFallbacksEnabled(False)

        lineWidth = self.settings.value('lineWidth', '5')
        dotSize = self.settings.value('dotSize', '8')

        self.lineWidthSpinner.setValue(int(lineWidth))
        self.dotSizeSpinner.setValue(int(dotSize))

        allowNewVersionPopup = self.settings.value('allowNewVersionPopup', 'true')
        if allowNewVersionPopup == 'true':
            self.allowNewVersionPopupCheckbox.setChecked(True)
        else:
            self.allowNewVersionPopupCheckbox.setChecked(False)

        self.allowNewVersionPopupCheckbox.installEventFilter(self)

        tabNameList = self.settings.value('tablist', [])
        if tabNameList:
            self.redoTabOrder(tabNameList)
            # self.switchToTabNamed(tabNameList[0])

        # This is a 'hack' to override QtDesigner which has evolved somehow the abilty to block my attempts
        # at setting reasonable size parameters in the drag-and drop Designer.
        self.resize(QSize(0, 0))
        self.setMinimumSize(QSize(0, 0))

        self.logFile = None
        self.detectabilityLogFile = None
        self.normalizationLogFile = None

        # Use 'sticky' settings to size and position the main screen
        self.resize(self.settings.value('size', QSize(800, 800)))
        self.move(self.settings.value('pos', QPoint(50, 50)))
        usediff = self.settings.value('usediff', 'true') == 'true'
        self.enableDiffractionCalculationBox.setChecked(usediff)
        doOCRcheck = self.settings.value('doOCRcheck', 'true') == 'true'
        self.showOCRcheckFramesCheckBox.setChecked(doOCRcheck)
        showCameraResponse = self.settings.value('showCameraResponse', 'false') == 'true'
        self.showCameraResponseCheckBox.setChecked(showCameraResponse)
        showTimestamps = self.settings.value('showTimestamps', 'true') == 'true'
        self.showTimestampsCheckBox.setChecked(showTimestamps)

        self.ne3NotInUseRadioButton.setChecked(self.settings.value('ne3NotInUse', 'false') == 'true')
        self.dnrOffRadioButton.setChecked(self.settings.value('dnrOff', 'false') == 'true')
        self.dnrLowRadioButton.setChecked(self.settings.value('dnrLow', 'false') == 'true')
        self.dnrMiddleRadioButton.setChecked(self.settings.value('dnrMiddle', 'false') == 'true')
        self.dnrHighRadioButton.setChecked(self.settings.value('dnrHigh', 'false') == 'true')

        self.dnrLowDspinBox.setValue(float(self.settings.value('dnrLowDtc', 0.50)))
        self.dnrLowRspinBox.setValue(float(self.settings.value('dnrLowRtc', 0.50)))

        self.dnrMiddleDspinBox.setValue(float(self.settings.value('dnrMiddleDtc', 1.00)))
        self.dnrMiddleRspinBox.setValue(float(self.settings.value('dnrMiddleRtc', 1.00)))

        self.dnrHighDspinBox.setValue(float(self.settings.value('dnrHighDtc', 4.50)))
        self.dnrHighRspinBox.setValue(float(self.settings.value('dnrHighRtc', 2.00)))

        self.dnrHighDspinBox.valueChanged.connect(self.processTimeConstantChange)
        self.dnrHighRspinBox.valueChanged.connect(self.processTimeConstantChange)
        self.dnrMiddleDspinBox.valueChanged.connect(self.processTimeConstantChange)
        self.dnrMiddleRspinBox.valueChanged.connect(self.processTimeConstantChange)
        self.dnrLowDspinBox.valueChanged.connect(self.processTimeConstantChange)
        self.dnrLowRspinBox.valueChanged.connect(self.processTimeConstantChange)

        # splitterOne is the vertical splitter in the lower panel.
        # splitterTwo is the vertical splitter in the upper panel
        # splitterThree is the horizontal splitter between the top and bottom panel

        if self.settings.value('splitterOne') is not None:
            self.splitterOne.restoreState(self.settings.value('splitterOne'))
            self.splitterTwo.restoreState(self.settings.value('splitterTwo'))
            self.splitterThree.restoreState(self.settings.value('splitterThree'))

        self.pymovieFileInUse = False

        self.pymovieDataColumnPrefixComboBox.addItem("signal")
        self.pymovieDataColumnPrefixComboBox.addItem("appsum")
        self.pymovieDataColumnPrefixComboBox.addItem("avgbkg")
        self.pymovieDataColumnPrefixComboBox.addItem("stdbkg")
        self.pymovieDataColumnPrefixComboBox.addItem("nmaskpx")

        self.outliers = []
        self.timeDelta = None

        self.selPts = []
        self.initializeVariablesThatDontDependOnAfile()

        self.pathToVideo = None
        self.cascadePosition = None
        self.cascadeDelta = 25
        self.frameViews = []

        self.fieldMode = False

        self.d_underlying_lightcurve = None
        self.r_underlying_lightcurve = None

        self.d_candidates = None
        self.d_candidate_entry_nums = None
        self.r_candidates = None
        self.r_candidate_entry_nums = None
        self.b_intensity = None
        self.a_intensity = None
        self.penumbral_noise = None
        self.penumbralDerrBar = None
        self.penumbralRerrBar = None
        self.lastDmetric = 0.0
        self.lastRmetric = 0.0

        self.xlsxDict = {}

        self.checkForNewVersion()

        # We are removing this procedure in version 4.8.0 and above
        # so that can use pipenv to install PyOTE without the need
        # for an Anaconda3 install.
        # self.copy_desktop_icon_file_to_home_directory()

        self.helperThing = HelpDialog()

        if self.externalCsvFilePath is not None:
            if os.path.exists(self.externalCsvFilePath):
                self.showMsg(f'We will read: {self.externalCsvFilePath}')
                self.readDataFromFile()
            else:
                self.showMsg(f'Could not find csv file specified: {self.externalCsvFilePath}')
                self.externalCsvFilePath = None

        # self.firstEvent = True
        if self.allowNewVersionPopupCheckbox.isChecked():
            self.showHelp(self.allowNewVersionPopupCheckbox)

    def processYoffsetStepBy10(self):
        self.yOffsetStep10radioButton.repaint()
        for spinBox in self.yOffsetSpinBoxes:
            spinBox.setSingleStep(10)

    def processYoffsetStepBy100(self):
        self.yOffsetStep100radioButton.repaint()
        for spinBox in self.yOffsetSpinBoxes:
            spinBox.setSingleStep(100)

    def processYoffsetStepBy1000(self):
        self.yOffsetStep1000radioButton.repaint()
        for spinBox in self.yOffsetSpinBoxes:
            spinBox.setSingleStep(1000)

    def processStepBy2(self):
        self.smoothingIntervalSpinBox.setSingleStep(2)

    def processStepBy10(self):
        self.smoothingIntervalSpinBox.setSingleStep(10)

    def processStepBy100(self):
        self.smoothingIntervalSpinBox.setSingleStep(100)

    def initializeLightcurvePanel(self):
        for checkBox in self.targetCheckBoxes:
            checkBox.setChecked(False)
            checkBox.setEnabled(False)
        self.targetCheckBoxes[0].setChecked(True)

        for checkBox in self.showCheckBoxes:
            checkBox.setChecked(False)
            checkBox.setEnabled(False)
        self.showCheckBoxes[0].setChecked(True)

        for checkBox in self.referenceCheckBoxes:
            checkBox.setChecked(False)
            checkBox.setEnabled(False)

        for title in self.lightcurveTitles:
            title.setText('')

        for spinBox in self.yOffsetSpinBoxes:
            spinBox.setValue(0)
            spinBox.setEnabled(False)
        self.yOffsetSpinBoxes[0].setValue(0)
        self.yOffsetSpinBoxes[0].setEnabled(False)

        for spinBox in self.xOffsetSpinBoxes:
            spinBox.setValue(0)
            spinBox.setEnabled(False)

        self.recolorBlobs()

    def processYoffsetChange(self):
        # QtWidgets.QApplication.setOverrideCursor(QtGui.QCursor(QtCore.Qt.BusyCursor))
        # QtWidgets.QApplication.processEvents()
        self.newRedrawMainPlot()
        # QtWidgets.QApplication.setOverrideCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
        # QtWidgets.QApplication.processEvents()

    def processXoffsetChange(self):
        self.newRedrawMainPlot()

    def clearReferenceSelections(self):
        for checkBox in self.referenceCheckBoxes:
            checkBox.setChecked(False)
        for spinBox in self.xOffsetSpinBoxes:
            spinBox.setEnabled(False)

    def recolorBlobs(self):
        for i, colorBlob in enumerate(self.colorBlobs):
            colorBlob.setStyleSheet(self.colorStyle[i])
        for i, checkBox in enumerate(self.targetCheckBoxes):
            if checkBox.isChecked():
                self.colorBlobs[i].setStyleSheet('color: rgb(0,0,255)')  # Set target curve bright blue
                break
        for i, checkBox in enumerate(self.referenceCheckBoxes):
            if checkBox.isChecked():
                self.colorBlobs[i].setStyleSheet('color: rgb(0,255,0)')  # set reference curve bright green
                break

    def processReferenceSelection(self, i):
        # Undo any previous normalization
        self.fillTableViewOfData()
        self.smoothingIntervalSpinBox.setValue(0)

        if self.referenceCheckBoxes[i].isChecked():
            self.showMsg(f'{self.lightcurveTitles[i].text()} is selected as the reference curve for normalization.')
            self.clearReferenceSelections()
            self.xOffsetSpinBoxes[i].setEnabled(True)
            self.referenceCheckBoxes[i].setChecked(True)
            self.showCheckBoxes[i].setChecked(True)
            if i == 0:
                self.yRefStar = self.LC1[:]
            elif i == 1:
                self.yRefStar = self.LC2[:]
            elif i == 2:
                self.yRefStar = self.LC3[:]
            elif i == 3:
                self.yRefStar = self.LC4[:]
            else:
                self.yRefStar = self.extra[i - 4][:]
            self.recolorBlobs()
            self.newRedrawMainPlot()
        else:
            self.xOffsetSpinBoxes[i].setEnabled(False)
            for i, checkBox in enumerate(self.referenceCheckBoxes):
                if checkBox.isChecked():
                    self.showMsg(
                        f'{self.lightcurveTitles[i].text()} is selected as the reference curve for normalization.')
                    return
            self.yRefStar = []
            self.newRedrawMainPlot()
            self.showMsg(f'The reference curve has been deselected so normalization is disabled.')

    def processReferenceSelection1(self):
        self.processReferenceSelection(0)

    def processReferenceSelection2(self):
        self.processReferenceSelection(1)

    def processReferenceSelection3(self):
        self.processReferenceSelection(2)

    def processReferenceSelection4(self):
        self.processReferenceSelection(3)

    def processReferenceSelection5(self):
        self.processReferenceSelection(4)

    def processReferenceSelection6(self):
        self.processReferenceSelection(5)

    def processReferenceSelection7(self):
        self.processReferenceSelection(6)

    def processReferenceSelection8(self):
        self.processReferenceSelection(7)

    def processReferenceSelection9(self):
        self.processReferenceSelection(8)

    def processReferenceSelection10(self):
        self.processReferenceSelection(9)

    def forceTargetToShow(self):
        for i, checkBox in enumerate(self.targetCheckBoxes):
            if checkBox.isChecked():
                self.showCheckBoxes[i].setChecked(True)

    def processShowSelection(self):
        self.forceTargetToShow()
        self.newRedrawMainPlot()

    def clearTargetSelections(self):
        for checkBox in self.targetCheckBoxes:
            checkBox.setChecked(False)
        for yOffsetSpin in self.yOffsetSpinBoxes:
            yOffsetSpin.setEnabled(True)

    def noTargetSelected(self):
        for checkBox in self.targetCheckBoxes:
            if checkBox.isChecked():
                return False
        return True

    def processTargetSelection(self, i, redraw):
        if self.targetCheckBoxes[i].isChecked():
            self.clearTargetSelections()
            self.targetCheckBoxes[i].setChecked(True)
            self.showCheckBoxes[i].setChecked(True)
            self.yOffsetSpinBoxes[i].setEnabled(False)
            self.yOffsetSpinBoxes[i].setValue(0)
            if i == 0:
                self.yValues = self.LC1.copy()
            elif i == 1:
                self.yValues = self.LC2.copy()
            elif i == 2:
                self.yValues = self.LC3.copy()
            elif i == 3:
                self.yValues = self.LC4.copy()
            else:
                self.yValues = self.extra[i - 4].copy()
            if redraw:
                self.newRedrawMainPlot()
            self.recolorBlobs()
            self.yOffsetSpinBoxes[i].setEnabled(False)
            self.yOffsetSpinBoxes[i].setValue(0)
        else:
            if self.noTargetSelected():
                self.targetCheckBoxes[i].setChecked(True)
                self.showCheckBoxes[i].setChecked(True)
                self.yOffsetSpinBoxes[i].setEnabled(False)
                self.yOffsetSpinBoxes[i].setValue(0)

    def processTargetSelection1(self):
        self.processTargetSelection(0, redraw=True)

    def processTargetSelection2(self):
        self.processTargetSelection(1, redraw=True)

    def processTargetSelection3(self):
        self.processTargetSelection(2, redraw=True)

    def processTargetSelection4(self):
        self.processTargetSelection(3, redraw=True)

    def processTargetSelection5(self):
        self.processTargetSelection(4, redraw=True)

    def processTargetSelection6(self):
        self.processTargetSelection(5, redraw=True)

    def processTargetSelection7(self):
        self.processTargetSelection(6, redraw=True)

    def processTargetSelection8(self):
        self.processTargetSelection(7, redraw=True)

    def processTargetSelection9(self):
        self.processTargetSelection(8, redraw=True)

    def processTargetSelection10(self):
        self.processTargetSelection(9, redraw=True)

    def processTimeConstantChange(self):
        # We don't want to respond to Ne3 time constant changes until
        # the initial solution has been found, hence the following test
        # if self.solution is not None:
        #     self.showMsg('Called findEvent()')
        #     self.findEvent()
        pass

    def handlePymovieColumnChange(self):
        # self.showInfo('Got a changed column!')
        if self.pymovieFileInUse:
            self.externalCsvFilePath = self.filename
            self.readDataFromFile()

    def redoTabOrder(self, tabnames):

        def getIndexOfTabFromName(name):
            for i_local in range(self.tabWidget.count()):
                if self.tabWidget.tabText(i_local) == name:
                    return i_local
            return -1

        numTabs = self.tabWidget.count()
        if not len(tabnames) == numTabs:
            # self.showMsg(f'Mismatch in saved tab list versus current number of tabs.')
            return

        for i in range(len(tabnames)):
            from_index = getIndexOfTabFromName(tabnames[i])
            to_index = i
            if from_index < 0:
                # self.showMsg(f'Could not locate {tabnames[i]} in the existing tabs')
                return
            else:
                self.tabWidget.tabBar().moveTab(from_index, to_index)

    def clearBaselineRegions(self):
        self.bkgndRegionLimits = []
        self.clearBaselineRegionsButton.setEnabled(False)
        self.calcStatsFromBaselineRegionsButton.setEnabled(False)
        self.showMsg('Background regions cleared.')
        x = [i for i in range(self.dataLen) if self.yStatus[i] == BASELINE]
        for i in x:
            self.yStatus[i] = INCLUDED
        self.newRedrawMainPlot()

    def markEventRegion(self):
        if len(self.selectedPoints) == 0:
            x = [i for i in range(self.dataLen) if self.yStatus[i] == EVENT]
            for i in x:
                self.yStatus[i] = INCLUDED
            self.newRedrawMainPlot()
            # self.showInfo('No points were selected. Exactly two are required')
            return

        if len(self.selectedPoints) != 2:
            self.showInfo('Exactly two points must be selected for this operation.')
            return

        # Erase any previously selected event points
        x = [i for i in range(self.dataLen) if self.yStatus[i] == EVENT]
        for i in x:
            self.yStatus[i] = INCLUDED
        self.newRedrawMainPlot()

        selIndices = [key for key, _ in self.selectedPoints.items()]
        selIndices.sort()

        leftEdge = int(min(selIndices))
        rightEdge = int(max(selIndices))
        self.showMsg('Event region selected: ' + str([leftEdge, rightEdge]))

        # The following loop marks event points and removes the 2 red points
        # by overwriting then with the EVENT color
        self.selectedPoints = {}
        for i in range(leftEdge, rightEdge + 1):
            self.yStatus[i] = EVENT

        self.calcEventStatisticsFromMarkedRegion()

        self.newRedrawMainPlot()

    def markBaselineRegion(self):
        # If the user has not selected any points, we ignore the request
        if len(self.selectedPoints) == 0:
            self.showInfo('No points were selected. Exactly two are required')
            return

        if len(self.selectedPoints) != 2:
            self.showInfo('Exactly two points must be selected for this operation.')
            return

        self.clearBaselineRegionsButton.setEnabled(True)
        self.calcStatsFromBaselineRegionsButton.setEnabled(True)

        selIndices = [key for key, _ in self.selectedPoints.items()]
        selIndices.sort()

        leftEdge = int(min(selIndices))
        rightEdge = int(max(selIndices))

        self.bkgndRegionLimits.append([leftEdge, rightEdge])

        # The following loop removes the 2 red points because they get overwritten by the BASELINE color
        self.selectedPoints = {}
        for i in range(leftEdge, rightEdge + 1):
            self.yStatus[i] = BASELINE

        self.showMsg('Background region selected: ' + str([leftEdge, rightEdge]))
        self.newRedrawMainPlot()

    def calcEventStatisticsFromMarkedRegion(self):
        y = [self.yValues[i] for i in range(self.dataLen) if self.yStatus[i] == EVENT]
        mean = np.mean(y)
        self.A = mean
        self.sigmaA = float(np.std(y))

        self.userDeterminedEventStats = True

        self.showMsg(f'mean event = {mean:0.2f}')
        self.showMsg(f'sigmaA: = {self.sigmaA:0.2f}')

    def calcBaselineStatisticsFromMarkedRegions(self):
        xIndices = [i for i in range(self.dataLen) if self.yStatus[i] == BASELINE]
        y = [self.yValues[i] for i in range(self.dataLen) if self.yStatus[i] == BASELINE]
        mean = np.mean(y)
        self.B = mean
        baselineXvals = []
        baselineYvals = []
        for i in xIndices:
            baselineXvals.append(i)
            baselineYvals.append(self.yValues[i])

        # Note: the getCorCoefs() routine uses a savgol linear filter of window 301 to remove trends during
        # the calculation of sigB
        self.newCorCoefs, self.numNApts, sigB = getCorCoefs(baselineXvals,
                                                            baselineYvals)
        self.showMsg('Baseline noise analysis done using ' + str(self.numNApts) +
                     ' baseline points')
        self.corCoefs = np.ndarray(shape=(len(self.newCorCoefs),))
        np.copyto(self.corCoefs, self.newCorCoefs)
        self.numPtsInCorCoefs = self.numNApts
        self.sigmaB = sigB

        self.userDeterminedBaselineStats = True

        self.prettyPrintCorCoefs()

        self.showMsg(f'mean baseline = {mean:0.2f}')
        self.showMsg(f'baseline snr = {mean / sigB:0.2f}')

    def getTimestampFromRdgNum(self, rdgNum):
        readingNumber = int(rdgNum)
        if readingNumber >= 0:
            if readingNumber < len(self.yTimes):
                tString = self.yTimes[readingNumber]
                if tString == '[]' or tString == '' or not self.showTimestampsCheckBox.isChecked():
                    return f'{readingNumber}'
                else:
                    return self.yTimes[readingNumber]
        return '???'

    def fillExcelReport(self):
        # Open a file select dialog
        xlsxfilepath, _ = QFileDialog.getOpenFileName(
            self,  # parent
            "Select Asteroid Occultation Report form",  # title for dialog
            self.settings.value('lightcurvedir', ""),  # starting directory
            "Excel files (*.xlsx)")

        if xlsxfilepath:
            # noinspection PyBroadException
            wb = load_workbook(xlsxfilepath)

            try:
                sheet = wb['DATA']

                # Validate that a proper Asteroid Occultation Report Form was selected by reading the report header
                if not sheet['G1'].value == 'Asteroid Occultation Report Form':
                    self.showMsg(f'The xlsx file selected does not appear to be an Asteroid Occultation Report Form')
                    return

                # We're going to ignore the named cell info and reference all the cells of interest in
                # their col/row coordinates (not all the cells of interest had names)

                Derr68 = 'L33'
                Derr95 = 'M33'
                Derr99 = 'N33'

                Rerr68 = 'L35'
                Rerr95 = 'M35'
                Rerr99 = 'N35'

                Dhour = 'F32'
                Dmin = 'H32'
                Dsec = 'J32'

                Rhour = 'F36'
                Rmin = 'H36'
                Rsec = 'J36'

                # Exposure = 'P25'
                OTA = 'O23'
                SNR = 'W40'
                Comment = 'D43'

                sheet[OTA].value = 'PYOTE'
                sheet[SNR].value = f'{self.snrB:0.2f}'

                if 'Comment' in self.xlsxDict:
                    sheet[Comment].value = self.xlsxDict['Comment']

                if 'Derr68' in self.xlsxDict:
                    # sheet[Derr68].value = f'{self.xlsxDict["Derr68"]:0.2f}'
                    sheet[Derr68].value = self.xlsxDict["Derr68"]

                if 'Derr95' in self.xlsxDict:
                    # sheet[Derr95].value = f'{self.xlsxDict["Derr95"]:0.2f}'
                    sheet[Derr95].value = self.xlsxDict["Derr95"]

                if 'Derr99' in self.xlsxDict:
                    # sheet[Derr99].value = f'{self.xlsxDict["Derr99"]:0.2f}'
                    sheet[Derr99].value = self.xlsxDict["Derr99"]

                if 'Rerr68' in self.xlsxDict:
                    # sheet[Rerr68].value = f'{self.xlsxDict["Rerr68"]:0.2f}'
                    sheet[Rerr68].value = self.xlsxDict["Rerr68"]

                if 'Rerr95' in self.xlsxDict:
                    # sheet[Rerr95].value = f'{self.xlsxDict["Rerr95"]:0.2f}'
                    sheet[Rerr95].value = self.xlsxDict["Rerr95"]

                if 'Rerr99' in self.xlsxDict:
                    # sheet[Rerr99].value = f'{self.xlsxDict["Rerr99"]:0.2f}'
                    sheet[Rerr99].value = self.xlsxDict["Rerr99"]

                if 'Dhour' in self.xlsxDict:
                    sheet[Dhour] = int(self.xlsxDict['Dhour'])
                if 'Dmin' in self.xlsxDict:
                    sheet[Dmin] = int(self.xlsxDict['Dmin'])
                if 'Dsec' in self.xlsxDict:
                    sheet[Dsec] = float(self.xlsxDict['Dsec'])

                if 'Rhour' in self.xlsxDict:
                    sheet[Rhour] = int(self.xlsxDict['Rhour'])
                if 'Rmin' in self.xlsxDict:
                    sheet[Rmin] = int(self.xlsxDict['Rmin'])
                if 'Rsec' in self.xlsxDict:
                    sheet[Rsec] = float(self.xlsxDict['Rsec'])

                # Overwriting the original file !!!
                wb.save(xlsxfilepath)

            except Exception as e:
                self.showMsg(repr(e))
                self.showMsg(f'FAILED to fill Asteroid Occultation Report Form', color='red', bold=True)
                self.showMsg(f'Is it possible that you have the file already open somewhere?', color='red', bold=True)
                return

            self.showMsg(f'Excel spreadsheet Asteroid Report Form entries made successfully.')

            # noinspection PyBroadException
            try:
                if platform.system() == 'Darwin':
                    subprocess.call(['open', xlsxfilepath])
                elif platform.system() == 'Windows':
                    os.startfile(xlsxfilepath)
                else:
                    subprocess.call(['xdg-open', xlsxfilepath])
            except Exception as e:
                self.showMsg('Attempt to get host OS to open xlsx file failed.', color='red', bold=True)
                self.showMsg(repr(e))

            # OS = sys.platform
            # if OS == 'darwin' or OS == 'linux':
            #     subprocess.check_call(['open', xlsxfilepath])
            # else:
            #     subprocess.check_call(['start', xlsxfilepath])

            # Fill with our current values
        else:
            return

    def validateLightcurveDataInput(self):
        ans = {'success': True}

        # Process exp dur entry
        try:
            exp_dur_str = self.expDurEdit.text().strip()
            if not exp_dur_str:
                ans.update({'exp_dur': None})
            else:
                exp_dur = float(exp_dur_str)
                if exp_dur > 0.0:
                    ans.update({'exp_dur': exp_dur})
                else:
                    self.showMsg(f'exposure duration must be > 0.0', bold=True)
                    ans.update({'exp_dur': None})
                    ans.update({'success': False})
        except ValueError as e:
            self.showMsg(f'{e}', bold=True)
            ans.update({'exp_dur': None})
            ans.update({'success': False})

        # Process ast_dist entry
        try:
            ast_dist_str = self.asteroidDistanceEdit.text().strip()
            if not ast_dist_str:
                ans.update({'ast_dist': None})
            else:
                ast_dist = float(ast_dist_str)
                if ast_dist > 0.0:
                    ans.update({'ast_dist': ast_dist})
                else:
                    self.showMsg(f'ast_dist must be > 0.0', bold=True)
                    ans.update({'ast_dist': None})
                    ans.update({'success': False})
        except ValueError as e:
            self.showMsg(f'{e}', bold=True)
            ans.update({'ast_dist': None})
            ans.update({'success': False})

        # Process shadow_speed entry
        try:
            shadow_speed_str = self.shadowSpeedEdit.text().strip()
            if not shadow_speed_str:
                ans.update({'shadow_speed': None})
            else:
                shadow_speed = float(shadow_speed_str)
                if shadow_speed > 0.0:
                    ans.update({'shadow_speed': shadow_speed})
                else:
                    self.showMsg(f'shadow speed must be > 0.0', bold=True)
                    ans.update({'shadow_speed': None})
                    ans.update({'success': False})
        except ValueError as e:
            self.showMsg(f'{e}', bold=True)
            ans.update({'shadow_speed': None})
            ans.update({'success': False})

        # Process asteroid diameter
        try:
            ast_diam_str = self.astSizeEdit.text().strip()
            if not ast_diam_str:
                ans.update({'ast_diam': None})
            else:
                ast_diam = float(ast_diam_str)
                if ast_diam > 0.0:
                    ans.update({'ast_diam': ast_diam})
                else:
                    self.showMsg(f'asteroid diameter must be > 0.0 or missing', bold=True)
                    ans.update({'ast_diam': None})
                    ans.update({'success': False})
        except ValueError as e:
            self.showMsg(f'{e}', bold=True)
            ans.update({'ast_diam': None})
            ans.update({'success': False})

        # Process centerline offset
        try:
            centerline_offset_str = self.pathOffsetEdit.text().strip()
            if not centerline_offset_str:
                ans.update({'centerline_offset': None})
            else:
                if ans['ast_diam'] is None:
                    ans.update({'centerline_offset': None})
                    ans.update({'success': False})
                    self.showMsg(f'centerline offset requires an asteroid diameter to be specified', bold=True)
                else:
                    centerline_offset = float(centerline_offset_str)
                    if 0.0 <= centerline_offset < ans['ast_diam'] / 2:
                        ans.update({'centerline_offset': centerline_offset})
                    else:
                        self.showMsg(f'centerline offset must be positive and less than the asteroid radius', bold=True)
                        ans.update({'centerline_offset': None})
                        ans.update({'success': False})
        except ValueError as e:
            self.showMsg(f'{e}', bold=True)
            ans.update({'centerline_offset': None})
            ans.update({'success': False})

        # Process star diam entry
        try:
            star_diam_str = self.starDiameterEdit.text().strip()
            if not star_diam_str:
                ans.update({'star_diam': None})
                self.penumbralFitCheckBox.setChecked(False)
                self.penumbralFitCheckBox.setEnabled(False)
            else:
                star_diam = float(star_diam_str)
                if star_diam > 0.0:
                    ans.update({'star_diam': star_diam})
                    self.penumbralFitCheckBox.setEnabled(True)
                else:
                    self.showMsg(f'star diameter must be > 0.0 or missing', bold=True)
                    ans.update({'star_diam': None})
                    ans.update({'success': False})
                    self.penumbralFitCheckBox.setChecked(False)
                    self.penumbralFitCheckBox.setEnabled(False)
        except ValueError as e:
            self.showMsg(f'{e}', bold=True)
            ans.update({'star_diam': None})
            ans.update({'success': False})
            self.penumbralFitCheckBox.setChecked(False)
            self.penumbralFitCheckBox.setEnabled(False)

        # Process D limb angle entry
        d_angle = self.dLimbAngle.value()
        ans.update({'d_angle': d_angle})

        # Process R limb angle entry
        r_angle = self.rLimbAngle.value()
        ans.update({'r_angle': r_angle})

        return ans

    # This method is needed because you cannot pass parameters from a clicked-connect
    def demoClickedUnderlyingLightcurves(self):
        if self.B is None or self.A is None:
            self.demoUnderlyingLightcurves(baseline=100.0, event=0.0, plots_wanted=True, ignore_timedelta=True)
        else:
            self.demoUnderlyingLightcurves(baseline=self.B, event=self.A, plots_wanted=True, ignore_timedelta=True)

    def demoUnderlyingLightcurves(self, baseline=None, event=None, plots_wanted=False, ignore_timedelta=False):

        diff_table_name = f'diffraction-table.p'
        diff_table_path = os.path.join(self.homeDir, diff_table_name)

        ans = self.validateLightcurveDataInput()

        if not ans['success']:
            self.showInfo('There is a problem with the data entry.\n\nCheck log for details.')
            return

        if not ignore_timedelta:
            if self.timeDelta is None or self.timeDelta < 0.001:
                if ans['exp_dur'] is not None:
                    frame_time = ans['exp_dur']
                else:
                    frame_time = 0.001
            else:
                frame_time = self.timeDelta
        else:
            if ans['exp_dur'] is not None:
                frame_time = ans['exp_dur']
            else:
                frame_time = 0.001

        if ans['exp_dur'] is not None and ans['ast_dist'] is None and ans['shadow_speed'] is None:
            pass  # User wants to ignore diffraction effects
        else:
            if self.enableDiffractionCalculationBox.isChecked() and \
                    (ans['ast_dist'] is None or ans['shadow_speed'] is None):
                self.showMsg(f'Cannot compute diffraction curve without both ast distance and shadow speed!', bold=True)
                return None

        if ans['ast_dist'] is not None:
            fresnel_length_at_500nm = fresnel_length_km(distance_AU=ans['ast_dist'], wavelength_nm=500.0)
            if plots_wanted:
                self.showMsg(f'Fresnel length @ 500nm: {fresnel_length_at_500nm:.4f} km', bold=True, color='green')

        if ans['star_diam'] is not None and (ans['d_angle'] is None or ans['r_angle'] is None):
            ans.update({'star_diam': None})
            self.showMsg(f'An incomplete set of star parameters was entered --- treating star_diam as None!', bold=True)
        elif ans['star_diam'] is not None and (ans['ast_dist'] is None or ans['shadow_speed'] is None):
            ans.update({'star_diam': None})
            self.showMsg(f'Need dist and shadow speed to utilize star diam --- treating star_diam as None!', bold=True)

        # noinspection PyBroadException
        try:
            matplotlib.pyplot.close(self.d_underlying_lightcurve)
            matplotlib.pyplot.close(self.r_underlying_lightcurve)
        except Exception:
            pass

        self.d_underlying_lightcurve, self.r_underlying_lightcurve, ans = generate_underlying_lightcurve_plots(
            diff_table_path=diff_table_path,
            b_value=baseline,
            a_value=event,
            frame_time=frame_time,
            ast_dist=ans['ast_dist'],
            shadow_speed=ans['shadow_speed'],
            ast_diam=ans['ast_diam'],
            centerline_offset=ans['centerline_offset'],
            star_diam=ans['star_diam'],
            d_angle=ans['d_angle'],
            r_angle=ans['r_angle'],
            suppress_diffraction=not self.enableDiffractionCalculationBox.isChecked(),
            title_addon=''
        )
        if plots_wanted:
            self.d_underlying_lightcurve.show()
            self.r_underlying_lightcurve.show()
        else:
            matplotlib.pyplot.close(self.d_underlying_lightcurve)
            matplotlib.pyplot.close(self.r_underlying_lightcurve)

        return ans

    def findTimestampFromFrameNumber(self, frame):
        # Currently PyMovie uses nn.00 for frame number
        # Limovie uses nn.0 for frame number
        # We use the 'starts with' flag so that we pick up both forms
        items = self.table.findItems(f'{frame:0.1f}', QtCore.Qt.MatchFlag.MatchStartsWith)
        for item in items:
            if item.column() == 0:  # Avoid a possible match from a data column
                ts = self.table.item(item.row(), 1).text()
                return ts
        return ''

    def showAnnotatedFrame(self, frame_to_show, annotation):

        frame_number = frame_to_show

        table_timestamp = self.findTimestampFromFrameNumber(frame_to_show)

        if not table_timestamp:
            table_timestamp = 'no timestamp found'

        if self.pathToVideo is None:
            return

        _, ext = os.path.splitext(self.pathToVideo)

        if ext == '.avi':
            ans = readAviFile(frame_number, full_file_path=self.pathToVideo)
            if not ans['success']:
                self.showMsg(f'Attempt to view frame returned errmsg: {ans["errmsg"]}')
                return
        elif ext == '.ser':
            ans = readSerFile(frame_number, full_file_path=self.pathToVideo)
            if not ans['success']:
                self.showMsg(f'Attempt to view frame returned errmsg: {ans["errmsg"]}')
                return
        elif ext == '':
            # We assume it's a FITS folder that we have been given
            ans = readFitsFile(frame_number, full_file_path=self.pathToVideo)
            if not ans['success']:
                self.showMsg(f'Attempt to view frame returned errmsg: {ans["errmsg"]}')
                return
        elif ext == '.aav':
            ans = readAavFile(frame_number, full_file_path=self.pathToVideo)
            if not ans['success']:
                self.showMsg(f'Attempt to view frame returned errmsg: {ans["errmsg"]}')
                return
        else:
            self.showMsg(f'Unsupported file extension: {ext}')
            return

        # Check to see if user has closed all frame views
        frame_visible = False
        for frame_view in self.frameViews:
            if frame_view and frame_view.isVisible():
                frame_visible = True

        if not frame_visible:
            self.cascadePosition = 100

        title = f'{annotation} {table_timestamp} @ frame {frame_number}'
        self.frameViews.append(pg.GraphicsWindow(title=title))

        cascade_origin = self.pos() + QPoint(self.cascadePosition, self.cascadePosition)

        self.frameViews[-1].move(cascade_origin)
        self.cascadePosition += self.cascadeDelta

        self.frameViews[-1].resize(1000, 600)
        layout = QtWidgets.QGridLayout()
        self.frameViews[-1].setLayout(layout)
        imv = pg.ImageView()
        layout.addWidget(imv, 0, 0)

        imv.ui.menuBtn.hide()
        imv.ui.roiBtn.hide()

        image = ans['image']

        if self.fieldViewCheckBox.isChecked():
            upper_field = image[0::2, :]
            lower_field = image[1::2, :]
            image = np.concatenate((upper_field, lower_field))

        if self.flipYaxisCheckBox.isChecked():
            image = np.flipud(image)

        if self.flipXaxisCheckBox.isChecked():
            image = np.fliplr(image)

        imv.setImage(image)

        for i, frame_view in enumerate(self.frameViews):
            if frame_view and not frame_view.isVisible():
                # User has closed the image.  Remove it so that garbage collection occurs.
                self.frameViews[i].close()
                self.frameViews[i] = None
            else:
                if frame_view:
                    frame_view.raise_()

    def viewFrame(self):
        if self.pathToVideo is None:
            return

        frame_to_show = self.frameNumSpinBox.value()
        self.showAnnotatedFrame(frame_to_show=frame_to_show, annotation='User selected frame:')

    def helpButtonClicked(self):
        self.showHelp(self.helpButton)

    def ne3ExplanationClicked(self):
        self.showHelp(self.ne3ExplanationButton)

    def tutorialButtonClicked(self):
        self.showHelp(self.tutorialButton)

    def lightcurvesHelpButtonClicked(self):
        self.showHelp(self.lightcurvesHelpButton)

    def plotHelpButtonClicked(self):
        self.showHelp(self.plotHelpButton)

    def detectabilityHelpButtonClicked(self):
        self.showHelp(self.detectabilityHelpButton)

    @staticmethod
    def htmlFixup(html):
        output = ''
        endIndex = len(html) - 1
        for i in range(len(html)):
            if not (html[i] == '.' or html[i] == ','):
                output += html[i]
            else:
                if i == endIndex:
                    output += html[i]
                    return output
                if html[i + 1] == ' ':
                    output += html[i] + '&nbsp;'
                else:
                    output += html[i]
        return output

    def showHelp(self, obj):

        if obj.toolTip():
            self.helperThing.raise_()
            self.helperThing.show()
            self.helperThing.textEdit.clear()
            stuffToShow = self.htmlFixup(obj.toolTip())
            self.helperThing.textEdit.insertHtml(stuffToShow)
            self.helperThing.setHidden(True)
            self.helperThing.setVisible(True)

    @staticmethod
    def processKeystroke(event):
        _ = event.key()  # Just to satisfy PEP8
        return False

    def eventFilter(self, obj, event):
        # if self.firstEvent and self.targetCheckBox_1.isVisible():
        if self.allowNewVersionPopupCheckbox.isChecked():
            self.helperThing.raise_()
        #     self.firstEvent = False
        if event.type() == QtCore.QEvent.Type.KeyPress:
            handled = self.processKeystroke(event)
            if handled:
                return True
            else:
                return super(SimplePlot, self).eventFilter(obj, event)

        if event.type() == QtCore.QEvent.Type.MouseButtonPress:
            if event.button() == QtCore.Qt.MouseButton.RightButton:
                if obj.toolTip():
                    self.helperThing.raise_()
                    self.helperThing.show()
                    self.helperThing.textEdit.clear()
                    stuffToShow = self.htmlFixup(obj.toolTip())
                    self.helperThing.textEdit.insertHtml(stuffToShow)
                    return True
            return super(SimplePlot, self).eventFilter(obj, event)
            # return False

        if event.type() == QtCore.QEvent.Type.ToolTip:
            return True

        return super(SimplePlot, self).eventFilter(obj, event)

    def writeCSVfile(self):
        _, name = os.path.split(self.filename)
        name = self.removeCsvExtension(name)

        name += '.PYOTE.csv'

        myOptions = QFileDialog.Options()
        # myOptions |= QFileDialog.DontConfirmOverwrite
        myOptions |= QFileDialog.DontUseNativeDialog
        myOptions |= QFileDialog.ShowDirsOnly

        starterFilePath = str(Path(self.settings.value('lightcurvedir', "") + '/' + name))

        self.csvFile, _ = QFileDialog.getSaveFileName(
            self,  # parent
            "Select directory/modify filename",  # title for dialog
            starterFilePath,  # starting directory
            "", options=myOptions)

        if self.csvFile:
            with open(self.csvFile, 'w') as fileObject:
                if not self.aperture_names:
                    fileObject.write('# ' + 'PYOTE ' + version.version() + '\n')
                else:
                    fileObject.write('# PyMovie file written by ' + 'PYOTE ' + version.version() + '\n')

                for hdr in self.headers:
                    fileObject.write(f'#  {hdr}\n')
                if not self.aperture_names:
                    # Handle non-PyMovie csv file
                    columnHeadings = 'FrameNum,timeInfo,primaryData'
                    if len(self.LC2) > 0:
                        columnHeadings += ',LC2'
                    if len(self.LC3) > 0:
                        columnHeadings += ',LC3'
                    if len(self.LC4) > 0:
                        columnHeadings += ',LC4'
                else:
                    columnHeadings = 'FrameNum,timeInfo'
                    for column_name in self.aperture_names:
                        columnHeadings += f',signal-{column_name}'
                fileObject.write(columnHeadings + '\n')

                for i in range(self.table.rowCount()):
                    if self.left <= i <= self.right:
                        line = self.table.item(i, 0).text()
                        for j in range(1, self.table.columnCount()):
                            # Deal with empty columns
                            if self.table.item(i, j) is not None:
                                line += ',' + self.table.item(i, j).text()
                        fileObject.write(line + '\n')

    @staticmethod
    def getTimestampString(timeValue):
        hours = int(timeValue / 3600)
        timeValue -= hours * 3600
        minutes = int(timeValue / 60)
        timeValue -= minutes * 60
        return f'{hours:02d}:{minutes:02d}:{timeValue:0.4f}'

    def writeExampleLightcurveToFile(self, lgtCurve, timeDelta):
        _, name = os.path.split(self.filename)
        name = self.removeCsvExtension(name)

        name += 'PYOTE.example-lightcurve.csv'

        myOptions = QFileDialog.Options()
        # myOptions |= QFileDialog.DontConfirmOverwrite
        myOptions |= QFileDialog.DontUseNativeDialog
        myOptions |= QFileDialog.ShowDirsOnly

        starterFilePath = str(Path(self.settings.value('lightcurvedir', "") + '/' + name))

        self.csvFile, _ = QFileDialog.getSaveFileName(
            self,  # parent
            "Select directory/modify filename",  # title for dialog
            starterFilePath,  # starting directory
            "", options=myOptions)

        if self.csvFile:
            with open(self.csvFile, 'w') as fileObject:
                fileObject.write('# ' + 'PYOTE ' + version.version() + '\n')

                fileObject.write(f'#  Example lightcurve from detectability analysis\n')
                columnHeadings = 'FrameNum,timeInfo,primaryData'
                fileObject.write(columnHeadings + '\n')

                readingTime = 0.0
                for i in range(len(lgtCurve)):
                    ts = self.getTimestampString(readingTime)
                    line = f'{i},[{ts}],{lgtCurve[i]:0.2f}'
                    fileObject.write(line + '\n')
                    readingTime += timeDelta

    @staticmethod
    def copy_desktop_icon_file_to_home_directory():
        if platform.mac_ver()[0]:
            icon_dest_path = f"{os.environ['HOME']}{r'/Desktop/run-pyote'}"
            if not os.path.exists(icon_dest_path):
                # Here is where the .bat file will be when running an installed pyote
                icon_src_path = f"{os.environ['HOME']}" + r"/Anaconda3/Lib/site-packages/pyoteapp/run-pyote-mac.bat"
                if not os.path.exists(icon_src_path):
                    # But here is where the .bat file is during a development run
                    icon_src_path = os.path.join(os.path.split(__file__)[0], 'run-pyote-mac.bat')
                with open(icon_src_path) as src, open(icon_dest_path, 'w') as dest:
                    dest.writelines(src.readlines())
                os.chmod(icon_dest_path, 0o755)  # Make it executable
        else:
            # We must be on a Windows machine because Mac version number was empty
            icon_dest_path = r"C:\Anaconda3\PYOTE.bat"

            if not os.path.exists(icon_dest_path):
                # Here is where the .bat file will be when running an installed pyote
                icon_src_path = r"C:\Anaconda3\Lib\site-packages\pyoteapp\PYOTE.bat"
                if not os.path.exists(icon_src_path):
                    # But here is where the .bat file is during a development run
                    icon_src_path = os.path.join(os.path.split(__file__)[0], 'PYOTE.bat')
                with open(icon_src_path) as src, open(icon_dest_path, 'w') as dest:
                    dest.writelines(src.readlines())

    def toggleManualEntryButton(self):
        if self.manualTimestampCheckBox.isChecked():
            self.manualEntryPushButton.setEnabled(True)
        else:
            self.manualEntryPushButton.setEnabled(False)

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
            self.mainPlot.viewport().setProperty("cursor", QtGui.QCursor(QtCore.Qt.CursorShape.BlankCursor))

    def keyPressEvent(self, ev):
        if ev.key() == QtCore.Qt.Key.Key_Shift:
            # self.showMsg('Shift key pressed')
            if self.blankCursor:
                self.mainPlot.viewport().setProperty("cursor", QtGui.QCursor(QtCore.Qt.CursorShape.ArrowCursor))
                self.blankCursor = False
            else:
                self.blankCursor = True
                self.mainPlot.viewport().setProperty("cursor", QtGui.QCursor(QtCore.Qt.CursorShape.BlankCursor))

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

    # def changeSecondary(self):
    #     if len(self.aperture_names) > 0:
    #         self.secondarySelector.setMaximum(len(self.aperture_names))
    #     else:
    #         self.secondarySelector.setMaximum(self.getNumberOfUnnamedLightCurves())
    #
    #     secondarySelText = self.secondarySelector.text()
    #     normNum = int(secondarySelText)
    #
    #     primarySelText = self.curveToAnalyzeSpinBox.text()
    #     refNum = int(primarySelText)
    #
    #     if len(self.aperture_names) > 0:
    #         pymovieColumnType = self.pymovieDataColumnPrefixComboBox.currentText()
    #         if not refNum == 0:
    #             if (refNum - 1) < len(self.aperture_names):
    #                 self.lightCurveNameEdit.setText(self.aperture_names[refNum - 1])
    #         if normNum == 0:
    #             self.showMsg('There is no secondary reference selected.')
    #         elif (normNum - 1) < len(self.aperture_names):
    #             self.normalizationLightCurveNameEdit.setText(self.aperture_names[normNum - 1])
    #             self.showMsg('Secondary reference ' + secondarySelText + ' selected - PyMovie aperture name: ' +
    #                          self.aperture_names[normNum - 1] + f" ({pymovieColumnType})")
    #     else:
    #         if not normNum == 0:
    #             self.showMsg('Secondary reference ' + secondarySelText + ' selected.')
    #         else:
    #             self.showMsg('There is no secondary reference selected.')
    #
    #     if normNum == 0:
    #         self.yRefStar = []
    #         self.normalizationLightCurveNameEdit.setText('')
    #     if normNum == 1:
    #         self.yRefStar = self.LC1
    #     if normNum == 2:
    #         self.yRefStar = self.LC2
    #     if normNum == 3:
    #         self.yRefStar = self.LC3
    #     if normNum == 4:
    #         self.yRefStar = self.LC4
    #     if normNum > 4:
    #         self.yRefStar = self.extra[normNum - 4 - 1]
    #
    #     self.smoothSecondary = []
    #     self.newRedrawMainPlot()
    #     self.mainPlot.autoRange()

    def getNumberOfUnnamedLightCurves(self):
        if len(self.LC1) == 0:
            return 0
        elif len(self.LC2) == 0:
            return 1
        elif len(self.LC3) == 0:
            return 2
        elif len(self.LC4) == 0:
            return 3
        elif len(self.extra) == 0:
            return 4
        else:
            return 4 + len(self.extra)

    # def changePrimary(self):
    #     if len(self.aperture_names) > 0:
    #         self.curveToAnalyzeSpinBox.setMaximum(len(self.aperture_names))
    #     else:
    #         self.curveToAnalyzeSpinBox.setMaximum(self.getNumberOfUnnamedLightCurves())
    #
    #     selText = self.curveToAnalyzeSpinBox.text()
    #     refNum = int(selText)
    #
    #     selText2 = self.secondarySelector.text()
    #     normNum = int(selText2)
    #
    #     if len(self.aperture_names) > 0:
    #         pymovieColumnType = self.pymovieDataColumnPrefixComboBox.currentText()
    #         if refNum == 0:
    #             self.showMsg('There is no curve selected for analysis.')
    #         elif (refNum - 1) < len(self.aperture_names):
    #             self.lightCurveNameEdit.setText(self.aperture_names[refNum - 1])
    #             self.showMsg('Analyze data ' + selText + ' selected - PyMovie aperture name: ' +
    #                          self.aperture_names[refNum - 1] + f" ({pymovieColumnType})")
    #         if not normNum == 0:
    #             if (normNum - 1) < len(self.aperture_names):
    #                 self.normalizationLightCurveNameEdit.setText(self.aperture_names[normNum - 1])
    #     else:
    #         if not refNum == 0:
    #             self.showMsg('Analyze light curve ' + selText + ' selected.')
    #         else:
    #             self.showMsg('There is no curve selected for analysis.')
    #
    #     if refNum == 0:
    #         self.yValues = []
    #         self.lightCurveNameEdit.setText('')
    #     if refNum == 1:
    #         self.yValues = self.LC1
    #     if refNum == 2:
    #         self.yValues = self.LC2
    #     if refNum == 3:
    #         self.yValues = self.LC3
    #     if refNum == 4:
    #         self.yValues = self.LC4
    #     if refNum > 4:
    #         self.yValues = self.extra[refNum - 4 - 1].copy()
    #
    #     self.solution = None
    #     self.newRedrawMainPlot()
    #     self.mainPlot.autoRange()

    def installLatestVersion(self, pyoteversion):
        self.showMsg(f'Asking to upgrade to: {pyoteversion}')
        pipResult = upgradePyote(pyoteversion)
        for line in pipResult:
            self.showMsg(line, blankLine=False)

        self.showMsg('', blankLine=False)
        self.showMsg('The new version is installed but not yet running.', color='red', bold=True)
        self.showMsg('Close and reopen pyote to start the new version running.', color='red', bold=True)

    def checkForNewVersion(self):
        gotVersion, latestVersion = getMostRecentVersionOfPyOTEViaJson()
        if gotVersion:
            if latestVersion <= version.version():
                # self.showMsg(f'Found the latest version is: {latestVersion}')
                self.showMsg('You are running the most recent version of PyOTE', color='red', bold=True)
            else:
                self.showMsg('Version ' + latestVersion + ' is available', color='red', bold=True)
                # if self.queryWhetherNewVersionShouldBeInstalled() == QMessageBox.Yes:
                #     self.showMsg('You have opted to install latest version of PyOTE')
                #     self.installLatestVersion(f'pyote=={latestVersion}')
                #     self.allowNewVersionPopupCheckbox.setChecked(True)
                #     self.settings.setValue('allowNewVersionPopup', True)
                # else:
                #     self.showMsg('You have declined the opportunity to install latest PyOTE')
        else:
            self.showMsg(f'latestVersion found: {latestVersion}')

    @staticmethod
    def queryWhetherNewVersionShouldBeInstalled():
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Question)
        msg.setText('A newer version of PyOTE is available. Do you wish to install it?')
        msg.setWindowTitle('Get latest version of PyOTE query')
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        retval = msg.exec_()
        return retval

    @staticmethod
    def queryWhetherBlockIntegrationShouldBeAcccepted():
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Question)
        msg.setText(
            'Do you want the pyote estimation of block integration parameters to be used'
            ' for block integration?')
        msg.setWindowTitle('Is auto-determined block integration ok')
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

        exporter = FixedImageExporter(self.falsePositivePlotItem)
        exporter.makeWidthHeightInts()
        targetFileDur = self.graphicFile + '.false-positive.PYOTE.png'
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
            self,  # parent
            "Select directory/modify filename (png will be appended for you)",  # title for dialog
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

            exporter = FixedImageExporter(self.falsePositivePlotItem)
            exporter.makeWidthHeightInts()
            targetFileDur = self.graphicFile + '.false-positive.PYOTE.png'
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
            self,  # parent
            "Select directory/modify filename (png will be appended for you)",  # title for dialog
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

        self.left = None  # Used during block integration
        self.right = None  # "
        self.selPts = []  # "

        self.exponentialDtheoryPts = None
        self.exponentialRtheoryPts = None

        self.penumbralFitCheckBox.setEnabled(False)
        self.penumbralFitCheckBox.setChecked(False)

        self.flashEdges = []
        self.normalized = False
        self.timesAreValid = True  # until we find out otherwise
        self.selectedPoints = {}  # Clear/declare 'selected points' dictionary
        self.baselineXvals = []
        self.baselineYvals = []
        self.underlyingLightcurveAns = None
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
        self.eRegion = None
        self.dLimits = []
        self.rLimits = []
        self.eLimits = []
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
            self.newRedrawMainPlot()
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
                rightEdge = self.right - 1  # Enforce at least 1 'a' point
            if leftEdge < self.left + 1:
                leftEdge = self.left + 1  # Enforce at least 1 'b' point

        if rightEdge < self.left or rightEdge <= leftEdge:
            self.removePointSelections()
            self.newRedrawMainPlot()
            return

        self.setDataLimits.setEnabled(False)

        self.locateEvent.setEnabled(True)

        self.dLimits = [leftEdge, rightEdge]
        self.minEventEdit.clear()
        self.maxEventEdit.clear()

        if self.rLimits:
            self.eventType = 'DandR'
        else:
            self.eventType = 'Donly'

        self.dRegion = pg.LinearRegionItem(
            [leftEdge, rightEdge], movable=False, brush=(0, 200, 0, 50))
        self.mainPlot.addItem(self.dRegion)

        self.showMsg('D zone selected: ' + str([leftEdge, rightEdge]))
        self.removePointSelections()
        self.newRedrawMainPlot()

    def showRzone(self):
        # If the user has not selected any points, we remove any rRegion that may
        # have been present
        if len(self.selectedPoints) == 0:
            self.rRegion = None
            self.rLimits = None
            self.newRedrawMainPlot()
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
                leftEdge = self.left + 1  # Enforce  1 'b' point

        if rightEdge <= leftEdge:
            self.removePointSelections()
            self.newRedrawMainPlot()
            return

        self.setDataLimits.setEnabled(False)

        self.locateEvent.setEnabled(True)

        self.rLimits = [leftEdge, rightEdge]
        self.minEventEdit.clear()
        self.maxEventEdit.clear()

        if self.dLimits:
            # self.DandR.setChecked(True)
            self.eventType = 'DandR'
        else:
            self.eventType = 'Ronly'
            # self.Ronly.setChecked(True)

        self.rRegion = pg.LinearRegionItem(
            [leftEdge, rightEdge], movable=False, brush=(200, 0, 0, 50))
        self.mainPlot.addItem(self.rRegion)

        self.showMsg('R zone selected: ' + str([leftEdge, rightEdge]))
        self.removePointSelections()
        self.newRedrawMainPlot()

    def calculateFlashREdge(self):
        if len(self.selectedPoints) != 2:
            self.showInfo(
                'Exactly two points must be selected for this operation.')
            return

        self.minEventEdit.setText('')
        self.maxEventEdit.setText('')

        selIndices = [key for key, _ in self.selectedPoints.items()]
        selIndices.sort()

        savedLeft = self.left
        savedRight = self.right

        leftEdge = int(min(selIndices))
        rightEdge = int(max(selIndices))

        self.left = leftEdge
        self.right = rightEdge

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
                leftEdge = self.left + 1  # Enforce  1 'b' point

        if rightEdge <= leftEdge:
            self.removePointSelections()
            self.newRedrawMainPlot()
            return

        self.locateEvent.setEnabled(True)

        self.rLimits = [leftEdge, rightEdge]

        if self.dLimits:
            self.eventType = 'DandR'
        else:
            self.eventType = 'Ronly'

        self.rRegion = pg.LinearRegionItem(
            [leftEdge, rightEdge], movable=False, brush=(200, 0, 0, 50))
        self.mainPlot.addItem(self.rRegion)

        self.showMsg('R zone selected: ' + str([leftEdge, rightEdge]))
        self.removePointSelections()
        # self.newRedrawMainPlot()

        self.findEvent()

        self.left = savedLeft
        self.right = savedRight
        self.newRedrawMainPlot()

        if self.solution:
            frameDelta = float(self.yFrame[1]) - float(self.yFrame[0])
            frameZero = float(self.yFrame[0])
            flashFrame = self.solution[1] * frameDelta + frameZero
            # self.flashEdges.append(self.solution[1] + float(self.yFrame[0]))
            self.flashEdges.append(flashFrame)
            self.flashEdges[-1] = '%0.2f' % self.flashEdges[-1]
            msg = 'flash edges (in frame units): %s' % str(self.flashEdges)
            self.showMsg(msg, bold=True, color='red')

    def newNormalize(self):
        self.newSmoothRefStar()  # Produces/updates self.smoothSecondary

        # Find the mean value in the smoothedSecondary curve and use it as ref (previously we had the user
        # click on the point to use, but as that had no effect on magdrop calculations, was never needed,)
        ref = np.mean(self.smoothSecondary)
        # self.showInfo(f'ref: {ref:0.2f}')

        xOffset = 0
        # Look for time shift (xOffset)
        for i, checkBox in enumerate(self.referenceCheckBoxes):
            if checkBox.isChecked():
                xOffset = self.xOffsetSpinBoxes[i].value()

        # Reminder: the smoothSecondary[] only cover self.left to self.right inclusive,
        # hence the index manipulation in the following code
        maxK = len(self.smoothSecondary) - 1
        targetY = []
        referenceY = []
        for i in range(self.left, self.right + 1):
            k = i - xOffset - self.left
            if k < 0 or k > maxK:
                self.yStatus[i] = EXCLUDED
                continue
            else:
                if not self.yStatus[i] == BASELINE:
                    self.yStatus[i] = INCLUDED
                targetY.append(self.yValues[i])
                referenceY.append(self.smoothSecondary[k])
            try:
                self.yValues[i] = (ref * self.yValues[i]) / self.smoothSecondary[k]
            except Exception as e:
                self.showMsg(str(e))

        # Compute standard deviation of normalized target curve - minimize this metric
        yValuesInMetric = []
        baselineSelectionAvailable = False
        for i in range(len(self.yValues)):
            if self.yStatus[i] == BASELINE:
                baselineSelectionAvailable = True
                break
        if baselineSelectionAvailable:
            for i in range(len(self.yValues)):
                if self.yStatus[i] == BASELINE:
                    yValuesInMetric.append(self.yValues[i])
        else:
            for i in range(len(self.yValues)):
                if self.yStatus[i] == INCLUDED:
                    yValuesInMetric.append(self.yValues[i])

        targetStd = np.std(yValuesInMetric)
        self.showMsg(f'Flatness  (minimize this value): {targetStd:0.2f} '
                     f'(readings: {self.smoothingIntervalSpinBox.value()})  (X offset: {xOffset})',
                     color='green', bold=True, alternateLogFile=self.normalizationLogFile)

        self.fillTableViewOfData()  # This should capture/write the effects of the normalization to the table

        self.normalized = True

    def newSmoothRefStar(self):
        if self.right is None:
            self.right = self.dataLen
        if self.left is None:
            self.left = 0

        if (self.right - self.left) < 2:
            self.showInfo('The smoothing algorithm requires a minimum data set of 3 points')
            return

        y = [self.yRefStar[i] for i in range(self.left, self.right + 1)]

        userSpecedWindow = self.smoothingIntervalSpinBox.value()

        # if userSpecedWindow < 3:
        #     userSpecedWindow = 3

        try:
            if len(y) > userSpecedWindow:
                window = userSpecedWindow
            else:
                window = len(y)

            # Enforce the odd window size required by savgol_filter()
            if window % 2 == 0:
                window += 1

            if window == 1:
                self.smoothSecondary = np.array(y)
            else:
                # We do a double pass with a first order (straight line with slope) savgol filter
                filteredY = scipy.signal.savgol_filter(np.array(y), window, 1)
                self.smoothSecondary = scipy.signal.savgol_filter(filteredY, window, 1)

        except Exception as e:
            self.showMsg(str(e))

        # self.showMsg('Smoothing of secondary star light-curve performed with window size: %i' % window)

    def switchToTabNamed(self, title):
        tabCount = self.tabWidget.count()  # Returns number of tabs
        for i in range(tabCount):
            if self.tabWidget.tabText(i) == title:
                self.tabWidget.setCurrentIndex(i)
                return

        self.popupMsg(f'Cannot find tab with title: {title}')

    # def smoothRefStar(self):
    #     if (self.right - self.left) < 4:
    #         self.showInfo('The smoothing algorithm requires a minimum selection of 5 points')
    #         return
    #
    #     y = [self.yRefStar[i] for i in range(self.left, self.right+1)]
    #
    #     userSpecedWindow = 101
    #
    #     numPts = self.numSmoothPointsEdit.text().strip()
    #     if numPts:
    #         if not numPts.isnumeric():
    #             self.showInfo('Invalid entry for smoothing window size - defaulting to 101')
    #         else:
    #             userSpecedWindow = int(numPts)
    #             if userSpecedWindow < 5:
    #                 self.showInfo('smoothing window must be size 5 or greater - defaulting to 101')
    #                 userSpecedWindow = 101
    #
    #     window = None
    #     try:
    #         if len(y) > userSpecedWindow:
    #             window = userSpecedWindow
    #         else:
    #             window = len(y)
    #
    #         # Enforce the odd window size required by savgol_filter()
    #         if window % 2 == 0:
    #             window -= 1
    #
    #         # We do a double pass with a third order savgol filter
    #         filteredY = scipy.signal.savgol_filter(np.array(y), window, 3)
    #         self.smoothSecondary = scipy.signal.savgol_filter(filteredY, window, 3)
    #
    #         # New in version 3.7.2: we remove the extrapolated points at each end of self.smoothSecondary
    #         self.extra_point_count = window // 2
    #         self.selectedPoints = {self.left + self.extra_point_count: 3,
    #                                self.right - self.extra_point_count: 3}
    #         saved_smoothSecondary = self.smoothSecondary
    #         self.doTrim()
    #         self.smoothSecondary = saved_smoothSecondary
    #         self.smoothSecondary = self.smoothSecondary[self.extra_point_count:-self.extra_point_count]
    #
    #         # self.left += self.extra_point_count
    #         # self.right -= self.extra_point_count
    #         self.newRedrawMainPlot()
    #     except Exception as e:
    #         self.showMsg(str(e))
    #
    #     self.showMsg('Smoothing of secondary star light curve performed with window size: %i' % window)
    #
    #     self.normalizeButton.setEnabled(True)

    def toggleDisplayOfTimestampErrors(self):
        self.newRedrawMainPlot()
        self.mainPlot.autoRange()

    # def toggleDisplayOfSecondaryStar(self):
    #     if self.showSecondaryCheckBox.isChecked():
    #         self.secondarySelector.setEnabled(True)
    #     else:
    #         self.secondarySelector.setEnabled(False)
    #
    #     if self.showSecondaryCheckBox.isChecked():
    #         self.changeSecondary()
    #     else:
    #         self.newRedrawMainPlot()
    #         self.mainPlot.autoRange()

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

    def fillPrimaryAndRef(self):
        # Load self.yValues and sel.yRefStar with proper light curves as
        # indicated by the spinner values

        # Get indices of selected primary and reference light curves
        primary = 0
        for i, checkBox in enumerate(self.targetCheckBoxes):
            if checkBox.isChecked():
                primary = i
                break

        reference = None
        for i, checkBox in enumerate(self.referenceCheckBoxes):
            if checkBox.isChecked():
                reference = i
                break

        if primary == 0:
            self.yValues = self.LC1
        elif primary == 1:
            self.yValues = self.LC2
        elif primary == 2:
            self.yValues = self.LC3
        elif primary == 3:
            self.yValues = self.LC4
        else:
            self.yValues = self.extra[primary - 4]

        if reference is not None:
            if reference == 0:
                self.yRefStar = self.LC1
            elif reference == 1:
                self.yRefStar = self.LC2
            elif reference == 2:
                self.yRefStar = self.LC3
            elif reference == 3:
                self.yRefStar = self.LC4
            else:
                self.yRefStar = self.extra[reference - 4]
        else:
            self.yRefStar = []

        # noinspection PyUnusedLocal
        self.yStatus = [INCLUDED for _ in range(self.dataLen)]

    def doIntegration(self):

        if len(self.selectedPoints) == 0:
            self.showMsg('Analysis of all possible block integration sizes and offsets',
                         color='red', bold=True)
            notchList = []
            kList = []
            offsetList = []

            self.progressBar.setValue(0)
            progress = 0
            integrationSizes = [2, 4, 8, 16, 32, 48, 64, 96, 128, 256]
            for k in integrationSizes:
                kList.append(k)
                ans = mean_std_versus_offset(k, self.yValues)
                progress += 1
                self.progressBar.setValue((progress // len(integrationSizes)) * 100)

                QtWidgets.QApplication.processEvents()
                offsetList.append(np.argmin(ans))
                median = np.median(ans)
                notch = np.min(ans) / median
                notchList.append(notch)
                s = '%3d notch %0.2f [' % (k, notch)

                for item in ans:
                    s = s + '%0.1f, ' % item
                self.showMsg(s[:-2] + ']', blankLine=False)
                # QtGui.QApplication.processEvents()
                QtWidgets.QApplication.processEvents()

            self.progressBar.setValue(0)
            # QtGui.QApplication.processEvents()
            QtWidgets.QApplication.processEvents()

            best = int(np.argmin(notchList))
            blockSize = kList[best]
            offset = int(offsetList[best])
            self.showMsg(' ', blankLine=False)
            s = '\r\nBest integration estimate: blockSize: %d @ offset %d' % (blockSize, offset)
            self.showMsg(s, color='red', bold=True)

            brush1 = (0, 200, 0, 70)
            brush2 = (200, 0, 0, 70)

            leftEdge = offset - 0.5
            rightEdge = leftEdge + blockSize
            bFlag = True

            while rightEdge <= len(self.yValues):
                if bFlag:
                    bFlag = False
                    brushToUse = brush2
                else:
                    bFlag = True
                    brushToUse = brush1

                if bFlag:
                    self.mainPlot.addItem(pg.LinearRegionItem([leftEdge, rightEdge],
                                                              movable=False, brush=brushToUse))
                leftEdge += blockSize
                rightEdge += blockSize

            # Set the integration selection point indices
            self.bint_left = offset
            self.bint_right = offset + blockSize - 1
            self.selPts = [self.bint_left, self.bint_right]

            self.acceptBlockIntegration.setEnabled(True)

        elif len(self.selectedPoints) != 2:
            self.showInfo('Exactly two points must be selected for a block integration')
            return
        else:
            self.bint_left = None  # Force use of selectPoints in applyIntegration()
            # self.acceptBlockIntegration.setEnabled(False)
            self.applyIntegration()

    def validateSinglePointDrop(self):
        if not self.userDeterminedBaselineStats:
            self.showInfo(f'You need to select baseline points and calculate statistics first. '
                          f'\n\nGo to the Noise anlaysis tab to do this.')
            return

        if len(self.selectedPoints) != 1:
            self.showInfo('Exactly one point must be selected for this operation.')
            return
        selectedPoints = [key for key in self.selectedPoints.keys()]
        selectedPoint = selectedPoints[0]
        # self.showInfo(f'{selectedPoint} was selected.')

        self.errBarWin = pg.GraphicsWindow(
            title='simulated drop distribution for single point event')
        self.errBarWin.resize(1200, 500)
        layout = QtWidgets.QGridLayout()
        self.errBarWin.setLayout(layout)
        drop = self.B - self.yValues[selectedPoint]
        pw, falsePositive, probability = self.falsePositiveReport(
            event_duration=1, num_trials=50000, observation_size=self.right - self.left + 1,
            observed_drop=drop,
            posCoefs=self.corCoefs, sigma=self.sigmaB)
        layout.addWidget(pw, 0, 0)

        self.showMsg(f"Reading number {selectedPoint} with drop {drop:0.2f} was selected for validation", color='blue',
                     bold=True, blankLine=False)
        if falsePositive:
            self.showMsg(f'==== The single point event is NOT valid - it has non-zero chance of being due to noise',
                         color='red', bold=True)
        else:
            self.showMsg(f'==== The single point event is valid - it is unlikely to stem from noise',
                         color='blue', bold=True)

    def applyIntegration(self):
        if self.bint_left is None:
            if self.outliers:
                self.showInfo('This data set contains some erroneous time steps, which have ' +
                              'been marked with red lines.  Best practice is to ' +
                              'choose an integration block that is ' +
                              'positioned in an unmarked region, hopefully containing ' +
                              'the "event".  Block integration ' +
                              'proceeds to the left and then to the right of the marked block.')

            self.selPts = [key for key in self.selectedPoints.keys()]
            self.removePointSelections()
            self.bint_left = min(self.selPts)
            self.bint_right = max(self.selPts)

        # Time to do the work
        p0 = self.bint_left
        span = self.bint_right - self.bint_left + 1  # Number of points in integration block
        self.blockSize = span
        newFrame = []
        newTime = []
        newLC1 = []
        newLC2 = []
        newLC3 = []
        newLC4 = []
        newDemoLightCurve = []
        newExtra = [[] for _ in range(len(self.extra))]

        if not self.blockSize % 2 == 0:
            self.showInfo(f'Blocksize is {self.blockSize}\n\nAn odd number for blocksize is likely an error!')

        p = p0 - span  # Start working toward the left
        while p > 0:
            avg = np.mean(self.LC1[p:(p + span)])
            newLC1.insert(0, avg)

            if len(self.LC2) > 0:
                avg = np.mean(self.LC2[p:(p + span)])
                newLC2.insert(0, avg)

            if len(self.LC3) > 0:
                avg = np.mean(self.LC3[p:(p + span)])
                newLC3.insert(0, avg)

            if len(self.LC4) > 0:
                avg = np.mean(self.LC4[p:(p + span)])
                newLC4.insert(0, avg)

            if len(newExtra) > 0:
                for k, lc in enumerate(self.extra):
                    avg = np.mean(lc[p:(p + span)])
                    newExtra[k].insert(0, avg)

            if len(self.demoLightCurve) > 0:
                avg = np.mean(self.demoLightCurve[p:(p + span)])
                newDemoLightCurve.insert(0, avg)

            newFrame.insert(0, self.yFrame[p])
            newTime.insert(0, self.yTimes[p])
            p = p - span

        p = p0  # Start working toward the right
        while p < self.dataLen - span:
            avg = np.mean(self.LC1[p:(p + span)])
            newLC1.append(avg)

            if len(self.LC2) > 0:
                avg = np.mean(self.LC2[p:(p + span)])
                newLC2.append(avg)

            if len(self.LC3) > 0:
                avg = np.mean(self.LC3[p:(p + span)])
                newLC3.append(avg)

            if len(self.LC4) > 0:
                avg = np.mean(self.LC4[p:(p + span)])
                newLC4.append(avg)

            if len(newExtra) > 0:
                for k, lc in enumerate(self.extra):
                    avg = np.mean(lc[p:(p + span)])
                    newExtra[k].append(avg)

            if len(self.demoLightCurve) > 0:
                avg = np.mean(self.demoLightCurve[p:(p + span)])
                newDemoLightCurve.append(avg)

            newFrame.append(self.yFrame[p])
            newTime.append(self.yTimes[p])
            p = p + span

        self.dataLen = len(newLC1)

        self.LC1 = np.array(newLC1)
        self.LC2 = np.array(newLC2)
        self.LC3 = np.array(newLC3)
        self.LC4 = np.array(newLC4)
        if len(newExtra) > 0:
            for k in range(len(newExtra)):
                self.extra[k] = np.array(newExtra[k])
        self.demoLightCurve = np.array(newDemoLightCurve)

        # auto-select all points
        self.left = 0
        self.right = self.dataLen - 1

        self.fillPrimaryAndRef()

        self.yTimes = newTime[:]
        self.yFrame = newFrame[:]
        self.fillTableViewOfData()

        self.selPts.sort()
        self.showMsg('Block integration started at entry ' + str(self.selPts[0]) +
                     ' with block size of ' + str(self.selPts[1] - self.selPts[0] + 1) + ' readings')

        self.timeDelta, self.outliers, self.errRate = getTimeStepAndOutliers(self.yTimes)
        self.showMsg('timeDelta: ' + fp.to_precision(self.timeDelta, 6) + ' seconds per block', blankLine=False)
        self.showMsg('timestamp error rate: ' + fp.to_precision(100 * self.errRate, 2) + '%')

        self.expDurEdit.setText(fp.to_precision(self.timeDelta, 6))

        self.illustrateTimestampOutliers()

        self.doBlockIntegration.setEnabled(False)
        self.acceptBlockIntegration.setEnabled(False)

        self.newRedrawMainPlot()
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
        self.suppressNormalization = True
        self.newRedrawMainPlot()  # Redraw plot to show selection change
        self.suppressNormalization = False

    def processClick(self, event):
        # Don't allow mouse clicks to select points unless the cursor is blank
        if self.blankCursor and self.left is not None and self.right is not None:
            # This try/except handles case where user clicks in plot area before a
            # plot has been drawn.
            try:
                mousePoint = self.mainPlotViewBox.mapSceneToView(event.scenePos())
                index = round(mousePoint.x())
                if event.button() == 1:  # left button clicked?
                    if index < self.left:
                        index = self.left
                    if index > self.right:
                        index = self.right
                    self.togglePointSelected(index)
                    self.acceptBlockIntegration.setEnabled(False)
                    self.suppressNormalization = True
                # Move the table view of data so that clicked point data is visible
                self.table.setCurrentCell(index, 0)
            except AttributeError:
                pass

    def initializeTableView(self):
        self.table.clear()
        self.table.setRowCount(3)
        if not self.aperture_names:
            # Handle non-PyMovie csv file
            colLabels = ['FrameNum', 'timeInfo', 'LC1', 'LC2', 'LC3', 'LC4']
            if len(self.LC1) > 0:
                self.lightcurveTitles[0].setText('LC1')
                self.targetCheckBoxes[0].setEnabled(True)
                self.showCheckBoxes[0].setEnabled(True)
                self.yOffsetSpinBoxes[0].setEnabled(True)
                self.yOffsetSpinBoxes[0].setValue(0)
                self.referenceCheckBoxes[0].setEnabled(True)

            if len(self.LC2) > 0:
                self.lightcurveTitles[1].setText('LC2')
                self.targetCheckBoxes[1].setEnabled(True)
                self.showCheckBoxes[1].setEnabled(True)
                self.yOffsetSpinBoxes[1].setEnabled(True)
                self.yOffsetSpinBoxes[1].setValue(0)
                self.referenceCheckBoxes[1].setEnabled(True)

            if len(self.LC3) > 0:
                self.lightcurveTitles[2].setText('LC3')
                self.targetCheckBoxes[2].setEnabled(True)
                self.showCheckBoxes[2].setEnabled(True)
                self.yOffsetSpinBoxes[2].setEnabled(True)
                self.yOffsetSpinBoxes[2].setValue(0)
                self.referenceCheckBoxes[2].setEnabled(True)

            if len(self.LC4) > 0:
                self.lightcurveTitles[3].setText('LC4')
                self.targetCheckBoxes[3].setEnabled(True)
                self.showCheckBoxes[3].setEnabled(True)
                self.yOffsetSpinBoxes[3].setEnabled(True)
                self.yOffsetSpinBoxes[3].setValue(0)
                self.referenceCheckBoxes[3].setEnabled(True)

            self.table.setColumnCount(6)
        else:
            self.table.setColumnCount(2 + len(self.aperture_names))
            colLabels = ['FrameNum', 'timeInfo']
            k = 0
            for column_name in self.aperture_names:
                colLabels.append(column_name)
                if k < 10:
                    self.lightcurveTitles[k].setText(column_name)
                    self.targetCheckBoxes[k].setEnabled(True)
                    self.showCheckBoxes[k].setEnabled(True)
                    self.yOffsetSpinBoxes[k].setEnabled(True)
                    self.yOffsetSpinBoxes[k].setValue(0)
                    self.referenceCheckBoxes[k].setEnabled(True)
                    k += 1

        self.table.setHorizontalHeaderLabels(colLabels)

    def closeEvent(self, event):
        # Open (or create) file for holding 'sticky' stuff
        self.settings = QSettings('pyote.ini', QSettings.IniFormat)

        self.settings.setValue('allowNewVersionPopup', self.allowNewVersionPopupCheckbox.isChecked())

        self.settings.setValue('lineWidth', self.lineWidthSpinner.value())
        self.settings.setValue('dotSize', self.dotSizeSpinner.value())

        tabOrderList = []
        numTabs = self.tabWidget.count()
        # print(f'numTabs: {numTabs}')
        for i in range(numTabs):
            tabName = self.tabWidget.tabText(i)
            # print(f'{i}: |{tabName}|')
            tabOrderList.append(tabName)

        self.settings.setValue('tablist', tabOrderList)
        # Capture the close request and update 'sticky' settings
        self.settings.setValue('size', self.size())
        self.settings.setValue('pos', self.pos())
        self.settings.setValue('usediff', self.enableDiffractionCalculationBox.isChecked())
        self.settings.setValue('doOCRcheck', self.showOCRcheckFramesCheckBox.isChecked())
        self.settings.setValue('showTimestamps', self.showTimestampsCheckBox.isChecked())
        self.settings.setValue('showCameraResponse', self.showCameraResponseCheckBox.isChecked())

        self.settings.setValue('ne3NotInUse', self.ne3NotInUseRadioButton.isChecked())
        self.settings.setValue('dnrOff', self.dnrOffRadioButton.isChecked())
        self.settings.setValue('dnrLow', self.dnrLowRadioButton.isChecked())
        self.settings.setValue('dnrMiddle', self.dnrMiddleRadioButton.isChecked())
        self.settings.setValue('dnrHigh', self.dnrHighRadioButton.isChecked())

        self.settings.setValue('dnrLowDtc', self.dnrLowDspinBox.value())
        self.settings.setValue('dnrLowRtc', self.dnrLowRspinBox.value())

        self.settings.setValue('dnrMiddleDtc', self.dnrMiddleDspinBox.value())
        self.settings.setValue('dnrMiddleRtc', self.dnrMiddleRspinBox.value())

        self.settings.setValue('dnrHighDtc', self.dnrHighDspinBox.value())
        self.settings.setValue('dnrHighRtc', self.dnrHighRspinBox.value())

        self.helperThing.close()

        tabOrderList = []
        numTabs = self.tabWidget.count()
        # print(f'numTabs: {numTabs}')
        for i in range(numTabs):
            tabName = self.tabWidget.tabText(i)
            # print(f'{i}: |{tabName}|')
            tabOrderList.append(tabName)

        self.settings.setValue('tablist', tabOrderList)

        self.settings.setValue('splitterOne', self.splitterOne.saveState())
        self.settings.setValue('splitterTwo', self.splitterTwo.saveState())
        self.settings.setValue('splitterThree', self.splitterThree.saveState())

        if self.d_underlying_lightcurve:
            matplotlib.pyplot.close(self.d_underlying_lightcurve)

        if self.r_underlying_lightcurve:
            matplotlib.pyplot.close(self.r_underlying_lightcurve)

        for frame_view in self.frameViews:
            if frame_view:
                frame_view.close()

        curDateTime = datetime.datetime.today().ctime()
        self.showMsg('')
        self.showMsg('#' * 20 + ' Session ended: ' + curDateTime + '  ' + '#' * 20)

        if self.errBarWin:
            self.errBarWin.close()

        event.accept()

    def rowClick(self, row):
        self.highlightReading(row)

    def cellClick(self, row):
        self.togglePointSelected(row)

    def highlightReading(self, rdgNum):
        x = [rdgNum]
        y = [self.yValues[x]]
        self.newRedrawMainPlot()
        self.mainPlot.plot(x, y, pen=None, symbol='o', symbolPen=(255, 0, 0),
                           symbolBrush=(255, 255, 0), symbolSize=10)

    def showMsg(self, msg, color=None, bold=False, blankLine=True, alternateLogFile=None):
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
        if alternateLogFile is not None:
            fileObject = open(alternateLogFile, 'a')
            fileObject.write(msg + '\n')
            if blankLine:
                fileObject.write('\n')
            fileObject.close()
        elif self.logFile:
            fileObject = open(self.logFile, 'a')
            fileObject.write(msg + '\n')
            if blankLine:
                fileObject.write('\n')
            fileObject.close()

    def reportSpecialProcedureUsed(self):
        if self.blockSize == 1:
            self.showMsg('This light curve has not been block integrated.',
                         color='blue', bold=True, blankLine=False)
        else:
            self.showMsg('Block integration of size %d has been applied to '
                         'this light curve.' %
                         self.blockSize, color='blue', bold=True, blankLine=False)

        if self.normalized:
            self.showMsg('', blankLine=False)
            self.showMsg('This light curve has been normalized against a '
                         'reference star.',
                         color='blue', bold=True, blankLine=False)
            for i, checkBox in enumerate(self.referenceCheckBoxes):
                if checkBox.isChecked():
                    self.showMsg(f'... {self.lightcurveTitles[i].text()} was used for normalization with '
                                 f'X offset: {self.xOffsetSpinBoxes[i].value()}', color='blue', bold=True,
                                 blankLine=False)
                    self.showMsg(f'... {self.smoothingIntervalSpinBox.value()} readings were used for smoothing',
                                 color='blue', bold=True, blankLine=True)

        if not (self.left == 0 and self.right == self.dataLen - 1):
            self.showMsg('This light curve has been trimmed.',
                         color='blue', bold=True, blankLine=False)

        self.showMsg('', blankLine=False)

        ans = self.validateLightcurveDataInput()
        if ans['success']:
            self.showMsg(f'The following lightcurve parameters were utilized:',
                         color='blue', bold=True)

            if self.enableDiffractionCalculationBox.isChecked():
                self.showMsg(f"==== use diff: is checked", bold=True, blankLine=False)
            else:
                self.showMsg(f"==== use diff: is NOT checked", bold=True, blankLine=False)

            if ans['exp_dur'] is not None:
                self.showMsg(f"==== exp: {ans['exp_dur']:0.6f}", bold=True, blankLine=False)

            if ans['ast_dist'] is not None:
                self.showMsg(f"==== dist(AU): {ans['ast_dist']:0.4f}", bold=True, blankLine=False)

            if ans['shadow_speed'] is not None:
                self.showMsg(f"==== speed(km/sec): {ans['shadow_speed']:0.4f}", bold=True, blankLine=False)

            if ans['star_diam'] is not None:
                self.showMsg(f"==== Star diam(mas): {ans['star_diam']:0.4f}", bold=True, blankLine=False)
                if ans['d_angle'] is not None:
                    self.showMsg(f"==== D limb angle: {ans['d_angle']:0.1f}", bold=True, blankLine=False)
                if ans['r_angle'] is not None:
                    self.showMsg(f"==== R limb angle: {ans['r_angle']:0.1f}", bold=True, blankLine=False)

        else:
            self.showMsg(f'Some invalid entries were found in the lightcurve parameters panel',
                         color='blue', bold=True, blankLine=False)

        if not self.ne3NotInUseRadioButton.isChecked():
            self.writeNe3UsageReport()

        # for i, checkBox in enumerate(self.referenceCheckBoxes):
        #     if checkBox.isChecked():
        #         self.showMsg('', blankLine=False)
        #         self.showMsg(f"{self.lightcurveTitles[i].text()} "
        #                      f"used for normalization with smoothing interval of {self.smoothingIntervalSpinBox.value()}",
        #                      bold=True, blankLine=False)

        self.showMsg('', blankLine=False)

    def computeErrorBarPair(self, deltaHi, deltaLo, edge):
        noiseAsymmetry = self.snrA / self.snrB
        if (noiseAsymmetry > 0.7) and (noiseAsymmetry < 1.3):
            plus = (deltaHi - deltaLo) / 2
            minus = plus
        else:
            if edge == 'D':
                plus = deltaHi
                minus = -deltaLo
            else:
                plus = -deltaLo  # Deliberate 'inversion'
                minus = deltaHi  # Deliberate 'inversion'

        return plus, minus

    def Dreport(self, deltaDhi, deltaDlo):
        D, _ = self.solution

        intD = int(D)  # So that we can do lookup in the data table

        plusD, minusD = self.computeErrorBarPair(deltaHi=deltaDhi, deltaLo=deltaDlo, edge='D')

        # Save these for the 'envelope' plotter
        self.plusD = plusD
        self.minusD = minusD

        frameNum = float(self.yFrame[intD])

        Dframe = (D - intD) * self.framesPerEntry() + frameNum
        self.showMsg('D: %.2f {+%.2f,-%.2f} (frame number)' % (Dframe, plusD * self.framesPerEntry(),
                                                               minusD * self.framesPerEntry()),
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

        plusR, minusR = self.computeErrorBarPair(deltaHi=deltaRhi, deltaLo=deltaRlo, edge='R')

        # Save these for the 'envelope' plotter
        self.plusR = plusR
        self.minusR = minusR

        intR = int(R)
        frameNum = float(self.yFrame[intR])
        Rframe = (R - intR) * self.framesPerEntry() + frameNum
        self.showMsg('R: %.2f {+%.2f,-%.2f} (frame number)' % (Rframe, plusR * self.framesPerEntry(),
                                                               minusR * self.framesPerEntry()),
                     blankLine=False)

        ts = self.yTimes[int(R)]
        time = convertTimeStringToTime(ts)
        adjTime = time + (R - int(R)) * self.timeDelta
        ts = convertTimeToTimeString(adjTime)
        self.showMsg('R: %s  {+%.4f,-%.4f} seconds' %
                     (ts, plusR * self.timeDelta, minusR * self.timeDelta)
                     )
        return adjTime

    # noinspection PyStringFormat
    def confidenceIntervalReport(self, numSigmas, deltaDurhi, deltaDurlo, deltaDhi, deltaDlo,
                                 deltaRhi, deltaRlo):

        D, R = self.solution

        self.showMsg('B: %0.2f  {+/- %0.2f}  sigmaB: %0.2f' %
                     (self.B, numSigmas * self.sigmaB / np.sqrt(self.nBpts), self.sigmaB))
        self.showMsg('A: %0.2f  {+/- %0.2f}  sigmaA: %0.2f' %
                     (self.A, numSigmas * self.sigmaA / np.sqrt(self.nApts), self.sigmaA))

        self.magdropReport(numSigmas)

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
                         ((R - D) * self.framesPerEntry(),
                          plusDur * self.framesPerEntry(), minusDur * self.framesPerEntry()),
                         blankLine=False)
            plusDur = ((deltaDurhi - deltaDurlo) / 2) * self.timeDelta
            minusDur = plusDur
            self.showMsg('Duration (R - D): %.4f {+%.4f,-%.4f} seconds' %
                         (Rtime - Dtime, plusDur, minusDur))

    # noinspection PyStringFormat
    def penumbralConfidenceIntervalReport(self, numSigmas, deltaDurhi, deltaDurlo, deltaDhi, deltaDlo,
                                          deltaRhi, deltaRlo):

        D, R = self.solution

        self.showMsg('B: %0.2f  {+/- %0.2f}' % (self.B, numSigmas * self.sigmaB / np.sqrt(self.nBpts)))
        self.showMsg('A: %0.2f  {+/- %0.2f}' % (self.A, numSigmas * self.sigmaA / np.sqrt(self.nApts)))

        self.magdropReport(numSigmas)

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
                         ((R - D) * self.framesPerEntry(),
                          plusDur * self.framesPerEntry(), minusDur * self.framesPerEntry()),
                         blankLine=False)
            plusDur = ((deltaDurhi - deltaDurlo) / 2) * self.timeDelta
            minusDur = plusDur
            self.showMsg('Duration (R - D): %.4f {+%.4f,-%.4f} seconds' %
                         (Rtime - Dtime, plusDur, minusDur))

    def magDropString(self, B, A, numSigmas):
        if not 0 < A < B:
            return 'NA because 0 < A < B is not satisfied'
        stdB = self.sigmaB / np.sqrt(self.nBpts)
        stdA = self.sigmaA / np.sqrt(self.nApts)
        ratio = A / B
        ratioError = numSigmas * np.sqrt((stdB / B) ** 2 + (stdA / A) ** 2) * ratio
        lnError = ratioError / ratio
        magdroperr = (2.5 / np.log(10.0)) * lnError
        magDrop = (np.log10(B) - np.log10(A)) * 2.5
        if numSigmas == 1:
            ciStr = '(0.68 ci)'
        elif numSigmas == 2:
            ciStr = '(0.95 ci)'
        else:
            ciStr = '(0.9973 ci)'
        return f'{magDrop:0.3f}  +/- {magdroperr:0.3f}  {ciStr}'

    def magdropReport(self, numSigmas):
        Anom = self.A
        Bnom = self.B

        self.showMsg(f'magDrop: {self.magDropString(Bnom, Anom, numSigmas)}')

    # noinspection PyStringFormat
    def finalReportPenumbral(self):

        self.displaySolution(True)
        self.minusD = self.plusD = self.penumbralDerrBar  # This the 2 sigma (95% ci) value
        self.minusR = self.plusR = self.penumbralRerrBar
        self.drawEnvelope()  # Shows error bars at the 95% ci level

        self.deltaDlo68 = self.deltaDhi68 = self.plusD / 2.0
        self.deltaDlo95 = self.deltaDhi95 = self.plusD
        self.deltaDlo99 = self.deltaDhi99 = 3.0 * self.plusD / 2.0

        self.deltaRlo68 = self.deltaRhi68 = self.plusR / 2.0
        self.deltaRlo95 = self.deltaRhi95 = self.plusR
        self.deltaRlo99 = self.deltaRhi99 = 3.0 * self.plusR / 2.0

        self.deltaDurhi68 = np.sqrt(self.deltaDhi68 ** 2 + self.deltaRhi68 ** 2)
        self.deltaDurlo68 = - self.deltaDurhi68
        self.deltaDurhi95 = 2.0 * self.deltaDurhi68
        self.deltaDurlo95 = - self.deltaDurhi95
        self.deltaDurhi99 = 3.0 * self.deltaDurhi68
        self.deltaDurlo99 = - self.deltaDurhi99

        # Grab the D and R values found and apply our timing convention
        D, R = self.solution

        if self.eventType == 'DandR':
            self.showMsg('Timestamp validity check ...')
            self.reportTimeValidity(D, R)

        # self.calcNumBandApoints()

        self.showMsg('================= 0.68 containment interval report =================')

        self.penumbralConfidenceIntervalReport(1, self.deltaDurhi68, self.deltaDurlo68,
                                               self.deltaDhi68, self.deltaDlo68,
                                               self.deltaRhi68, self.deltaRlo68)

        self.showMsg('=============== end 0.68 containment interval report ===============')

        self.showMsg('================= 0.95 containment interval report =================')

        self.penumbralConfidenceIntervalReport(2, self.deltaDurhi95, self.deltaDurlo95,
                                               self.deltaDhi95, self.deltaDlo95,
                                               self.deltaRhi95, self.deltaRlo95)

        self.showMsg('=============== end 0.95 containment interval report ===============')

        self.showMsg('================= 0.9973 containment interval report ===============')

        self.penumbralConfidenceIntervalReport(3, self.deltaDurhi99, self.deltaDurlo99,
                                               self.deltaDhi99, self.deltaDlo99,
                                               self.deltaRhi99, self.deltaRlo99)

        self.showMsg('=============== end 0.9973 containment interval report =============')

        self.doDframeReport()
        self.doRframeReport()
        self.doDurFrameReport()

        self.showMsg('=============== Summary report for Excel file =====================')

        self.reportSpecialProcedureUsed()  # This includes use of asteroid distance/speed and star diameter

        if not self.timesAreValid:
            self.showMsg("Times are invalid due to corrupted timestamps!",
                         color='red', bold=True)

        self.showMsg(f'magDrop: {self.magDropString(self.B, self.A, 2)}')
        self.xlsxDict['Comment'] = f'Nominal measured mag drop = {self.magDropString(self.B, self.A, 2)}'

        self.showMsg('snr: %0.2f' % self.snrB)

        self.doDtimeReport()
        self.doRtimeReport()
        self.doDurTimeReport()

        self.showMsg(
            'Enter D and R error bars for each containment interval in Excel spreadsheet without + or - sign (assumed to be +/-)')

        self.showMsg('=========== end Summary report for Excel file =====================')

        self.showMsg("Solution 'envelope' in the main plot drawn using 0.95 containment interval error bars")

        return

    def finalReport(self, false_positive, false_probability):

        self.xlsxDict = {}
        self.writeDefaultGraphicsPlots()

        # Grab the D and R values found and apply our timing convention
        D, R = self.solution

        if self.eventType == 'DandR':
            self.showMsg('Timestamp validity check ...')
            self.reportTimeValidity(D, R)

        self.calcNumBandApoints()

        self.showMsg('================= 0.68 containment interval report =================')

        self.confidenceIntervalReport(1, self.deltaDurhi68, self.deltaDurlo68,
                                      self.deltaDhi68, self.deltaDlo68,
                                      self.deltaRhi68, self.deltaRlo68)

        self.showMsg('=============== end 0.68 containment interval report ===============')

        self.showMsg('================= 0.95 containment interval report =================')

        self.confidenceIntervalReport(2, self.deltaDurhi95, self.deltaDurlo95,
                                      self.deltaDhi95, self.deltaDlo95,
                                      self.deltaRhi95, self.deltaRlo95)

        envelopePlusR = self.plusR
        envelopePlusD = self.plusD
        envelopeMinusR = self.minusR
        envelopeMinusD = self.minusD

        self.showMsg('=============== end 0.95 containment interval report ===============')

        self.showMsg('================= 0.9973 containment interval report ===============')

        self.confidenceIntervalReport(3, self.deltaDurhi99, self.deltaDurlo99,
                                      self.deltaDhi99, self.deltaDlo99,
                                      self.deltaRhi99, self.deltaRlo99)

        self.showMsg('=============== end 0.9973 containment interval report =============')

        # Set the values to be used for the envelope plot (saved during 0.95 ci calculations)
        self.plusR = envelopePlusR
        self.plusD = envelopePlusD
        self.minusR = envelopeMinusR
        self.minusD = envelopeMinusD

        self.doDframeReport()
        self.doRframeReport()
        self.doDurFrameReport()

        self.showMsg('=============== Summary report for Excel file =====================')

        self.reportSpecialProcedureUsed()  # This includes use of asteroid distance/speed and star diameter

        if false_positive:
            self.showMsg(f"This 'drop' has a {false_probability:0.5f} probability of being an artifact of noise.",
                         bold=True, color='red', blankLine=False)
        else:
            self.showMsg(f"This 'drop' has a zero probability of being an artifact of noise.",
                         bold=True, color='green', blankLine=False)

        self.showMsg(f">>>> probability > 0.0000 indicates the 'drop' may be spurious (a noise artifact)."
                     f" Consult with an IOTA Regional Coordinator.", color='blue', blankLine=False)
        self.showMsg(f">>>> probability = 0.0000 indicates the 'drop' is unlikely to be a noise artifact, but"
                     f" does not prove that the 'drop' is due to an occultation", color='blue', blankLine=False)
        self.showMsg(f">>>> Consider 'drop' shape, timing, mag drop, duration and other positive observer"
                     f" chords before reporting the 'drop' as a positive.", color='blue')

        # self.showMsg("All timestamps are treated as being start-of-exposure times.",
        #              color='red', bold=True)
        self.showMsg("All times are calculated/reported based on the assumption that timestamps are "
                     "start-of-exposure times.",
                     color='blue', bold=True)
        self.showMsg("It is critical that you make appropriate time adjustments when timestamps are "
                     "NOT start-of-exposure times.", color='red', bold=True, blankLine=False)
        self.showMsg("If you use the North American Excel Spreadsheet report, all times will be properly corrected",
                     color='red', bold=True, blankLine=False)
        self.showMsg("for camera delay and reported start-of-exposure times.",
                     color='red', bold=True, blankLine=False)
        self.showMsg("For other users worldwide, use the appropriate corrections documented in the North American "
                     "Spreadsheet report form - use",
                     color='red', bold=True, blankLine=False)
        self.showMsg("the documentation shown on the Corrections Tables tab.",
                     color='red', bold=True)

        if not self.timesAreValid:
            self.showMsg("Times are invalid due to corrupted timestamps!",
                         color='red', bold=True)

        # if self.choleskyFailed and self.ne3NotInUseRadioButton.isChecked():
        if self.choleskyFailed:
            self.showMsg('Cholesky decomposition failed during error bar '
                         'calculations. '
                         'Noise has therefore been treated as being '
                         'uncorrelated.',
                         bold=True, color='red')

        self.xlsxDict['Comment'] = f'mag drop = {self.magDropString(self.B, self.A, 2)}'
        self.showMsg(f'magDrop: {self.magDropString(self.B, self.A, 2)}')

        # noinspection PyStringFormat
        self.showMsg('snr: %0.2f' % self.snrB)

        self.doDtimeReport()
        self.doRtimeReport()
        self.doDurTimeReport()

        self.showMsg(
            'Enter D and R error bars for each containment interval in Excel spreadsheet without + or - sign (assumed to be +/-)')

        self.showMsg('=========== end Summary report for Excel file =====================')

        self.showMsg("Solution 'envelope' in the main plot drawn using 0.95 containment interval error bars")

        self.showHelp(self.helpLabelForFalsePositive)

    def doDframeReport(self):
        if self.eventType == 'DandR' or self.eventType == 'Donly':
            D, _ = self.solution
            entryNum = int(D)
            frameNum = float(self.yFrame[entryNum])

            Dframe = (D - int(D)) * self.framesPerEntry() + frameNum
            self.showMsg('D frame number: {0:0.2f}'.format(Dframe), blankLine=False)
            errBar = max(abs(self.deltaDlo68), abs(self.deltaDhi68)) * self.framesPerEntry()
            self.showMsg('D: 0.6800 containment intervals:  {{+/- {0:0.2f}}} (readings)'.format(errBar),
                         blankLine=False)
            errBar = max(abs(self.deltaDlo95), abs(self.deltaDhi95)) * self.framesPerEntry()
            self.showMsg('D: 0.9500 containment intervals:  {{+/- {0:0.2f}}} (readings)'.format(errBar),
                         blankLine=False)
            errBar = max(abs(self.deltaDlo99), abs(self.deltaDhi99)) * self.framesPerEntry()
            self.showMsg('D: 0.9973 containment intervals:  {{+/- {0:0.2f}}} (readings)'.format(errBar))

    def framesPerEntry(self):
        # Normally, there is 1 frame per reading (entry), but if the source file was recorded
        # in field mode, there is only 0.5 frames per reading (entry).  Here we make the correction.
        if self.fieldMode:
            return self.blockSize / 2
        else:
            return self.blockSize

    def doRframeReport(self):
        if self.eventType == 'DandR' or self.eventType == 'Ronly':
            _, R = self.solution
            entryNum = int(R)
            frameNum = float(self.yFrame[entryNum])

            Rframe = (R - int(R)) * self.framesPerEntry() + frameNum
            self.showMsg('R frame number: {0:0.2f}'.format(Rframe), blankLine=False)
            errBar = max(abs(self.deltaRlo68), abs(self.deltaRhi68)) * self.framesPerEntry()
            self.showMsg('R: 0.6800 containment intervals:  {{+/- {0:0.2f}}} (readings)'.format(errBar),
                         blankLine=False)
            errBar = max(abs(self.deltaRlo95), abs(self.deltaRhi95)) * self.framesPerEntry()
            self.showMsg('R: 0.9500 containment intervals:  {{+/- {0:0.2f}}} (readings)'.format(errBar),
                         blankLine=False)
            errBar = max(abs(self.deltaRlo99), abs(self.deltaRhi99)) * self.framesPerEntry()
            self.showMsg('R: 0.9973 containment intervals:  {{+/- {0:0.2f}}} (readings)'.format(errBar))

    def doDurFrameReport(self):
        if self.eventType == 'DandR':
            D, R = self.solution
            self.showMsg('Duration (R - D): {0:0.4f} readings'.format((R - D) * self.framesPerEntry()), blankLine=False)
            errBar = ((self.deltaDurhi68 - self.deltaDurlo68) / 2) * self.framesPerEntry()
            self.showMsg('Duration: 0.6800 containment intervals:  {{+/- {0:0.4f}}} (readings)'.format(errBar),
                         blankLine=False)
            errBar = ((self.deltaDurhi95 - self.deltaDurlo95) / 2) * self.framesPerEntry()
            self.showMsg('Duration: 0.9500 containment intervals:  {{+/- {0:0.4f}}} (readings)'.format(errBar),
                         blankLine=False)
            errBar = ((self.deltaDurhi99 - self.deltaDurlo99) / 2) * self.framesPerEntry()
            self.showMsg('Duration: 0.9973 containment intervals:  {{+/- {0:0.4f}}} (readings)'.format(errBar))

    def doDtimeReport(self):
        if self.eventType == 'DandR' or self.eventType == 'Donly':
            D, _ = self.solution
            ts = self.yTimes[int(D)]

            time = convertTimeStringToTime(ts)
            adjTime = time + (D - int(D)) * self.timeDelta
            self.Dtime = adjTime  # This is needed for the duration report (assumed to follow!!!)
            ts = convertTimeToTimeString(adjTime)

            tsParts = ts[1:-1].split(':')
            self.xlsxDict['Dhour'] = tsParts[0]
            self.xlsxDict['Dmin'] = tsParts[1]
            self.xlsxDict['Dsec'] = tsParts[2]

            self.showMsg('D time: %s' % ts, blankLine=False)

            plusD, minusD = self.computeErrorBarPair(deltaHi=self.deltaDhi68, deltaLo=self.deltaDlo68, edge='D')
            errBar = max(abs(plusD), abs(minusD)) * self.timeDelta
            self.xlsxDict['Derr68'] = errBar
            self.showMsg('D: 0.6800 containment intervals:  {{+/- {0:0.4f}}} seconds'.format(errBar), blankLine=False)

            plusD, minusD = self.computeErrorBarPair(deltaHi=self.deltaDhi95, deltaLo=self.deltaDlo95, edge='D')
            errBar = max(abs(plusD), abs(minusD)) * self.timeDelta
            self.xlsxDict['Derr95'] = errBar
            self.showMsg('D: 0.9500 containment intervals:  {{+/- {0:0.4f}}} seconds'.format(errBar), blankLine=False)

            plusD, minusD = self.computeErrorBarPair(deltaHi=self.deltaDhi99, deltaLo=self.deltaDlo99, edge='D')
            errBar = max(abs(plusD), abs(minusD)) * self.timeDelta
            self.xlsxDict['Derr99'] = errBar
            self.showMsg('D: 0.9973 containment intervals:  {{+/- {0:0.4f}}} seconds'.format(errBar))

    def doRtimeReport(self):
        if self.eventType == 'DandR' or self.eventType == 'Ronly':
            _, R = self.solution
            ts = self.yTimes[int(R)]

            time = convertTimeStringToTime(ts)
            adjTime = time + (R - int(R)) * self.timeDelta
            self.Rtime = adjTime  # This is needed for the duration report (assumed to follow!!!)
            ts = convertTimeToTimeString(adjTime)

            tsParts = ts[1:-1].split(':')
            self.xlsxDict['Rhour'] = tsParts[0]
            self.xlsxDict['Rmin'] = tsParts[1]
            self.xlsxDict['Rsec'] = tsParts[2]

            self.showMsg('R time: %s' % ts, blankLine=False)

            plusR, minusR = self.computeErrorBarPair(deltaHi=self.deltaDhi68, deltaLo=self.deltaDlo68, edge='R')
            errBar = max(abs(plusR), abs(minusR)) * self.timeDelta
            self.xlsxDict['Rerr68'] = errBar
            self.showMsg('R: 0.6800 containment intervals:  {{+/- {0:0.4f}}} seconds'.format(errBar), blankLine=False)

            plusR, minusR = self.computeErrorBarPair(deltaHi=self.deltaDhi95, deltaLo=self.deltaDlo95, edge='R')
            errBar = max(abs(plusR), abs(minusR)) * self.timeDelta
            self.xlsxDict['Rerr95'] = errBar
            self.showMsg('R: 0.9500 containment intervals:  {{+/- {0:0.4f}}} seconds'.format(errBar), blankLine=False)

            plusR, minusR = self.computeErrorBarPair(deltaHi=self.deltaDhi99, deltaLo=self.deltaDlo99, edge='R')
            errBar = max(abs(plusR), abs(minusR)) * self.timeDelta
            self.xlsxDict['Rerr99'] = errBar
            self.showMsg('R: 0.9973 containment intervals:  {{+/- {0:0.4f}}} seconds'.format(errBar))

    def doDurTimeReport(self):
        if self.eventType == 'DandR':
            dur = self.Rtime - self.Dtime
            if dur < 0:  # We have bracketed midnight
                dur = dur + 3600 * 24  # Add seconds in a day
            self.showMsg('Duration (R - D): {0:0.4f} seconds'.format(dur), blankLine=False)
            errBar = ((self.deltaDurhi68 - self.deltaDurlo68) / 2) * self.timeDelta
            self.showMsg('Duration: 0.6800 containment intervals:  {{+/- {0:0.4f}}} seconds'.format(errBar),
                         blankLine=False)
            errBar = ((self.deltaDurhi95 - self.deltaDurlo95) / 2) * self.timeDelta
            self.showMsg('Duration: 0.9500 containment intervals:  {{+/- {0:0.4f}}} seconds'.format(errBar),
                         blankLine=False)
            errBar = ((self.deltaDurhi99 - self.deltaDurlo99) / 2) * self.timeDelta
            self.showMsg('Duration: 0.9973 containment intervals:  {{+/- {0:0.4f}}} seconds'.format(errBar))

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
            self.timesAreValid = False
            self.showMsg('Timestamps are corrupted in a manner that caused a '
                         'timeDelta of '
                         '0.0 to be estimated!', color='red', bold=True)
            self.showInfo('Timestamps are corrupted in a manner that caused a '
                          'timeDelta of '
                          '0.0 to be estimated!')
            return

        numEnclosedReadings = int(round((rTime - dTime) / self.timeDelta))
        self.showMsg(
            'From timestamps at D and R, calculated %d reading blocks.  From reading blocks, calculated %d blocks.' %
            (numEnclosedReadings, intR - intD))
        if numEnclosedReadings == intR - intD:
            self.showMsg('Timestamps appear valid @ D and R')
            self.timesAreValid = True
        else:
            self.timesAreValid = False
            self.showMsg('! There is something wrong with timestamps at D '
                         'and/or R or frames have been dropped !', bold=True,
                         color='red')

    def calculateVarianceFromHistogram(self, y, x):
        peakIndex = np.where(y == np.max(y))
        # print(x[0],x[-1])
        # print(peakIndex)
        self.showMsg(f'peakIndex: {peakIndex[0][0]} xPeak: {x[peakIndex[0][0]]:0.2f}')

    def computeErrorBars(self):

        if self.penumbralFitCheckBox.isChecked():
            self.finalReportPenumbral()
            return

        if self.sigmaB == 0.0:
            self.sigmaB = MIN_SIGMA

        if self.sigmaA == 0.0:
            self.sigmaA = MIN_SIGMA

        self.snrB = (self.B - self.A) / self.sigmaB
        self.snrA = (self.B - self.A) / self.sigmaA
        snr = max(self.snrB, 0.2)  # A more reliable number

        D = int(round(80 / snr ** 2 + 0.5))

        D = max(10, D)
        if self.corCoefs.size > 1:
            D = round(1.5 * D)
        numPts = 2 * (D - 1) + 1
        posCoefs = []
        for entry in self.corCoefs:
            if entry < acfCoefThreshold:
                break
            posCoefs.append(entry)

        # noinspection PyTypeChecker
        distGen = edgeDistributionGenerator(
            ntrials=100000, numPts=numPts, D=D, acfcoeffs=posCoefs,
            B=self.B, A=self.A, sigmaB=self.sigmaB, sigmaA=self.sigmaA)

        dist = None
        self.choleskyFailed = False
        for dist in distGen:
            if type(dist) == float:
                if dist == -1.0:
                    self.choleskyFailed = True
                    # if self.ne3NotInUseRadioButton.isChecked():
                    if True:
                        self.showInfo(
                            'The Cholesky-Decomposition routine has failed.  This may be because the light curve ' +
                            'required block integration and you failed to do either a manual block integration or to accept the '
                            'PyOTE estimated block integration.  Please '
                            'examine the light curve for that possibility.' +
                            '\nWe treat this situation as though there is no '
                            'correlation in the noise.')
                        self.showMsg('Cholesky decomposition has failed.  '
                                     'Proceeding by '
                                     'treating noise as being uncorrelated.',
                                     bold=True, color='red')
                self.progressBar.setValue(int(dist * 100))
                QtWidgets.QApplication.processEvents()
                if self.cancelRequested:
                    self.cancelRequested = False
                    self.showMsg('Error bar calculation was cancelled')
                    self.progressBar.setValue(0)
                    return
            else:
                self.progressBar.setValue(0)

        # pickle.dump(dist, open('sample-dist.p', 'wb'))

        y, x = np.histogram(dist, bins=1000)
        self.loDbar68, _, self.hiDbar68, self.deltaDlo68, self.deltaDhi68 = ciBars(dist=dist, ci=0.6827, D=D)
        self.loDbar95, _, self.hiDbar95, self.deltaDlo95, self.deltaDhi95 = ciBars(dist=dist, ci=0.95, D=D)
        self.loDbar99, _, self.hiDbar99, self.deltaDlo99, self.deltaDhi99 = ciBars(dist=dist, ci=0.9973, D=D)

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
        self.loDurbar68, _, self.hiDurbar68, self.deltaDurlo68, self.deltaDurhi68 = \
            ciBars(dist=durDist, ci=0.6827, D=0.0)
        self.loDurbar95, _, self.hiDurbar95, self.deltaDurlo95, self.deltaDurhi95 = \
            ciBars(dist=durDist, ci=0.95, D=0.0)
        self.loDurbar99, _, self.hiDurbar99, self.deltaDurlo99, self.deltaDurhi99 = \
            ciBars(dist=durDist, ci=0.9973, D=0.0)

        pg.setConfigOptions(antialias=True)
        pen = pg.mkPen((0, 0, 0), width=2)

        # Get rid of a previous errBarWin that may have been closed (but not properly disposed of) by the user.
        if self.errBarWin is not None:
            self.errBarWin.close()

        self.errBarWin = pg.GraphicsWindow(
            title='Solution distributions with containment intervals marked --- false positive distribution')
        self.errBarWin.resize(1200, 1000)
        layout = QtWidgets.QGridLayout()
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

        pw3, false_positive, false_probability = self.doFalsePositiveReport(posCoefs)
        self.falsePositivePlotItem = pw3.getPlotItem()

        layout.addWidget(pw, 0, 0)
        layout.addWidget(pw2, 0, 1)
        layout.addWidget(pw3, 1, 0, 1, 2)  # (pw3, row_start, col_start, n_rows_to_span, n_cols_to_span)

        pw.plot(x - D, y, stepMode=True, fillLevel=0, brush=(0, 0, 255, 150))
        pw.addLine(y=0, z=-10, pen=[0, 0, 255])
        pw.addLine(x=0, z=+10, pen=[255, 0, 0])

        yp = max(y) * 0.75
        x1 = self.loDbar68 - D
        pw.plot(x=[x1, x1], y=[0, yp], pen=pen)

        x2 = self.hiDbar68 - D
        pw.plot(x=[x2, x2], y=[0, yp], pen=pen)

        pw.addLegend()
        legend68 = '[%0.2f,%0.2f] @ 0.6827' % (x1, x2)
        pw.plot(name=legend68)

        self.showMsg("Error bar report based on 100,000 simulations (units are readings)...")

        self.showMsg('loDbar   @ .68 ci: %8.4f' % (x1 * self.framesPerEntry()), blankLine=False)
        self.showMsg('hiDbar   @ .68 ci: %8.4f' % (x2 * self.framesPerEntry()), blankLine=False)

        yp = max(y) * 0.25
        x1 = self.loDbar95 - D
        pw.plot(x=[x1, x1], y=[0, yp], pen=pen)
        x2 = self.hiDbar95 - D
        pw.plot(x=[x2, x2], y=[0, yp], pen=pen)

        self.showMsg('loDbar   @ .95 ci: %8.4f' % (x1 * self.framesPerEntry()), blankLine=False)
        self.showMsg('hiDbar   @ .95 ci: %8.4f' % (x2 * self.framesPerEntry()), blankLine=False)

        legend95 = '[%0.2f,%0.2f] @ 0.95' % (x1, x2)
        pw.plot(name=legend95)

        yp = max(y) * 0.15
        x1 = self.loDbar99 - D
        pw.plot(x=[x1, x1], y=[0, yp], pen=pen)
        x2 = self.hiDbar99 - D
        pw.plot(x=[x2, x2], y=[0, yp], pen=pen)

        self.showMsg('loDbar   @ .9973 ci: %8.4f' % (x1 * self.framesPerEntry()), blankLine=False)
        self.showMsg('hiDbar   @ .9973 ci: %8.4f' % (x2 * self.framesPerEntry()), blankLine=True)

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

        self.showMsg('loDurBar @ .68 ci: %8.4f' % (x1 * self.framesPerEntry()), blankLine=False)
        self.showMsg('hiDurBar @ .68 ci: %8.4f' % (x2 * self.framesPerEntry()), blankLine=False)

        yp = max(ydur) * 0.25
        x1 = self.loDurbar95
        pw2.plot(x=[x1, x1], y=[0, yp], pen=pen)
        x2 = self.hiDurbar95
        pw2.plot(x=[x2, x2], y=[0, yp], pen=pen)

        self.showMsg('loDurBar @ .95 ci: %8.4f' % (x1 * self.framesPerEntry()), blankLine=False)
        self.showMsg('hiDurBar @ .95 ci: %8.4f' % (x2 * self.framesPerEntry()), blankLine=False)

        legend95 = '[%0.2f,%0.2f] @ 0.95' % (x1, x2)
        pw2.plot(name=legend95)

        yp = max(ydur) * 0.15
        x1 = self.loDurbar99
        pw2.plot(x=[x1, x1], y=[0, yp], pen=pen)
        x2 = self.hiDurbar99
        pw2.plot(x=[x2, x2], y=[0, yp], pen=pen)

        self.showMsg('loDurBar @ .9973 ci: %8.4f' % (x1 * self.framesPerEntry()), blankLine=False)
        self.showMsg('hiDurBar @ .9973 ci: %8.4f' % (x2 * self.framesPerEntry()), blankLine=True)

        legend99 = '[%0.2f,%0.2f] @ 0.9973' % (x1, x2)
        pw2.plot(name=legend99)

        pw2.hideAxis('left')

        self.writeBarPlots.setEnabled(True)

        if self.timestampListIsEmpty(self.yTimes):
            self.showMsg('Cannot produce final report because timestamps are missing.', bold=True, color='red')
        else:
            self.finalReport(false_positive, false_probability)
            self.fillExcelReportButton.setEnabled(True)

        self.newRedrawMainPlot()  # To add envelope to solution

    def calcDetectability(self):
        if self.timeDelta == 0:
            self.showInfo(f'Cannot use the detectabilty tool on a light curve without timestamps.')
            return

        if not self.userDeterminedBaselineStats:
            self.showInfo(f'Baseline statistics have not been extracted yet.')
            return

        posCoefs = self.newCorCoefs

        durText = self.observationDurEdit.text()
        if durText == '':
            self.showInfo(f'You need to fill in an observation duration (in seconds) for this operation.')
            return
        try:
            obs_duration_secs = float(durText)
            obs_duration = int(np.ceil(obs_duration_secs / self.timeDelta))
            if obs_duration < 1:
                self.showInfo(f'The obs duration: {obs_duration} is too small to use.')
                return
        except ValueError:
            self.showInfo(f'{durText} is invalid as a duration in seconds value')
            return

        durText = self.eventDurationEdit.text()
        if durText == '':
            self.showInfo(f'You need to fill in a duration (in seconds) for this operation.')
            return
        try:
            event_duration_secs = float(durText)

            # 4.6.2 change
            event_duration = int(np.ceil(event_duration_secs / self.timeDelta))  # In readings
            # event_duration = round((event_duration_secs / self.timeDelta))  # In readings

            if event_duration < 1:
                self.showInfo(f'The event duration: {event_duration} is too small to use.')
                return
            if obs_duration <= event_duration:
                self.showInfo(f'The event duration: {event_duration} '
                              f'is too large for an observation with {obs_duration} points.')
                return

        except ValueError:
            self.showInfo(f'{durText} is invalid as a duration in seconds value')
            return

        magDropText = self.detectabilityMagDropEdit.text()
        if magDropText == '':
            self.showInfo(f'You need to fill in a magDrop for this operation.')
            return
        try:
            event_magDrop = float(magDropText)
            if event_magDrop < 0:
                self.showInfo(f'magDrop must be a positive value.')
                return
        except ValueError:
            self.showInfo(f'{magDropText} is invalid as a magDrop value')
            return

        durStep = 0.0
        durStepText = self.durStepEdit.text()
        if not durStepText == '':
            try:
                durStep = float(durStepText)
                if durStep < 0:
                    self.showInfo(f'durStep must be a positive number.')
                    return
            except ValueError:
                self.showInfo(f'{durStepText} is invalid as a durSep value')
                return

        sigma = self.sigmaB
        # Use given magDrop to calculate observed drop
        ratio = 10 ** (event_magDrop / 2.5)
        self.A = self.B / ratio
        observed_drop = self.B - self.A
        num_trials = 50_000

        self.minDetectableDurationRdgs = None
        self.minDetectableDurationSecs = None

        while True:
            i = 0
            for i, checkBox in enumerate(self.targetCheckBoxes):
                if checkBox.isChecked():
                    break
            self.showMsg(f'Using lightcurve: {self.lightcurveTitles[i].text()} ...',
                         blankLine=False, alternateLogFile=self.detectabilityLogFile)
            self.showMsg(f'... processing detectability of magDrop: {event_magDrop:0.2f} '
                         f'dur(secs): {event_duration_secs:0.3f} event', alternateLogFile=self.detectabilityLogFile,
                         blankLine=False)
            QtWidgets.QApplication.processEvents()

            drops = compute_drops(event_duration=event_duration, observation_size=obs_duration,
                                  noise_sigma=sigma, corr_array=np.array(posCoefs), num_trials=num_trials)

            title = (f'Distribution of drops found in correlated noise --- '
                     f'Event duration: {event_duration} readings ------ '
                     f'Event duration: {event_duration_secs:0.2f} seconds')

            pw = PlotWidget(viewBox=CustomViewBox(border=(0, 0, 0)),
                            enableMenu=False,
                            title=title,
                            labels={'bottom': 'drop size (ADU)',
                                    'left': 'relative number of times noise produced drop'})
            pw.hideButtons()

            y, x = np.histogram(drops, bins=50, density=True, range=(0, np.max(drops)))

            blackDrop = np.max(x)
            redDrop = observed_drop
            # 4.6.2 change
            if event_duration_secs < self.timeDelta:
                redDrop = observed_drop * (event_duration_secs / self.timeDelta)
                self.showMsg(f'... drop reduced from {observed_drop:0.2f} to {redDrop:0.2f} due to sub-frame exposure',
                             alternateLogFile=self.detectabilityLogFile)
            else:
                self.showMsg('', alternateLogFile=self.detectabilityLogFile, blankLine=False)
            # end 4.6.2 change
            redMinusBlack = redDrop - blackDrop

            pw.plot(x, y, stepMode=True, fillLevel=0, brush=(0, 0, 255, 150))
            pw.plot(x=[redDrop, redDrop], y=[0, 1.5 * np.max(y)], pen=pg.mkPen([255, 0, 0], width=4))
            pw.plot(x=[np.max(x), np.max(x)], y=[0, 0.25 * np.max(y)], pen=pg.mkPen([0, 0, 0], width=4))

            pw.addLegend()
            pw.plot(
                name=f'red line @ {redDrop:0.2f} = location of a {event_magDrop} magDrop event in histogram of all detected events due to noise')
            pw.plot(name=f'black line @ {blackDrop:0.2f} = max drop found in {num_trials} trials with correlated noise')
            pw.plot(name=f'B: {self.B:0.2f}  A: {self.A:0.2f} (calculated from expected magDrop of event)')
            pw.plot(name=f'magDrop: {magDropText}')

            if redMinusBlack > 0:
                pw.plot(name=f'red - black = {redMinusBlack:0.2f}  An event of this duration and magDrop is detectable')
                self.minDetectableDurationRdgs = event_duration  # This is in 'readings'
                self.minDetectableDurationSecs = event_duration_secs  # This is in seconds
            else:
                pw.plot(name=f'red - black = {redMinusBlack:0.2f}  Detection of this event is unlikely')

            self.detectabilityWin = pg.GraphicsWindow(title='Detectability test: False positive distribution')
            self.detectabilityWin.resize(1700, 700)
            layout = QtWidgets.QGridLayout()
            self.detectabilityWin.setLayout(layout)
            layout.addWidget(pw, 0, 0)

            pw.getPlotItem().setFixedHeight(700)
            pw.getPlotItem().setFixedWidth(1700)

            lightCurveDir = os.path.dirname(self.filename)  # This gets the folder where the light-curve.csv is located
            detectibiltyPlotPath = lightCurveDir + '/DetectabilityPlots/'
            if not os.path.exists(detectibiltyPlotPath):
                os.mkdir(detectibiltyPlotPath)
            targetFile = detectibiltyPlotPath + f'plot.detectability-dur{event_duration_secs:0.3f}-magDrop{event_magDrop:0.2f}.PYOTE.png'

            exporter = FixedImageExporter(pw.getPlotItem())
            exporter.makeWidthHeightInts()

            if durStep == 0.0:
                # Always write plot for single detectability requests
                exporter.export(targetFile)
                QtWidgets.QApplication.processEvents()

            if redMinusBlack < 0 and not durStep == 0.0:  # A failed detect during step down
                # Only write the final plot for "find minimum duration detectability" requests
                exporter.export(targetFile)
                QtWidgets.QApplication.processEvents()
                self.showMsg(
                    f'Undetectability reached at magDrop: {event_magDrop:0.2f}  duration=: {event_duration_secs:0.3f}',
                    alternateLogFile=self.detectabilityLogFile)
                break

            if durStep == 0.0:
                break
            else:
                event_duration_secs -= durStep  # Try next smaller duration

                # 4.6.2 change
                event_duration = int(np.ceil(event_duration_secs / self.timeDelta))
                # event_duration = round((event_duration_secs / self.timeDelta))  # readings

                if event_duration < 1:
                    # 4.6.1 change
                    event_duration_secs += durStep  # Last successful duration
                    event_duration = int(np.ceil(event_duration_secs / self.timeDelta))
                    # end 4.6.1 change

                    break

        if self.minDetectableDurationRdgs is None:
            self.showMsg(
                f'Undetectable event @ magDrop: {event_magDrop:0.2f}  duration: {event_duration_secs:0.3f}',
                color='red', bold=True
            )
        else:
            self.showMsg(
                f'An event of duration {self.minDetectableDurationSecs:0.3f} seconds with magDrop: {event_magDrop}'
                f' is likely detectable.', color='red', bold=True
            )

            # Show an example light-curve with the 'event' included.
            obs = noise_gen_jit(obs_duration, self.sigmaB)
            obs = simple_convolve(obs, np.array(self.newCorCoefs))
            title = (f'Example light curve at the minimum detectable duration found ---  '
                     f'Event duration: {event_duration} readings ------ '
                     f'Event duration:{event_duration_secs + durStep:0.3f} seconds ------ '
                     f'magDrop: {magDropText}'
                     )
            pw = PlotWidget(viewBox=CustomViewBox(border=(0, 0, 0)),
                            enableMenu=False,
                            title=title,
                            labels={'bottom': 'reading number', 'left': 'intensity'})
            pw.hideButtons()
            pw.getPlotItem().showAxis('bottom', True)
            pw.getPlotItem().showAxis('left', True)
            pw.showGrid(y=True, alpha=.5)
            pw.getPlotItem().setFixedHeight(700)
            pw.getPlotItem().setFixedWidth(1700)

            self.detectabilityWin = pg.GraphicsWindow(
                title='Detectability test: sample light curve with minimum duration event')
            self.detectabilityWin.resize(1700, 700)
            layout = QtWidgets.QGridLayout()
            self.detectabilityWin.setLayout(layout)
            layout.addWidget(pw, 0, 0)

            obs += self.B
            center = int(obs_duration / 2)
            eventStart = center - int(event_duration / 2)

            # 4.6.1 change
            # obs[eventStart:eventStart+event_duration+1] -= observed_drop
            obs[eventStart:eventStart + event_duration] -= redDrop

            pw.plot(obs)
            pw.plot(obs, pen=None, symbol='o', symbolBrush=(0, 0, 255), symbolSize=6)
            left_marker = pg.InfiniteLine(pos=eventStart, pen='r')
            pw.addItem(left_marker)
            right_marker = pg.InfiniteLine(pos=eventStart + event_duration, pen='g')
            pw.addItem(right_marker)

            if self.writeExampleLightcurveCheckBox.isChecked():
                # Write the example lightcurve (obs) to a csv file
                self.writeExampleLightcurveToFile(obs, self.timeDelta)

        return

    def doFalsePositiveReport(self, posCoefs):
        d, r = self.solution
        if self.eventType == 'Donly':
            event_duration = self.right - int(np.trunc(d))
        elif self.eventType == 'Ronly':
            event_duration = int(np.ceil(r)) - self.left
        else:
            event_duration = int(np.ceil(r - d))

        observation_size = self.right - self.left + 1
        sigma = max(self.sigmaA, self.sigmaB)
        observed_drop = self.B - self.A
        num_trials = 50_000

        return self.falsePositiveReport(event_duration, num_trials, observation_size, observed_drop, posCoefs, sigma)

    @staticmethod
    def falsePositiveReport(event_duration, num_trials, observation_size, observed_drop, posCoefs, sigma):
        drops = compute_drops(event_duration=event_duration, observation_size=observation_size,
                              noise_sigma=sigma, corr_array=np.array(posCoefs), num_trials=num_trials)
        pw = PlotWidget(viewBox=CustomViewBox(border=(0, 0, 0)),
                        enableMenu=False,
                        title=f'Distribution of drops found in correlated noise for event duration: {event_duration}',
                        labels={'bottom': 'drop size', 'left': 'number of times noise produced drop'})
        pw.hideButtons()
        y, x = np.histogram(drops, bins=50)
        y[0] = y[1] / 2.0
        pw.plot(x, y, stepMode=True, fillLevel=0, brush=(0, 0, 255, 150))
        pw.plot(x=[observed_drop, observed_drop], y=[0, 1.5 * np.max(y)], pen=pg.mkPen([255, 0, 0], width=6))
        pw.plot(x=[np.max(x), np.max(x)], y=[0, 0.25 * np.max(y)], pen=pg.mkPen([0, 0, 0], width=6))
        pw.addLegend()
        pw.plot(name='red line: the drop (B - A) extracted from lightcurve')
        pw.plot(name=f'black line: max drop found in {num_trials} trials against pure noise')
        pw.plot(name='If the red line is to the right of the black line, false positive prob = 0')
        sorted_drops = np.sort(drops)
        index_of_observed_drop_inside_sorted_drops = None
        for i, value in enumerate(sorted_drops):
            if value >= observed_drop:
                index_of_observed_drop_inside_sorted_drops = i
                break
        if index_of_observed_drop_inside_sorted_drops is None:
            false_probability = 0.0
            false_positive = False
        else:
            false_probability = 1.0 - index_of_observed_drop_inside_sorted_drops / drops.size
            false_positive = True
        return pw, false_positive, false_probability

    def displaySolution(self, subframe=True):
        D, R = self.solution

        # D and R are floats and may be fractional because of sub-frame timing.
        # We have to remove the effects of sub-frame timing to calulate the D
        # and R transition points as integers.

        solMsg2 = ''
        frameConv = float(self.yFrame[0])
        DinFrameUnits = None
        RinFrameUnits = None
        if D and R:
            Dtransition = trunc(floor(self.solution[0]))
            Rtransition = trunc(floor(self.solution[1]))

            DinFrameUnits = Dtransition * self.framesPerEntry() + frameConv
            RinFrameUnits = Rtransition * self.framesPerEntry() + frameConv

            if subframe:
                solMsg = ('D: %d  R: %d  D(subframe): %0.4f  R(subframe): %0.4f' %
                          (Dtransition, Rtransition, D, R))

                solMsg2 = ('D: %d  R: %d  D(subframe): %0.4f  R(subframe): '
                           '%0.4f' %
                           (DinFrameUnits,
                            RinFrameUnits,
                            D * self.framesPerEntry() + frameConv, R * self.framesPerEntry() + frameConv))
            else:
                solMsg = ('D: %d  R: %d' % (D, R))
            self.showMsg('in entryNum units: ' + solMsg)
            if solMsg2:
                self.showMsg('in frameNum units: ' + solMsg2, bold=True)
        elif D:
            Dtransition = trunc(floor(self.solution[0]))
            DinFrameUnits = Dtransition * self.framesPerEntry() + frameConv

            if subframe:
                solMsg = ('D: %d  D(subframe): %0.4f' % (Dtransition, D))
                solMsg2 = ('D: %d  D(subframe): %0.4f' %
                           (DinFrameUnits, D * self.framesPerEntry() + frameConv))
            else:
                solMsg = ('D: %d' % D)
            self.showMsg('in entryNum units: ' + solMsg)
            if solMsg2:
                self.showMsg('in frameNum units: ' + solMsg2, bold=True)
        else:
            Rtransition = trunc(floor(self.solution[1]))
            RinFrameUnits = Rtransition * self.framesPerEntry() + frameConv

            if subframe:
                solMsg = ('R: %d  R(subframe): %0.4f' % (Rtransition, R))
                solMsg2 = ('R: %d  R(subframe): %0.4f' %
                           (RinFrameUnits, R * self.framesPerEntry() + frameConv))
            else:
                solMsg = ('R: %d' % R)

            self.showMsg('in entryNum units: ' + solMsg)
            if solMsg2:
                self.showMsg('in frameNum units: ' + solMsg2, bold=True)

        # This function is called twice: once without a subframe calculation and then again with
        # subframe calculations enabled.  We only want to display D and/or R frames at the end
        # of the second pass
        if subframe:
            if self.showOCRcheckFramesCheckBox.isChecked():
                if self.pathToVideo:
                    if DinFrameUnits:
                        self.showAnnotatedFrame(int(DinFrameUnits), "D edge:")
                    if RinFrameUnits:
                        self.showAnnotatedFrame(int(RinFrameUnits), 'R edge:')
                    return True
        return False

    def extract_noise_parameters_from_iterative_solution(self):

        D, R = self.solution
        # D and R are floats and may be fractional because of sub-frame timing.
        # Here we remove the effects of sub-frame timing to calulate the D and
        # R transition points as integers.
        if D:
            D = trunc(floor(D))
        if R:
            R = trunc(floor(R))

        if not self.userDeterminedBaselineStats:
            self.corCoefs = []

        if D and R:
            self.sigmaA = None
            # self.corCoefs = []

            self.processBaselineNoiseFromIterativeSolution(self.left, D - 1)

            self.processBaselineNoiseFromIterativeSolution(R, self.right)

            self.processEventNoiseFromIterativeSolution(D, R - 1)

            # Try to warn user about the possible need for block integration by testing the lag 1
            # and lag 2 correlation coefficients.  The tests are just guesses on my part, so only
            # warnings are given.  Later, the Cholesky-Decomposition may fail because block integration
            # was really needed.  That is a fatal error but is trapped and the user alerted to the problem

            if len(self.corCoefs) > 1:
                if self.corCoefs[1] >= 0.7 and self.ne3NotInUseRadioButton.isChecked():
                    self.showInfo(
                        'The auto-correlation coefficient at lag 1 is suspiciously large. '
                        'This may be because the light curve needs some degree of block integration. '
                        'Failure to do a needed block integration allows point-to-point correlations caused by '
                        'the camera integration to artificially induce non-physical correlated noise.')
                elif len(self.corCoefs) > 2:
                    if self.corCoefs[2] >= 0.3 and self.ne3NotInUseRadioButton.isChecked():
                        self.showInfo(
                            'The auto-correlation coefficient at lag 2 is suspiciously large. '
                            'This may be because the light curve needs some degree of block integration. '
                            'Failure to do a needed block integration allows point-to-point correlations caused by '
                            'the camera integration to artificially induce non-physical correlated noise.')

            if self.sigmaA is None:
                self.sigmaA = self.sigmaB
        elif D:
            self.sigmaA = None

            self.processBaselineNoiseFromIterativeSolution(self.left, D - 1)

            self.processEventNoiseFromIterativeSolution(D, self.right)
            if self.sigmaA is None:
                self.sigmaA = self.sigmaB
        else:  # R only
            self.sigmaA = None

            self.processBaselineNoiseFromIterativeSolution(R, self.right)

            self.processEventNoiseFromIterativeSolution(self.left, R - 1)
            if self.sigmaA is None:
                self.sigmaA = self.sigmaB

        self.prettyPrintCorCoefs()

        return

    def try_to_get_solution(self):
        self.solution = None
        self.newRedrawMainPlot()
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
                if not self.userDeterminedBaselineStats:
                    self.B = item[1]
                if self.userTrimInEffect:
                    self.B = item[1]
                if not self.userDeterminedEventStats:
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

    def extractBaselineAndEventData(self):
        if self.dLimits and self.rLimits:
            left_baseline_pts = self.yValues[self.left:self.dLimits[0]]
            right_baseline_pts = self.yValues[self.rLimits[1] + 1:self.right + 1]
            baseline_pts = np.concatenate((left_baseline_pts, right_baseline_pts))
            event_pts = self.yValues[self.dLimits[1] + 1:self.rLimits[0]]
        elif self.dLimits:
            baseline_pts = self.yValues[self.left:self.dLimits[0]]
            event_pts = self.yValues[self.dLimits[1] + 1:self.right + 1]
        elif self.rLimits:
            baseline_pts = self.yValues[self.rLimits[1] + 1:self.right + 1]
            event_pts = self.yValues[self.left:self.rLimits[0]]
        else:
            self.showInfo(f'No D or R region has been marked!')
            return None, None, None, None, None, None

        B = np.mean(baseline_pts)
        Bnoise = np.std(baseline_pts)
        numBpts = len(baseline_pts)
        A = np.mean(event_pts)
        Anoise = np.std(event_pts)
        numApts = len(event_pts)

        return B, Bnoise, numBpts, A, Anoise, numApts

    def doPenumbralFit(self):

        if self.firstPassPenumbralFit:

            self.firstPassPenumbralFit = False
            self.lastDmetric = self.lastRmetric = 0.0
            self.penumbralFitIterationNumber = 1
            b_intensity, b_noise, num_b_pts, a_intensity, a_noise, num_a_pts = self.extractBaselineAndEventData()

            if b_intensity is None:
                return  # An info message will have already been raised.  No need to do anything else.

            # Get current underlying lightcurve
            self.underlyingLightcurveAns = self.demoUnderlyingLightcurves(baseline=b_intensity, event=a_intensity,
                                                                          plots_wanted=False)
            # Adjust b_intensity and a_intensity to match the underlying lightcurve table
            b_intensity = self.underlyingLightcurveAns['B']
            a_intensity = self.underlyingLightcurveAns['A']
            self.showMsg(
                f'B: {b_intensity:0.2f}  A: {a_intensity:0.2f}   B noise: {b_noise:0.3f}  A noise: {a_noise:0.3f}')

            d_candidates = []
            r_candidates = []
            d_candidate_entry_nums = []
            r_candidate_entry_nums = []

            if self.dLimits:
                self.eventType = 'Donly'
                d_region_intensities = self.yValues[self.dLimits[0]:self.dLimits[1] + 1]
                d_region_entry_nums = range(self.dLimits[0], self.dLimits[1] + 1)

                middle = len(d_region_intensities) // 2
                i = middle
                while d_region_intensities[i] > a_intensity:
                    d_candidates.append(d_region_intensities[i])
                    d_candidate_entry_nums.append(d_region_entry_nums[i])
                    i += 1
                    if i == len(d_region_intensities):
                        break
                i = middle - 1
                while d_region_intensities[i] < b_intensity:
                    d_candidates.append(d_region_intensities[i])
                    d_candidate_entry_nums.append(d_region_entry_nums[i])
                    i -= 1
                    if i < 0:
                        break

                if not d_candidates:
                    self.showMsg('No valid transition points found in designated D region.', bold=True, color='red')
                    return

                # Sort the parallel lists into ascending entry number order
                zipped_lists = zip(d_candidate_entry_nums, d_candidates)
                sorted_pairs = sorted(zipped_lists)
                tuples = zip(*sorted_pairs)
                d_candidate_entry_nums, d_candidates = [list(item) for item in tuples]
                print("d_candidates", d_candidates)
                print("D entry nums", d_candidate_entry_nums)

            if self.rLimits:
                self.eventType = 'Ronly'
                r_region_intensities = self.yValues[self.rLimits[0]:self.rLimits[1] + 1]
                r_region_entry_nums = range(self.rLimits[0], self.rLimits[1] + 1)

                middle = len(r_region_intensities) // 2
                i = middle
                while r_region_intensities[i] < b_intensity:
                    r_candidates.append(r_region_intensities[i])
                    r_candidate_entry_nums.append(r_region_entry_nums[i])
                    i += 1
                    if i == len(r_region_intensities):
                        break
                i = middle - 1
                while r_region_intensities[i] > a_intensity:
                    r_candidates.append(r_region_intensities[i])
                    r_candidate_entry_nums.append(r_region_entry_nums[i])
                    i -= 1
                    if i < 0:
                        break

                if not r_candidates:
                    self.showMsg('No valid transition points found in designated R region.', bold=True, color='red')

                # Sort the parallel lists into ascending entry number order
                zipped_lists = zip(r_candidate_entry_nums, r_candidates)
                sorted_pairs = sorted(zipped_lists)
                tuples = zip(*sorted_pairs)
                r_candidate_entry_nums, r_candidates = [list(item) for item in tuples]
                print("r_candidates", r_candidates)
                print("R entry nums", r_candidate_entry_nums)

            if self.dLimits and self.rLimits:
                self.eventType = 'DandR'

            self.dRegion = None  # Erases the coloring of the D region (when self.newRedrawMainPlot is called)
            self.rRegion = None  # Erases the coloring of the R region (when self.newRedrawMainPlot is called)
            self.eRegion = None  # Erases the coloring of the E region (when self.newRedrawMainPlot is called)

            # Preserve these for possible next pass
            self.d_candidates = d_candidates
            self.d_candidate_entry_nums = d_candidate_entry_nums
            self.r_candidates = r_candidates
            self.r_candidate_entry_nums = r_candidate_entry_nums
            self.penumbral_noise = (b_noise + a_noise) / 2.0

            if not self.userDeterminedEventStats:
                self.A = a_intensity
                self.sigmaA = a_noise

            self.nBpts = num_b_pts
            self.nApts = num_a_pts

            if not self.userDeterminedBaselineStats:
                self.sigmaB = b_noise
                self.B = b_intensity
            if self.userTrimInEffect:
                self.B = b_intensity

            self.snrA = (self.B - self.A) / self.sigmaA
            self.snrB = (self.B - self.A) / self.sigmaB

        # Get current underlying lightcurve
        self.underlyingLightcurveAns = self.demoUnderlyingLightcurves(baseline=self.B, event=self.A,
                                                                      plots_wanted=False)

        # If an error in data entry has occurred, ans will be None
        if self.underlyingLightcurveAns is None:
            self.showMsg(f'An error in the underlying lightcurve parameters has occurred.', bold=True, color='red')
            return

        if len(self.d_candidates) > 0 and len(self.r_candidates) > 0:
            self.eventType = 'DandR'
        elif len(self.d_candidates) > 0:
            self.eventType = 'Donly'
        else:
            self.eventType = 'Ronly'

        if self.eventType in ['Donly', 'DandR']:
            d_list = []
            for i in range(len(self.d_candidates)):
                newD = self.dEdgeCorrected(self.d_candidates[i], self.d_candidate_entry_nums[i])
                if newD is not None:
                    d_list.append(newD)
                else:
                    print('newD came back as None')
            d_mean = np.mean(d_list)
            # print(d_list, d_mean)
        else:
            d_mean = None

        if self.eventType in ['Ronly', 'DandR']:
            r_list = []
            for i in range(len(self.r_candidates)):
                newR = self.rEdgeCorrected(self.r_candidates[i], self.r_candidate_entry_nums[i])
                if newR is not None:
                    r_list.append(newR)
                else:
                    print('newR came back as None')
            r_mean = np.mean(r_list)
            # print(r_list, r_mean)
        else:
            r_mean = None

        self.solution = [None, None]  # Convert from tuple so that next line will be accepted
        self.solution[0] = d_mean
        self.solution[1] = r_mean

        d_time_err_bar = r_time_err_bar = 0.0

        if self.eventType in ['Donly', 'DandR']:
            d_noise = self.penumbral_noise / np.sqrt(len(self.d_candidates))
            mid_intensity = (self.B + self.A) / 2.0
            d_time1 = time_correction(correction_dict=self.underlyingLightcurveAns,
                                      transition_point_intensity=mid_intensity - 2 * d_noise, edge_type='D')
            d_time2 = time_correction(correction_dict=self.underlyingLightcurveAns,
                                      transition_point_intensity=mid_intensity + 2 * d_noise, edge_type='D')
            d_time_err_bar = abs(d_time1 - d_time2) / 2.0

        if self.eventType in ['Ronly', 'DandR']:
            r_noise = self.penumbral_noise / np.sqrt(len(self.r_candidates))
            mid_intensity = (self.B + self.A) / 2.0
            r_time1 = time_correction(correction_dict=self.underlyingLightcurveAns,
                                      transition_point_intensity=mid_intensity - 2 * r_noise, edge_type='R')
            r_time2 = time_correction(correction_dict=self.underlyingLightcurveAns,
                                      transition_point_intensity=mid_intensity + 2 * r_noise, edge_type='R')
            r_time_err_bar = abs(r_time1 - r_time2) / 2.0

        self.minusD = self.plusD = None
        self.minusR = self.plusR = None

        self.penumbralDerrBar = d_time_err_bar
        self.penumbralRerrBar = r_time_err_bar

        self.doPenumbralFitIterationReport()

        if self.eventType in ['Donly', 'DandR']:
            self.Dreport(d_time_err_bar, -d_time_err_bar)
        if self.eventType in ['Ronly', 'DandR']:
            self.Rreport(r_time_err_bar, -r_time_err_bar)

        self.newRedrawMainPlot()
        self.drawSolution()
        self.calcErrBars.setEnabled(True)

        d_improved_msg = 'starting value'
        r_improved_msg = 'starting value'

        d_metric, r_metric = self.calculatePenumbralMetrics(d_mean, r_mean)

        if self.eventType in ['Donly', 'DandR']:
            if self.penumbralFitIterationNumber > 1:
                if d_metric < self.lastDmetric:
                    d_improved_msg = 'improved'
                elif d_metric > self.lastDmetric:
                    d_improved_msg = 'got worse'
                else:
                    d_improved_msg = 'unchanged'
            self.showMsg(f'D fit metric: {d_metric:0.1f}  ({d_improved_msg})', bold=True, blankLine=False)
        if self.eventType in ['Ronly', 'DandR']:
            if self.penumbralFitIterationNumber > 1:
                if r_metric < self.lastRmetric:
                    r_improved_msg = 'improved'
                elif r_metric > self.lastRmetric:
                    r_improved_msg = 'got worse'
                else:
                    r_improved_msg = 'unchanged'
            self.showMsg(f'R fit metric: {r_metric:0.1f}  ({r_improved_msg})', bold=True)

        self.penumbralFitIterationNumber += 1

        self.lastDmetric = d_metric
        self.lastRmetric = r_metric

        return

    def calculatePenumbralMetrics(self, D=None, R=None):
        d_metric = r_metric = None
        time_ranges = self.getUnderlyingLightCurveTimeRanges()

        if D is not None:
            time_ranges[0] = time_ranges[0] / self.timeDelta + D
            time_ranges[1] = time_ranges[1] / self.timeDelta + D
            d_metric = 0.0
            # print('\nd participants in metric')
            # for i in range(int(np.ceil(time_ranges[0])), int(np.ceil(time_ranges[1]))):
            #     print(i, self.yValues[i], i - D,
            #           intensity_at_time(self.underlyingLightcurveAns, (i - D) * self.timeDelta, 'D'))
            n_vals_in_metric = 0
            for i in range(int(np.ceil(time_ranges[0])), int(np.ceil(time_ranges[1]))):
                lightcurve_intensity = intensity_at_time(self.underlyingLightcurveAns, (i - D) * self.timeDelta, 'D')
                d_metric += (self.yValues[i] - lightcurve_intensity) ** 2
                n_vals_in_metric += 1
            d_metric = d_metric / n_vals_in_metric

        if R is not None:
            time_ranges[2] = time_ranges[2] / self.timeDelta + R
            time_ranges[3] = time_ranges[3] / self.timeDelta + R
            r_metric = 0.0
            # print('\nr participants in metric')
            # for i in range(int(np.ceil(time_ranges[2])), int(np.ceil(time_ranges[3]))):
            #     print(i, self.yValues[i], i - R,
            #           intensity_at_time(self.underlyingLightcurveAns, (i - R) * self.timeDelta, 'R'))
            n_vals_in_metric = 0
            for i in range(int(np.ceil(time_ranges[2])), int(np.ceil(time_ranges[3]))):
                lightcurve_intensity = intensity_at_time(self.underlyingLightcurveAns, (i - R) * self.timeDelta, 'R')
                r_metric += (self.yValues[i] - lightcurve_intensity) ** 2
                n_vals_in_metric += 1
            r_metric = r_metric / n_vals_in_metric

        return d_metric, r_metric

    def doPenumbralFitIterationReport(self):
        self.showMsg(f'Penumbral fit iteration {self.penumbralFitIterationNumber}:', bold=True, color='green')
        if self.eventType == 'DandR':
            self.showMsg(f'(in entryNum units) D: {self.solution[0]:0.4f}  R: {self.solution[1]:0.4f}')
        elif self.eventType == 'Donly':
            self.showMsg(f'(in entryNum units) D: {self.solution[0]:0.4f}')
        else:  # Ronly
            self.showMsg(f'(in entryNum units) R: {self.solution[1]:0.4f}')
        self.doLightcurveParameterReport()
        return

    def doLightcurveParameterReport(self):
        if self.enableDiffractionCalculationBox.isChecked():
            self.showMsg(f'Diffraction effects included', blankLine=False)
        else:
            self.showMsg(f'Diffraction effects suppressed', blankLine=False)
        self.showMsg(f'dist(AU): {self.asteroidDistanceEdit.text()}', blankLine=False)
        self.showMsg(f'speed(km/sec): {self.shadowSpeedEdit.text()}', blankLine=False)
        self.showMsg(f'Star diam(mas): {self.starDiameterEdit.text()}', blankLine=False)
        self.showMsg(f'D limb angle: {self.dLimbAngle.value()}', blankLine=False)
        self.showMsg(f'R limb angle: {self.rLimbAngle.value()}')
        return

    def rEdgeCorrected(self, r_best_value, r_best_value_index):
        r_time_corr = time_correction(correction_dict=self.underlyingLightcurveAns,
                                      transition_point_intensity=r_best_value, edge_type='R')
        if r_time_corr is not None:
            r_delta = r_time_corr / self.timeDelta
            r_adj = r_best_value_index + r_delta
            return r_adj
        else:
            return None

    def dEdgeCorrected(self, d_best_value, d_best_value_index):
        d_time_corr = time_correction(correction_dict=self.underlyingLightcurveAns,
                                      transition_point_intensity=d_best_value, edge_type='D')
        if d_time_corr is not None:
            d_delta = d_time_corr / self.timeDelta
            d_adj = d_best_value_index + d_delta
            return d_adj
        else:
            return None

    def getUnderlyingLightCurveTimeRanges(self):
        # We use this routine to find the 'range' of times that are covered by the underlying lightcurve.
        # The times returned are relative to the geometric edge (i.e., time = 0.00)
        B = self.underlyingLightcurveAns['B']
        A = self.underlyingLightcurveAns['A']
        hi_intensity = B - 0.05 * (B - A)
        lo_intensity = A + 0.05 * (B - A)
        d_early_time = - time_correction(correction_dict=self.underlyingLightcurveAns,
                                         transition_point_intensity=hi_intensity, edge_type='D')
        d_late_time = - time_correction(correction_dict=self.underlyingLightcurveAns,
                                        transition_point_intensity=lo_intensity, edge_type='D')
        r_early_time = - time_correction(correction_dict=self.underlyingLightcurveAns,
                                         transition_point_intensity=lo_intensity, edge_type='R')
        r_late_time = - time_correction(correction_dict=self.underlyingLightcurveAns,
                                        transition_point_intensity=hi_intensity, edge_type='R')
        return [d_early_time, d_late_time, r_early_time, r_late_time]

    def findEvent(self):

        if self.timeDelta == 0.0:
            self.showInfo(f'time per reading (timeDelta) has an invalid value of 0.0\n\nCannot proceed.')
            return

        if self.penumbralFitCheckBox.isChecked():
            self.doPenumbralFit()
            return

        need_to_invite_user_to_verify_timestamps = False

        if self.dLimits and self.rLimits:
            self.eventType = "DandR"
            self.showMsg('Locate a "D and R" event has been selected')
        elif self.dLimits:
            self.eventType = "Donly"
            self.showMsg('Locate a "D only" event has been selected')
        elif self.rLimits:
            self.eventType = "Ronly"
            self.showMsg('Locate an "R only" event has been selected')
        else:
            self.showMsg('Use min/max event to locate a "D and R" event has been selected')

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
                if self.minEvent < 2:
                    self.showInfo('minEvent must be greater than 1')
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

        if not minText == '<blank>':
            self.showMsg('minEvent: ' + minText + '  maxEvent: ' + maxText)

        if not minText == '<blank>' and not maxText == '<blank>':
            self.eventType = 'DandR'
            self.dLimits = []
            self.rLimits = []
        elif not self.dLimits and not self.rLimits:
            self.showInfo('No search criteria have been entered')
            return

        if not self.ne3NotInUseRadioButton.isChecked():
            yPosition = self.targetStarYpositionSpinBox.value()
            if yPosition == 0:
                self.showInfo("You need to set a valid value for the Night Eagle 3 target's Y position.")
                return

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

        self.runSolver = True
        self.calcErrBars.setEnabled(False)
        self.fillExcelReportButton.setEnabled(False)

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
                    self.minEvent = max(self.minEvent, 2)
                    self.maxEvent = self.rLimits[1] - self.left
                solverGen = find_best_r_only_from_min_max_size(
                    self.yValues, self.left, self.right, self.minEvent,
                    self.maxEvent
                )

            else:  # Donly
                self.showMsg('New solver results...', color='blue', bold=True)
                if candFrom == 'usedSize':
                    pass
                else:
                    self.minEvent = self.right - self.dLimits[1]
                    self.minEvent = max(self.minEvent, 2)
                    self.maxEvent = self.right - self.dLimits[0] - 1

                solverGen = find_best_d_only_from_min_max_size(
                    self.yValues, self.left, self.right, self.minEvent,
                    self.maxEvent
                )

            if solverGen is None:
                self.showInfo('Generator version not yet implemented')
                return

            self.cancelRequested = False

            d = r = -1
            b = a = 0.0
            sigmaB = sigmaA = 0.0

            for item in solverGen:
                if item[0] == 1.0:
                    self.progressBar.setValue(int(item[1] * 100))
                    QtWidgets.QApplication.processEvents()
                    if self.cancelRequested:
                        self.cancelRequested = False
                        self.runSolver = False
                        self.showMsg('Solution search was cancelled')
                        self.progressBar.setValue(0)
                        return
                elif item[0] == -1.0:
                    self.showMsg(
                        'No event fitting search criteria could be found.')
                    self.progressBar.setValue(0)
                    self.runSolver = False
                    return
                else:
                    # d, r, b, a, sigmaB, sigmaA, metric = item
                    _, _, d, r, b, a, sigmaB, sigmaA, metric = item
                    if d == -1.0:
                        d = None
                    if r == -1.0:
                        r = None
                    self.solution = [d, r]
                    self.progressBar.setValue(0)

            self.showMsg('Integer (non-subframe) solution...', blankLine=False)
            self.showMsg(
                'sigB:%.2f  sigA:%.2f B:%.2f A:%.2f' %
                (sigmaB, sigmaA, b, a),
                blankLine=False)
            self.displaySolution(subframe=False)  # First solution

            # This fills in self.sigmaB and self.sigmaA (incorrectly) but is useful
            # because it tests correlation coefficients to warn of the need for block integration
            self.extract_noise_parameters_from_iterative_solution()

            DfitMetric = RfitMetric = 0.0

            if not self.ne3NotInUseRadioButton.isChecked():
                # We're being asked to perform an exponential edge fit for the Night Eagle 3 camera
                snr = b / sigmaB
                if snr < 4.0:
                    self.dnrOffRadioButton.setChecked(True)
                    self.showMsg(f'The snr of {snr:0.1f} is too low to use an NE3 exponential solution...', color='red',
                                 blankLine=False)
                    self.showMsg(f'... NE3 DNR:Off has been automatically checked.', color='red')
                self.showUnderlyingLightcurveCheckBox.setChecked(True)
                resultFound, DfitMetric, RfitMetric = self.doExpFit(b, a)
                if not resultFound:
                    return

            # Here is where we get sigmaB and sigmaA with the transition points excluded
            subDandR, new_b, new_a, newSigmaB, newSigmaA = subFrameAdjusted(
                eventType=self.eventType, cand=(d, r), B=b, A=a,
                sigmaB=self.sigmaB, sigmaA=self.sigmaA, yValues=self.yValues,
                left=self.left, right=self.right)

            if not self.ne3NotInUseRadioButton.isChecked():
                if not self.dnrOffRadioButton.isChecked():
                    subDandR = self.solution

            # Here we apply the correction from our computed underlying lightcurves.
            if self.userDeterminedEventStats:
                AtoUse = self.A
            else:
                AtoUse = new_a

            if self.userDeterminedBaselineStats:
                BtoUse = self.B
            else:
                BtoUse = new_b

            self.underlyingLightcurveAns = self.demoUnderlyingLightcurves(baseline=BtoUse, event=AtoUse,
                                                                          plots_wanted=False)

            # If an error in data entry has occurred, ans will be None
            if self.underlyingLightcurveAns is None:
                self.showMsg(f'An error in the underlying lightcurve parameters has occurred.', bold=True, color='red')
                return

            D = R = 0
            if self.eventType == 'Donly' or self.eventType == 'DandR':
                D = int(subDandR[0])
            if self.eventType == 'Ronly' or self.eventType == 'DandR':
                R = int(subDandR[1])

            if (self.eventType == 'Donly' or self.eventType == 'DandR') and not D == subDandR[0]:
                if self.exponentialDtheoryPts is None:
                    d_time_corr = time_correction(correction_dict=self.underlyingLightcurveAns,
                                                  transition_point_intensity=self.yValues[D], edge_type='D')

                    d_delta = d_time_corr / self.timeDelta
                    d_adj = D + d_delta
                    # self.showMsg(f'd_time_correction: {d_time_corr:0.4f}  new D: {d_adj:0.4f}')
                    subDandR[0] = d_adj

            if (self.eventType == 'Ronly' or self.eventType == 'DandR') and not R == subDandR[1]:
                if self.exponentialRtheoryPts is None:
                    r_time_corr = time_correction(correction_dict=self.underlyingLightcurveAns,
                                                  transition_point_intensity=self.yValues[R], edge_type='R')

                    r_delta = r_time_corr / self.timeDelta
                    r_adj = R + r_delta
                    # self.showMsg(f'r_time_correction: {r_time_corr:0.4f}  new R: {r_adj:0.4f}')
                    subDandR[1] = r_adj

            if self.exponentialDtheoryPts is None and self.exponentialRtheoryPts is None:
                self.solution = subDandR

            self.showMsg('Subframe adjusted solution...', blankLine=False)
            self.showMsg(
                'sigB:%.2f  sigA:%.2f B:%.2f A:%.2f' %
                (newSigmaB, newSigmaA, new_b, new_a),
                blankLine=False)

            need_to_invite_user_to_verify_timestamps = self.displaySolution()  # Adjusted solution

            if not self.userDeterminedBaselineStats:
                self.B = new_b
                self.sigmaB = newSigmaB
            if self.userTrimInEffect:
                self.B = new_b
            if not self.userDeterminedEventStats:
                self.A = new_a
                self.sigmaA = newSigmaA

            self.dRegion = None
            self.rRegion = None

            self.showMsg('... end New solver results', color='blue', bold=True)

            if not self.ne3NotInUseRadioButton.isChecked():
                self.showMsg(f'D edge fit metric: {DfitMetric:0.4f}   R edge fit metric: {RfitMetric:0.4f}')

        if self.runSolver and self.solution:
            D, R = self.solution  # type: int
            if D is not None:
                D = round(D, 4)
            if R is not None:
                R = round(R, 4)
            self.solution = [D, R]
            if self.eventType == 'DandR':
                # ans = '(%.2f,%.2f) B: %.2f  A: %.2f' % (D, R, self.B, self.A)
                # Check for solution search based on min max event limits
                if self.maxEvent is not None:
                    if (R - D) > self.maxEvent:
                        self.newRedrawMainPlot()
                        self.showMsg('Invalid solution: max event limit constrained solution', color='red', bold=True)
                        self.showInfo('The solution is likely incorrect because the max event limit' +
                                      ' was set too low.  Increase that limit and try again.')
                        return
                    if self.minEvent >= (R - D) and self.minEvent > 2:
                        self.newRedrawMainPlot()
                        self.showMsg('Invalid solution: min event limit constrained solution!', color='red', bold=True)
                        self.showInfo('The solution is likely incorrect because the min event limit' +
                                      ' was set too high.  Decrease that limit and try again.')
                        return
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

        self.newRedrawMainPlot()

        self.calcErrBars.setEnabled(True)

        if need_to_invite_user_to_verify_timestamps:
            self.showInfo(f'The timing of the event found depends on the correctness '
                          f'of the timestamp assigned to the D and R frames.  Since '
                          f'OCR may have produced incorrect values, the relevant video frames have been found '
                          f'and displayed for your inspection.\n\n'
                          f'Please verify visually that the timestamp values are correct.\n\n'
                          f'If they are wrong, note the correct values and use manual timestamp entry '
                          f'to "rescue" the observation.')

    def getNe3TimeCorrection(self):
        yPosition = self.targetStarYpositionSpinBox.value()
        if yPosition == 0:
            self.showInfo("You need to set a valid value for the target's Y position.")
            return None
        # We assume that NE3 cameras always have a field time of 0.0167 seconds and a 640x480 format
        fieldTime = 0.0167
        correction = fieldTime - yPosition * (fieldTime / 480)
        return -correction

    def writeNe3UsageReport(self):
        self.showMsg('')
        self.showMsg('Night Eagle 3 exponential curve fitting was employed with parameters:',
                     color='blue', bold=True)
        if self.dnrOffRadioButton.isChecked():
            self.showMsg(f'==== DNR: OFF', color='black', bold=True, blankLine=False)
        elif self.dnrLowRadioButton.isChecked():
            tcD = self.dnrLowDspinBox.value()
            tcR = self.dnrLowRspinBox.value()
            msg = f'==== DNR:LOW    TC-D: {tcD:0.2f}    TC-R: {tcR:0.2f}'
            self.showMsg(msg, color='black', bold=True, blankLine=False)
        elif self.dnrMiddleRadioButton.isChecked():
            tcD = self.dnrMiddleDspinBox.value()
            tcR = self.dnrMiddleRspinBox.value()
            msg = f'==== DNR:MIDDLE   TC-D: {tcD:0.2f}    TC-R: {tcR:0.2f}'
            self.showMsg(msg, color='black', bold=True, blankLine=False)
        elif self.dnrHighRadioButton.isChecked():
            tcD = self.dnrHighDspinBox.value()
            tcR = self.dnrHighRspinBox.value()
            msg = f'==== DNR:HIGH    TC-D: {tcD:0.2f}    TC-R: {tcR:0.2f}'
            self.showMsg(msg, color='black', bold=True, blankLine=False)

        self.showMsg(f'==== y position of target: {self.targetStarYpositionSpinBox.value():4d}',
                     color='black', bold=True)

    def scoreExpFit(self, b, a, TC, edge):
        numTheoryPoints = 5

        if edge == 'D':
            D = int(self.solution[0])

            # Select a set of yValues to use for fitting
            p1 = D - 10
            p2 = D + 10
            if p1 < 0:
                p1 = 0
            if p2 > len(self.yValues) - 1:
                p2 = len(self.yValues) - 1

            actual = self.yValues[p1:p2 + 1]

            bestMatchPoint = ex.locateIndexOfBestMatchPoint(
                numTheoryPts=numTheoryPoints, B=b, A=a, offset=0, N=TC,
                actual=actual, edge=edge
            )

            bestOffset = ex.locateBestOffset(
                numTheoryPts=numTheoryPoints, B=b, A=a, N0=TC,
                actual=actual, matchPoint=bestMatchPoint, edge=edge
            )

            self.exponentialDtheoryPts = ex.getDedgePoints(
                numPoints=numTheoryPoints, B=b, A=a, offset=bestOffset, N=TC
            )
            self.exponentialDinitialX = bestMatchPoint + p1

            self.exponentialDedge = bestMatchPoint + bestOffset + p1 + numTheoryPoints

            DfitMetric = ex.scoreDedge(numTheoryPts=numTheoryPoints, B=b, A=a, offset=bestOffset,
                                       N=TC, actual=actual, matchPoint=bestMatchPoint)
            return DfitMetric

        if edge == 'R':
            R = int(self.solution[1])

            # Select a set of yValues to use for fitting
            p1 = R - 10
            p2 = R + 10
            if p1 < 0:
                p1 = 0
            if p2 > len(self.yValues) - 1:
                p2 = len(self.yValues) - 1

            actual = self.yValues[p1:p2 + 1]

            bestMatchPoint = ex.locateIndexOfBestMatchPoint(
                numTheoryPts=numTheoryPoints, B=b, A=a, offset=0, N=TC,
                actual=actual, edge=edge
            )

            bestOffset = ex.locateBestOffset(
                numTheoryPts=numTheoryPoints, B=b, A=a, N0=TC,
                actual=actual, matchPoint=bestMatchPoint, edge=edge
            )

            self.exponentialRtheoryPts = ex.getRedgePoints(
                numPoints=numTheoryPoints, B=b, A=a, offset=bestOffset, N=TC
            )
            self.exponentialRinitialX = bestMatchPoint + p1

            self.exponentialRedge = bestMatchPoint + bestOffset + p1 + numTheoryPoints

            RfitMetric = ex.scoreRedge(numTheoryPts=numTheoryPoints, B=b, A=a, offset=bestOffset,
                                       N=TC, actual=actual, matchPoint=bestMatchPoint)
            return RfitMetric

    def refineNe3TimeConstant(self, b, a, TC, edge):
        def doRefinementSteps(stepSize, timeConstant):
            oldScore = self.scoreExpFit(b=b, a=a, TC=timeConstant, edge=edge)
            direction = 1
            noImprovement = False
            while True:
                # Take a step
                timeConstant += direction * stepSize
                if timeConstant <= 0:
                    timeConstant = 0.1
                newScore = self.scoreExpFit(b=b, a=a, TC=timeConstant, edge=edge)
                if newScore < oldScore:
                    oldScore = newScore
                else:
                    if noImprovement:
                        break
                    direction *= -1
                    timeConstant += direction * stepSize
                    noImprovement = True
            return timeConstant - direction * stepSize

        newTC = doRefinementSteps(1.0, TC)
        newTC = doRefinementSteps(0.1, newTC)
        newTC = doRefinementSteps(0.01, newTC)

        return newTC

    def doExpFit(self, b, a):

        DfitMetric = RfitMetric = 0.0

        if self.userDeterminedEventStats:
            a = self.A

        if self.userDeterminedBaselineStats:
            b = self.B

        timeCorrection = self.getNe3TimeCorrection()
        if timeCorrection is None:
            return False, 0.0, 0.0

        # Convert timeCorrection to frame/field fraction
        deltaPosition = timeCorrection / self.timeDelta

        if self.eventType == 'Donly' or self.eventType == 'DandR':
            if self.dnrLowRadioButton.isChecked():
                DtimeConstant = self.dnrLowDspinBox.value()
            elif self.dnrMiddleRadioButton.isChecked():
                DtimeConstant = self.dnrMiddleDspinBox.value()
            elif self.dnrHighRadioButton.isChecked():
                DtimeConstant = self.dnrHighDspinBox.value()
            elif self.dnrOffRadioButton.isChecked():
                DtimeConstant = 0.001
            else:
                self.showInfo('Programming error - this point should never be reached')
                return False, 0.0, 0.0

            if not self.dnrOffRadioButton.isChecked():
                DtimeConstant = self.refineNe3TimeConstant(b=b, a=a, TC=DtimeConstant, edge='D')

                if self.dnrLowRadioButton.isChecked():
                    self.dnrLowDspinBox.setValue(DtimeConstant)
                elif self.dnrMiddleRadioButton.isChecked():
                    self.dnrMiddleDspinBox.setValue(DtimeConstant)
                elif self.dnrHighRadioButton.isChecked():
                    self.dnrHighDspinBox.setValue(DtimeConstant)

            self.showMsg(f'D timeconstant: {DtimeConstant:0.2f}', color='red', bold=True)

            DfitMetric = self.scoreExpFit(b=b, a=a, TC=DtimeConstant, edge='D')

            # Here we add self.timeDelta to follow the convention of the MLE solver that the output of the
            # camera shows what happened one timeDelta in the past and 'fudges' the solution so that
            # time-on-the-wire matches time-in-the-camera
            if self.fieldMode:
                self.solution[0] = self.exponentialDedge + 0.5
            else:
                self.solution[0] = self.exponentialDedge + 1.0

            self.solution[0] += deltaPosition

        if self.eventType == 'Ronly' or self.eventType == 'DandR':

            if self.dnrLowRadioButton.isChecked():
                RtimeConstant = self.dnrLowRspinBox.value()
            elif self.dnrMiddleRadioButton.isChecked():
                RtimeConstant = self.dnrMiddleRspinBox.value()
            elif self.dnrHighRadioButton.isChecked():
                RtimeConstant = self.dnrHighRspinBox.value()
            elif self.dnrOffRadioButton.isChecked():
                RtimeConstant = 0.001
            else:
                self.showInfo('Programming error - this point should never be reached')
                return False, 0.0, 0.0

            if not self.dnrOffRadioButton.isChecked():
                RtimeConstant = self.refineNe3TimeConstant(b=b, a=a, TC=RtimeConstant, edge='R')
                if self.dnrLowRadioButton.isChecked():
                    self.dnrLowRspinBox.setValue(RtimeConstant)
                elif self.dnrMiddleRadioButton.isChecked():
                    self.dnrMiddleRspinBox.setValue(RtimeConstant)
                elif self.dnrHighRadioButton.isChecked():
                    self.dnrHighRspinBox.setValue(RtimeConstant)

            self.showMsg(f'R timeconstant: {RtimeConstant:0.2f}', color='red', bold=True)

            RfitMetric = self.scoreExpFit(b=b, a=a, TC=RtimeConstant, edge='R')

            # Here we add self.timeDelta to follow the convention of the MLE solver that the output of the
            # camera shows what happened one timeDelta in the past and 'fudges' the solution so that
            # time-on-the-wire matches time-in-the-camera
            if self.fieldMode:
                self.solution[1] = self.exponentialRedge + 0.5
            else:
                self.solution[1] = self.exponentialRedge + 1.0

            self.solution[1] += deltaPosition

            # RfitMetric = ex.scoreRedge(numTheoryPts=numTheoryPoints, B=b, A=a, offset=bestOffset,
            #                            N=RtimeConstant, actual=actual, matchPoint=bestMatchPoint)

            # self.showInfo(f'bestMatchPoint: {bestMatchPoint}  bestOffset: {bestOffset:0.2f} R: {R}')

        return True, DfitMetric, RfitMetric

    def fillTableViewOfData(self):

        self.table.setRowCount(self.dataLen)
        self.table.setVerticalHeaderLabels([str(i) for i in range(self.dataLen)])

        min_frame = int(trunc(float(self.yFrame[0])))
        max_frame = int(trunc(float(self.yFrame[-1])))
        if self.frameNumSpinBox.isEnabled():
            self.frameNumSpinBox.setMinimum(min_frame)
            self.frameNumSpinBox.setMaximum(max_frame)

        for i in range(self.dataLen):
            neatStr = fp.to_precision(self.yValues[i], 6)
            newitem = QtWidgets.QTableWidgetItem(str(neatStr))
            self.table.setItem(i, self.targetIndex + 2, newitem)
            newitem = QtWidgets.QTableWidgetItem(str(self.yTimes[i]))
            self.table.setItem(i, 1, newitem)
            frameNum = float(self.yFrame[i])
            if not np.ceil(frameNum) == np.floor(frameNum):
                self.fieldMode = True
            newitem = QtWidgets.QTableWidgetItem(str(self.yFrame[i]))
            self.table.setItem(i, 0, newitem)
            nextColumn = 2
            if len(self.LC1) > 0:
                neatStr = fp.to_precision(self.LC1[i], 6)
                newitem = QtWidgets.QTableWidgetItem(str(neatStr))
                if not nextColumn == self.targetIndex + 2:
                    self.table.setItem(i, 2, newitem)
                nextColumn += 1
            if len(self.LC2) > 0:
                neatStr = fp.to_precision(self.LC2[i], 6)
                newitem = QtWidgets.QTableWidgetItem(str(neatStr))
                if not nextColumn == self.targetIndex + 2:
                    self.table.setItem(i, 3, newitem)
                nextColumn += 1
            if len(self.LC3) > 0:
                neatStr = fp.to_precision(self.LC3[i], 6)
                newitem = QtWidgets.QTableWidgetItem(str(neatStr))
                if not nextColumn == self.targetIndex + 2:
                    self.table.setItem(i, 4, newitem)
                nextColumn += 1
            if len(self.LC4) > 0:
                neatStr = fp.to_precision(self.LC4[i], 6)
                newitem = QtWidgets.QTableWidgetItem(str(neatStr))
                if not nextColumn == self.targetIndex + 2:
                    self.table.setItem(i, 5, newitem)
                nextColumn += 1
            if len(self.extra) > 0:
                for k, lightcurve in enumerate(self.extra):
                    neatStr = fp.to_precision(lightcurve[i], 6)
                    newitem = QtWidgets.QTableWidgetItem(str(neatStr))
                    if not nextColumn == self.targetIndex + 2:
                        self.table.setItem(i, 6 + k, newitem)
                    nextColumn += 1
            if len(self.demoLightCurve) > 0:
                neatStr = fp.to_precision(self.demoLightCurve[i], 6)
                newitem = QtWidgets.QTableWidgetItem(str(neatStr))
                self.table.setItem(i, nextColumn, newitem)

        self.table.resizeColumnsToContents()
        self.writeCSVButton.setEnabled(True)

    def doManualTimestampEntry(self):
        errmsg = ''
        while errmsg != 'ok':
            errmsg, manualTime, dataEntered, actualFrameCount, expectedFrameCount = \
                manualTimeStampEntry(self.yFrame, TSdialog(), self.flashEdges)
            if errmsg != 'ok':
                if errmsg == 'cancelled':
                    return
                else:
                    self.showInfo(errmsg)
            else:
                self.showMsg(dataEntered, bold=True)
                if abs(actualFrameCount - expectedFrameCount) >= 0.12:
                    msg = (
                        f'Possible dropped readings !!!\n\n'
                        f'Reading count input: {actualFrameCount:.2f}  \n\n'
                        f'Reading count computed from frame rate: {expectedFrameCount:.2f}'
                    )
                    self.showMsg(msg, color='red', bold=True)
                    self.showInfo(msg)

                # If user cancelled out of timestamp entry dialog,
                # then manualTime will be an empty list.
                if manualTime:
                    self.yTimes = manualTime[:]
                    self.timeDelta, self.outliers, self.errRate = getTimeStepAndOutliers(
                        self.yTimes)
                    self.expDurEdit.setText(fp.to_precision(self.timeDelta, 6))

                    self.fillTableViewOfData()
                    self.newRedrawMainPlot()
                    self.showMsg(
                        'timeDelta: ' + fp.to_precision(self.timeDelta, 6) +
                        ' seconds per reading' +
                        ' (timeDelta calculated from manual input timestamps)',
                        blankLine=False)
                    self.showMsg(
                        'timestamp error rate: ' + fp.to_precision(100 *
                                                                   self.errRate,
                                                                   3) + '%')
                    self.fillTableViewOfData()

    def enableDisableFrameViewControls(self, state_to_set):
        self.viewFrameButton.setEnabled(state_to_set)
        self.frameNumSpinBox.setEnabled(state_to_set)
        self.fieldViewCheckBox.setEnabled(state_to_set)
        self.flipXaxisCheckBox.setEnabled(state_to_set)
        self.flipYaxisCheckBox.setEnabled(state_to_set)

    def readDataFromFile(self):

        self.initializeVariablesThatDontDependOnAfile()
        self.blockSize = 1
        self.fieldMode = False

        self.pathToVideo = None

        self.enableDisableFrameViewControls(state_to_set=False)

        self.disableAllButtons()
        self.mainPlot.clear()
        self.textOut.clear()
        self.initializeTableView()

        if self.externalCsvFilePath is None:
            # Open a file select dialog
            self.filename, _ = QFileDialog.getOpenFileName(
                self,  # parent
                "Select light curve csv file",  # title for dialog
                self.settings.value('lightcurvedir', ""),  # starting directory
                "Csv files (*.csv)")
        else:
            self.filename = self.externalCsvFilePath
            self.externalCsvFilePath = None

        if self.filename:
            self.initializeLightcurvePanel()
            # self.switchToTabNamed('Lightcurves')
            QtWidgets.QApplication.processEvents()
            self.userDeterminedBaselineStats = False
            self.userDeterminedEventStats = False
            self.setWindowTitle('PYOTE Version: ' + version.version() + '  File being processed: ' + self.filename)
            dirpath, _ = os.path.split(self.filename)
            self.logFile, _ = os.path.splitext(self.filename)
            self.logFile = self.logFile + '.PYOTE.log'

            self.detectabilityLogFile, _ = os.path.splitext(self.filename)
            self.detectabilityLogFile = self.detectabilityLogFile + '.PYOTE.detectabillity.log'

            self.normalizationLogFile, _ = os.path.splitext(self.filename)
            self.normalizationLogFile = self.normalizationLogFile + '.PYOTE.normalization.log'

            curDateTime = datetime.datetime.today().ctime()
            self.showMsg('')
            self.showMsg(
                '#' * 20 + ' PYOTE ' + version.version() + '  session started: ' + curDateTime + '  ' + '#' * 20)

            self.showMsg('', alternateLogFile=self.detectabilityLogFile)
            self.showMsg(
                '#' * 20 + ' PYOTE ' + version.version() + '  session started: ' + curDateTime + '  ' + '#' * 20,
                alternateLogFile=self.detectabilityLogFile)

            self.showMsg('', alternateLogFile=self.normalizationLogFile)
            self.showMsg(
                '#' * 20 + ' PYOTE ' + version.version() + '  session started: ' + curDateTime + '  ' + '#' * 20,
                alternateLogFile=self.normalizationLogFile)

            # Make the directory 'sticky'
            self.settings.setValue('lightcurvedir', dirpath)
            self.settings.sync()
            self.showMsg('filename: ' + self.filename, bold=True, color="red")

            columnPrefix = self.pymovieDataColumnPrefixComboBox.currentText()

            try:
                self.outliers = []
                frame, time, value, self.secondary, self.ref2, self.ref3, self.extra, \
                    self.aperture_names, self.headers = readLightCurve(self.filename, pymovieColumnType=columnPrefix)
                self.showMsg(f'If the csv file came from PyMovie - columns with prefix: {columnPrefix} will be read.')
                values = [float(item) for item in value]
                self.yValues = np.array(values)  # yValues = curve to analyze
                self.dataLen = len(self.yValues)
                self.LC1 = np.array(values)

                self.pymovieFileInUse = False

                # Check headers to see if this is a PyMovie file.  Grab the
                # path to video file if it is a PyMovie file
                for header in self.headers:
                    if header.startswith('# PyMovie'):
                        self.pymovieFileInUse = True
                    if header.startswith('# PyMovie') or header.startswith('Limovie'):
                        for line in self.headers:
                            if line.startswith('# source:') or line.startswith('"FileName :'):

                                if line.startswith('# source:'):  # PyMovie format
                                    self.pathToVideo = line.replace('# source:', '', 1).strip()
                                if line.startswith('"FileName :'):  # Limovie format
                                    self.pathToVideo = line.replace('"FileName :', '', 1).strip()
                                    self.pathToVideo = self.pathToVideo.strip('"')

                                if os.path.isfile(self.pathToVideo):
                                    _, ext = os.path.splitext(self.pathToVideo)
                                    if ext == '.avi':
                                        ans = readAviFile(0, self.pathToVideo)
                                        if not ans['success']:
                                            self.showMsg(
                                                f'Attempt to read .avi file gave errmg: {ans["errmsg"]}',
                                                color='red', bold=True)
                                            self.pathToVideo = None
                                        else:
                                            self.showMsg(f'fourcc code of avi: {ans["fourcc"]}', blankLine=False)
                                            self.showMsg(f'fps: {ans["fps"]}', blankLine=False)
                                            self.showMsg(f'avi contains {ans["num_frames"]} frames')
                                            # Enable frame view controls
                                            self.enableDisableFrameViewControls(state_to_set=True)
                                    elif ext == '.ser':
                                        ans = readSerFile(0, self.pathToVideo)
                                        if not ans['success']:
                                            self.showMsg(
                                                f'Attempt to read .ser file gave errmg: {ans["errmsg"]}',
                                                color='red', bold=True)
                                            self.pathToVideo = None
                                        else:
                                            # Enable frame view controls
                                            self.enableDisableFrameViewControls(state_to_set=True)
                                    elif ext == '':
                                        ans = readFitsFile(0, self.pathToVideo)
                                        if not ans['success']:
                                            self.showMsg(
                                                f'Attempt to read FITS folder gave errmg: {ans["errmsg"]}',
                                                color='red', bold=True)
                                            self.pathToVideo = None
                                        else:
                                            # Enable frame view controls
                                            self.showMsg(f'{ans["num_frames"]} .fits files were found in FITS folder')
                                            self.enableDisableFrameViewControls(state_to_set=True)
                                    elif ext == '.adv':
                                        # For now, we assume that .adv files have embedded timestamps and
                                        # so there is no need to display frames for visual OCR verification
                                        self.pathToVideo = None
                                    elif ext == '.aav':
                                        ans = readAavFile(0, self.pathToVideo)
                                        if not ans['success']:
                                            self.showMsg(
                                                f'Attempt to read .aav file gave errmg: {ans["errmsg"]}',
                                                color='red', bold=True)
                                            self.pathToVideo = None
                                        else:
                                            # Enable frame view controls
                                            self.enableDisableFrameViewControls(state_to_set=True)
                                    else:
                                        self.showMsg(f'Unexpected file type of {ext} found.')
                                else:
                                    self.showMsg(f'video source file {self.pathToVideo} could not be found.')
                                    self.pathToVideo = None

                # Automatically select all points
                # noinspection PyUnusedLocal
                self.yStatus = [INCLUDED for _i in range(self.dataLen)]

                refStar = [float(item) for item in self.secondary]

                vals = [float(item) for item in self.secondary]
                self.LC2 = np.array(vals)

                vals = [float(item) for item in self.ref2]
                self.LC3 = np.array(vals)

                vals = [float(item) for item in self.ref3]
                self.LC4 = np.array(vals)

                # A pymovie csv file can have more than 4 lightcurves.  The 'extra' lightcurves beyond
                # the standard max of 4 are placed in self.extra which is an array of lightcurves
                if self.extra:
                    for i, light_curve in enumerate(self.extra):
                        vals = [float(item) for item in light_curve]
                        self.extra[i] = np.array(vals[:])

                self.initializeTableView()
                self.yOffsetSpinBoxes[0].setEnabled(False)

                # If no timestamps were found in the input file, prompt for manual entry
                if self.timestampListIsEmpty(time):
                    self.showMsg('Manual entry of timestamps is required.',
                                 bold=True)
                    # If the user knew there were no timestamps, the is no
                    # reason to show info box.
                    if not self.manualTimestampCheckBox.isChecked():
                        self.showInfo('This file does not contain timestamp '
                                      'entries --- manual entry of two '
                                      'timestamps is required.'
                                      '\n\nEnter the timestamp '
                                      'values that the avi '
                                      'processing software (Limovie, Tangra, '
                                      'etc) would have produced '
                                      'had the OCR process not failed using the '
                                      'View frame button to display the frames '
                                      'you want to use for timestamp purposes.\n\n'
                                      'By working in this manner, you can continue '
                                      'processing the file as though OCR had '
                                      'succeeded and then follow the standard '
                                      'procedure for reporting results through '
                                      'the IOTA event reporting spreadsheet ('
                                      'which will make the needed corrections for camera delay and VTI offset).')

                self.showMsg('=' * 20 + ' file header lines ' + '=' * 20, bold=True, blankLine=False)
                for item in self.headers:
                    self.showMsg(item, blankLine=False)
                self.showMsg('=' * 20 + ' end header lines ' + '=' * 20, bold=True)

                self.yTimes = time[:]
                self.yValues = np.array(values)
                self.yValCopy = np.ndarray(shape=(len(self.yValues),))
                np.copyto(self.yValCopy, self.yValues)
                self.yRefStarCopy = np.array(refStar)

                self.dataLen = len(self.yValues)
                self.yFrame = frame[:]

                # Automatically select all points
                # noinspection PyUnusedLocal
                self.yStatus = [INCLUDED for _i in range(self.dataLen)]
                self.left = 0
                self.right = self.dataLen - 1

                self.mainPlot.autoRange()

                self.mainPlot.setMouseEnabled(x=True, y=False)

                self.setDataLimits.setEnabled(True)
                self.writePlot.setEnabled(True)

                self.markDzone.setEnabled(True)
                self.markRzone.setEnabled(True)
                self.singlePointDropButton.setEnabled(True)
                self.markEzone.setEnabled(not self.ne3NotInUseRadioButton.isChecked())

                self.calcFlashEdge.setEnabled(True)
                self.minEventEdit.setEnabled(True)
                self.maxEventEdit.setEnabled(True)
                self.locateEvent.setEnabled(True)

                self.firstPassPenumbralFit = True

                self.doBlockIntegration.setEnabled(True)
                self.startOver.setEnabled(True)
                self.fillTableViewOfData()

                self.timeDelta, self.outliers, self.errRate = getTimeStepAndOutliers(self.yTimes)
                self.expDurEdit.setText(fp.to_precision(self.timeDelta, 6))

                self.showMsg('timeDelta: ' + fp.to_precision(self.timeDelta, 6) + ' seconds per reading',
                             blankLine=False)
                self.showMsg('timestamp error rate: ' + fp.to_precision(100 *
                                                                        self.errRate, 3) + '%')

                # self.changePrimary()  # Done only to fill in the light-curve name boxes
                self.solution = None
                #     self.newRedrawMainPlot()
                self.mainPlot.autoRange()

                if self.outliers:
                    self.showTimestampErrors.setEnabled(True)
                    self.showTimestampErrors.setChecked(True)
                self.newRedrawMainPlot()
                self.mainPlot.autoRange()

                if self.timeDelta == 0.0 and not self.manualTimestampCheckBox.isChecked():
                    self.showInfo("Analysis of timestamp fields resulted in "
                                  "an invalid timeDelta of 0.0\n\nSuggestion: Enable manual timestamp entry (checkbox at top center)"
                                  ", then press the now active 'Manual timestamp entry' button."
                                  "\n\nThis will give you a chance to "
                                  "manually correct the timestamps using "
                                  "the data available in the table in the "
                                  "lower left corner or incorporate flash timing data.")
            except Exception as e:
                self.showMsg(str(e))
                self.showMsg(f'This error may be because the data column name {columnPrefix} does not exist.')

    def illustrateTimestampOutliers(self):
        for pos in self.outliers:
            vLine = pg.InfiniteLine(pos=pos + 0.5, pen=(255, 0, 0))
            self.mainPlot.addItem(vLine)

    def prettyPrintCorCoefs(self):
        outStr = 'noise corr coefs: ['

        posCoefs = []
        for coef in self.corCoefs:
            if coef < acfCoefThreshold:
                break
            posCoefs.append(coef)

        for i in range(len(posCoefs) - 1):
            outStr = outStr + fp.to_precision(posCoefs[i], 3) + ', '
        outStr = outStr + fp.to_precision(posCoefs[-1], 3)
        outStr = outStr + ']  (based on ' + str(self.numPtsInCorCoefs) + ' points)'
        # outStr = outStr + '  sigmaB: ' + fp.to_precision(self.sigmaB, 4)
        outStr = outStr + '  sigmaB: ' + f'{self.sigmaB:.2f}'
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
            for i in range(left, right + 1):
                self.eventXvals.append(i)
                self.eventYvals.append(self.yValues[i])
            self.showSelectedPoints('Points selected for event noise '
                                    'analysis: ')

        self.removePointSelections()
        _, self.numNApts, self.sigmaA = getCorCoefs(self.eventXvals, self.eventYvals)
        self.showMsg('Event noise analysis done using ' + str(self.numNApts) +
                     ' points ---  sigmaA: ' + fp.to_precision(self.sigmaA, 4))

        self.newRedrawMainPlot()

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
            for i in range(left, right + 1):
                self.baselineXvals.append(i)
                self.baselineYvals.append(self.yValues[i])
            self.showSelectedPoints('Points selected for baseline noise '
                                    'analysis: ')

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
            if self.corCoefs[1] >= 0.7 and self.ne3NotInUseRadioButton.isChecked():
                self.showInfo('The auto-correlation coefficient at lag 1 is suspiciously large. '
                              'This may be because the light curve needs some degree of block integration. '
                              'Failure to do a needed block integration allows point-to-point correlations caused by '
                              'the camera integration to artificially induce non-physical correlated noise.')
            elif len(self.corCoefs) > 2:
                if self.corCoefs[2] >= 0.3 and self.ne3NotInUseRadioButton.isChecked():
                    self.showInfo('The auto-correlation coefficient at lag 2 is suspiciously large. '
                                  'This may be because the light curve needs some degree of block integration. '
                                  'Failure to do a needed block integration allows point-to-point correlations caused by '
                                  'the camera integration to artificially induce non-physical correlated noise.')

        if self.sigmaA is None:
            self.sigmaA = self.sigmaB

        self.newRedrawMainPlot()

        self.locateEvent.setEnabled(True)
        self.markDzone.setEnabled(True)
        self.markRzone.setEnabled(True)
        self.singlePointDropButton.setEnabled(True)
        self.markEzone.setEnabled(not self.ne3NotInUseRadioButton.isChecked())

        self.minEventEdit.setEnabled(True)
        self.maxEventEdit.setEnabled(True)

    def processBaselineNoiseFromIterativeSolution(self, left, right):
        assert left >= self.left
        assert right <= self.right

        self.baselineXvals = []
        self.baselineYvals = []
        for i in range(left, right + 1):
            self.baselineXvals.append(i)
            self.baselineYvals.append(self.yValues[i])

        if not self.userDeterminedBaselineStats:
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
        self.calcFlashEdge.setEnabled(False)
        self.setDataLimits.setEnabled(False)
        self.doBlockIntegration.setEnabled(False)
        self.locateEvent.setEnabled(False)
        self.calcErrBars.setEnabled(False)
        self.fillExcelReportButton.setEnabled(False)
        self.startOver.setEnabled(False)
        self.markDzone.setEnabled(False)
        self.markRzone.setEnabled(False)
        self.singlePointDropButton.setEnabled(False)
        self.markEzone.setEnabled(False)
        # self.numSmoothPointsEdit.setEnabled(False)
        self.minEventEdit.setEnabled(False)
        self.maxEventEdit.setEnabled(False)
        self.writeBarPlots.setEnabled(False)
        self.writeCSVButton.setEnabled(False)

    # noinspection PyUnusedLocal
    def restart(self):

        self.userDeterminedBaselineStats = False
        self.userDeterminedEventStats = False
        self.userTrimInEffect = False

        self.selectedPoints = {}

        savedFlashEdges = self.flashEdges
        self.initializeVariablesThatDontDependOnAfile()
        self.flashEdges = savedFlashEdges
        self.disableAllButtons()

        self.firstPassPenumbralFit = True

        # self.lightCurveNumberLabel.setEnabled(True)
        # self.curveToAnalyzeSpinBox.setEnabled(True)
        # self.normLabel.setEnabled(True)

        if self.errBarWin:
            self.errBarWin.close()

        self.dataLen = len(self.yTimes)
        self.timeDelta, self.outliers, self.errRate = getTimeStepAndOutliers(self.yTimes)
        self.expDurEdit.setText(fp.to_precision(self.timeDelta, 6))

        self.fillTableViewOfData()

        # Enable the initial set of buttons (allowed operations)
        self.startOver.setEnabled(True)
        self.setDataLimits.setEnabled(True)

        self.markDzone.setEnabled(True)
        self.markRzone.setEnabled(True)
        self.singlePointDropButton.setEnabled(True)
        self.markEzone.setEnabled(not self.ne3NotInUseRadioButton.isChecked())
        self.locateEvent.setEnabled(True)
        self.minEventEdit.setEnabled(True)
        self.maxEventEdit.setEnabled(True)

        self.clearBaselineRegionsButton.setEnabled(False)
        self.calcStatsFromBaselineRegionsButton.setEnabled(False)

        # Reset the data plot so that all points are visible
        self.mainPlot.autoRange()

        # Show all data points as INCLUDED
        self.yStatus = [INCLUDED for _i in range(self.dataLen)]

        # Set the 'left' and 'right' edges of 'included' data to 'all'
        self.left = 0
        self.right = self.dataLen - 1

        self.minEventEdit.clear()
        self.maxEventEdit.clear()

        self.bkgndRegionLimits = []

        self.newRedrawMainPlot()
        self.mainPlot.autoRange()
        self.showMsg('*' * 20 + ' starting over ' + '*' * 20, color='blue')

    def drawSolution(self):
        def plot(x, y, pen):
            self.mainPlot.plot(x, y, pen=pen, symbol=None)

        def plotDcurve():
            # The units of self.timeDelta are seconds per entry, so the conversion in the next line
            # gets us x_values[] converted to entryNum units.
            x_values = self.underlyingLightcurveAns['time deltas'] / self.timeDelta
            x_values += D
            y_values = self.underlyingLightcurveAns['D curve']
            if self.underlyingLightcurveAns['star D'] is not None:
                z_values = self.underlyingLightcurveAns['star D']
            else:
                z_values = self.underlyingLightcurveAns['raw D']

            x_trimmed = []
            y_trimmed = []
            z_trimmed = []
            for i in range(x_values.size):
                if self.left <= x_values[i] <= max_x:
                    x_trimmed.append(x_values[i])
                    y_trimmed.append(y_values[i])
                    z_trimmed.append(z_values[i])

            # (150, 100, 100) is the brownish color we use to show the underlying lightcurve
            if self.showUnderlyingLightcurveCheckBox.isChecked():
                plot(x_trimmed, z_trimmed, pen=pg.mkPen((150, 100, 100), width=self.lineWidthSpinner.value()))

            if self.exponentialDtheoryPts is None:
                if not self.showCameraResponseCheckBox.isChecked():
                    return
                # Now overplot with the blue camera response curve
                plot(x_trimmed, y_trimmed, pen=pg.mkPen((0, 0, 255), width=self.lineWidthSpinner.value()))
                # Extend camera response to the left and right if necessary...
                if x_trimmed:
                    if x_trimmed[0] > self.left:
                        plot([self.left, x_trimmed[0]], [y_trimmed[0], y_trimmed[0]],
                             pen=pg.mkPen((0, 0, 255), width=self.lineWidthSpinner.value()))
                    if x_trimmed[-1] < max_x:
                        plot([x_trimmed[-1], max_x], [y_trimmed[-1], y_trimmed[-1]],
                             pen=pg.mkPen((0, 0, 255), width=self.lineWidthSpinner.value()))

        def plotRcurve():
            # The units of self.timeDelta are seconds per entry, so the conversion in the next line
            # gets us x_values[] converted to entryNum units.
            x_values = self.underlyingLightcurveAns['time deltas'] / self.timeDelta
            x_values += R
            y_values = self.underlyingLightcurveAns['R curve']
            if self.underlyingLightcurveAns['star R'] is not None:
                z_values = self.underlyingLightcurveAns['star R']
            else:
                z_values = self.underlyingLightcurveAns['raw R']

            x_trimmed = []
            y_trimmed = []
            z_trimmed = []
            for i in range(x_values.size):
                if min_x <= x_values[i] <= self.right:
                    x_trimmed.append(x_values[i])
                    y_trimmed.append(y_values[i])
                    z_trimmed.append(z_values[i])

            # (150, 100, 100) is the brownish color we use to show the underlying lightcurve
            if self.showUnderlyingLightcurveCheckBox.isChecked() and x_trimmed:
                plot(x_trimmed, z_trimmed, pen=pg.mkPen((150, 100, 100), width=self.lineWidthSpinner.value()))

            if self.exponentialRtheoryPts is None and x_trimmed:
                if not self.showCameraResponseCheckBox.isChecked():
                    return
                # Now overplot with the blue camera response curve
                plot(x_trimmed, y_trimmed, pen=pg.mkPen((0, 0, 255), width=self.lineWidthSpinner.value()))
                # Extend camera response to the left and right if necessary...
                if x_trimmed[0] > min_x:
                    plot([min_x, x_trimmed[0]], [y_trimmed[0], y_trimmed[0]],
                         pen=pg.mkPen((0, 0, 255), width=self.lineWidthSpinner.value()))

                if x_trimmed[-1] < self.right:
                    plot([x_trimmed[-1], self.right], [y_trimmed[-1], y_trimmed[-1]],
                         pen=pg.mkPen((0, 0, 255), width=self.lineWidthSpinner.value()))

                if x_trimmed:
                    pass

        def plotGeometricShadowAtD():
            if self.showEdgesCheckBox.isChecked() and self.exponentialDtheoryPts is None:
                pen = pg.mkPen(color=(255, 0, 0), style=QtCore.Qt.PenStyle.DashLine,
                               width=self.lineWidthSpinner.value())
                self.mainPlot.plot([D, D], [lo_int, hi_int], pen=pen, symbol=None)

        def plotGeometricShadowAtR():
            if self.showEdgesCheckBox.isChecked() and self.exponentialRtheoryPts is None:
                pen = pg.mkPen(color=(0, 200, 0), style=QtCore.Qt.PenStyle.DashLine,
                               width=self.lineWidthSpinner.value())
                self.mainPlot.plot([R, R], [lo_int, hi_int], pen=pen, symbol=None)

        hi_int = max(self.yValues[self.left:self.right])
        lo_int = min(self.yValues[self.left:self.right])

        if self.eventType == 'DandR':
            # if self.exponentialDtheoryPts is None:
            D = self.solution[0]
            R = self.solution[1]
            # else:
            #     D = self.exponentialDedge
            #     R = self.exponentialRedge
            #     D = self.solution[0]
            #     R = self.solution[1]

            max_x = min_x = (D + R) / 2.0

            plotDcurve()
            plotGeometricShadowAtD()
            plotRcurve()
            plotGeometricShadowAtR()

        elif self.eventType == 'Donly':
            # if self.exponentialDtheoryPts is None:
            D = self.solution[0]
            # else:
            #     D = self.exponentialDedge

            max_x = self.right
            plotDcurve()
            plotGeometricShadowAtD()

        elif self.eventType == 'Ronly':
            R = self.solution[1]

            min_x = self.left
            plotRcurve()
            plotGeometricShadowAtR()
        else:
            raise Exception('Unrecognized event type of |' + self.eventType + '|')

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
        # (150, 100, 100) is the brownish color we use to show the underlying lightcurve
        # def plot(x, y):
        #     self.mainPlot.plot(x, y, pen=pg.mkPen((150, 100, 100), width=2), symbol=None)

        def plotGeometricShadowAtD(d):
            if self.showErrBarsCheckBox.isChecked():
                pen = pg.mkPen(color=(255, 0, 0), style=QtCore.Qt.PenStyle.DotLine, width=self.lineWidthSpinner.value())
                self.mainPlot.plot([d, d], [lo_int, hi_int], pen=pen, symbol=None)

        def plotGeometricShadowAtR(r):
            if self.showErrBarsCheckBox.isChecked():
                pen = pg.mkPen(color=(0, 200, 0), style=QtCore.Qt.PenStyle.DotLine, width=self.lineWidthSpinner.value())
                self.mainPlot.plot([r, r], [lo_int, hi_int], pen=pen, symbol=None)

        if self.solution is None:
            return

        # Make shortened geometric shadow markers to distinguish the error bar versions from the central value
        hi_int = max(self.yValues[self.left:self.right])
        lo_int = min(self.yValues[self.left:self.right])
        delta_int = (hi_int - lo_int) * 0.1
        hi_int -= delta_int
        lo_int += delta_int

        if self.eventType == 'Donly':
            D = self.solution[0]
            Dright = D + self.plusD
            Dleft = D - self.minusD
            plotGeometricShadowAtD(Dright)
            plotGeometricShadowAtD(Dleft)
            return

        if self.eventType == 'Ronly':
            R = self.solution[1]
            Rright = R + self.plusR
            Rleft = R - self.minusR
            plotGeometricShadowAtR(Rright)
            plotGeometricShadowAtR(Rleft)
            return

        if self.eventType == 'DandR':
            R = self.solution[1]
            D = self.solution[0]

            Rright = R + self.plusR
            Rleft = R - self.minusR
            Dright = D + self.plusD
            Dleft = D - self.minusD
            plotGeometricShadowAtR(Rright)
            plotGeometricShadowAtR(Rleft)
            plotGeometricShadowAtD(Dright)
            plotGeometricShadowAtD(Dleft)
            return

    def newRedrawMainPlot(self):
        QtWidgets.QApplication.processEvents()
        self.reDrawMainPlot()

    def reDrawMainPlot(self):
        # QtWidgets.QApplication.processEvents()
        if self.right is not None:
            right = min(self.dataLen, self.right + 1)
        else:
            right = self.dataLen
        if self.left is None:
            left = 0
        else:
            left = self.left

        self.mainPlot.clear()

        if self.yValues is None:
            return

        if self.showTimestampErrors.checkState():
            self.illustrateTimestampOutliers()

        self.mainPlot.addItem(self.verticalCursor)

        self.targetIndex = 0
        # Find index of target lightcurve
        for i, checkBox in enumerate(self.targetCheckBoxes):
            if checkBox.isChecked():
                self.targetIndex = i
                break

        refIndex = None
        # Find index of reference selection (if any)
        for i, checkBox in enumerate(self.referenceCheckBoxes):
            if checkBox.isChecked():
                refIndex = i
                break

        self.mainPlot.plot(self.yValues + self.yOffsetSpinBoxes[self.targetIndex].value())
        self.mainPlot.getPlotItem().showAxis('bottom')
        self.mainPlot.getPlotItem().showAxis('left')

        dotSize = self.dotSizeSpinner.value()

        minY = np.min(self.yValues + self.yOffsetSpinBoxes[self.targetIndex].value())
        maxY = np.max(self.yValues + self.yOffsetSpinBoxes[self.targetIndex].value())

        try:
            # Plot the 'target' lightcurve
            x = [i for i in range(self.dataLen) if self.yStatus[i] == INCLUDED]
            y = [self.yValues[i] for i in range(self.dataLen) if self.yStatus[i] == INCLUDED]
            y = np.array(y) + self.yOffsetSpinBoxes[self.targetIndex].value()
            self.mainPlot.plot(x, y, pen=None, symbol='o',
                               symbolBrush=(0, 32, 255), symbolSize=dotSize)

            x = [i for i in range(self.dataLen) if self.yStatus[i] == BASELINE]
            y = [self.yValues[i] for i in range(self.dataLen) if self.yStatus[i] == BASELINE]
            y = np.array(y) + self.yOffsetSpinBoxes[self.targetIndex].value()
            self.mainPlot.plot(x, y, pen=None, symbol='o',
                               symbolBrush=(255, 150, 100), symbolSize=dotSize)

            x = [i for i in range(self.dataLen) if self.yStatus[i] == EVENT]
            y = [self.yValues[i] for i in range(self.dataLen) if self.yStatus[i] == EVENT]
            y = np.array(y) + self.yOffsetSpinBoxes[self.targetIndex].value()
            self.mainPlot.plot(x, y, pen=None, symbol='o',
                               symbolBrush=(155, 150, 100), symbolSize=dotSize)

            x = [i for i in range(self.dataLen) if self.yStatus[i] == SELECTED]
            y = [self.yValues[i] for i in range(self.dataLen) if self.yStatus[i] == SELECTED]
            y = np.array(y) + self.yOffsetSpinBoxes[self.targetIndex].value()
            self.mainPlot.plot(x, y, pen=None, symbol='o',
                               symbolBrush=(255, 0, 0), symbolSize=dotSize + 4)
        except IndexError:
            pass

        if self.exponentialDtheoryPts is not None:
            xVals = [self.exponentialDinitialX]
            for i in range(len(self.exponentialDtheoryPts) - 1):
                xVals.append(xVals[-1] + 1)
            self.mainPlot.plot(xVals, self.exponentialDtheoryPts, pen=None, symbol='o',
                               symbolBrush=(160, 128, 96), symbolSize=dotSize + 4)
            hi_int = max(self.yValues[self.left:self.right])
            lo_int = min(self.yValues[self.left:self.right])
            pen = pg.mkPen(color=(200, 0, 0), style=QtCore.Qt.PenStyle.DashLine, width=self.lineWidthSpinner.value())
            self.mainPlot.plot([self.solution[0], self.solution[0]], [lo_int, hi_int], pen=pen, symbol=None)

        if self.exponentialRtheoryPts is not None:
            xVals = [self.exponentialRinitialX]
            for i in range(len(self.exponentialRtheoryPts) - 1):
                xVals.append(xVals[-1] + 1)
            self.mainPlot.plot(xVals, self.exponentialRtheoryPts, pen=None, symbol='o',
                               symbolBrush=(160, 128, 96), symbolSize=dotSize + 4)
            hi_int = max(self.yValues[self.left:self.right])
            lo_int = min(self.yValues[self.left:self.right])
            pen = pg.mkPen(color=(0, 200, 0), style=QtCore.Qt.PenStyle.DashLine, width=self.lineWidthSpinner.value())
            self.mainPlot.plot([self.solution[1], self.solution[1]], [lo_int, hi_int], pen=pen, symbol=None)

        if len(self.yRefStar) == self.dataLen:
            if not self.skipNormalization and not self.suppressNormalization:
                # Update reference curve smoothing
                if self.smoothingIntervalSpinBox.value() > 0:
                    # Start with pristine (original) values for the curve being analyzed.
                    self.processTargetSelection(self.targetIndex, redraw=False)
                    self.newNormalize()
                else:
                    self.smoothSecondary = []
                    # Return to original values
                    self.processTargetSelection(self.targetIndex, redraw=False)
                    self.fillTableViewOfData()
                    self.normalized = False
                self.skipNormalization = True
                self.newRedrawMainPlot()
            else:
                self.skipNormalization = False
                self.suppressNormalization = False

            # Plot the normalization reference lightcurve
            if refIndex is not None and self.showCheckBoxes[refIndex].isChecked():
                minY = min(minY, np.min(self.yRefStar + self.yOffsetSpinBoxes[refIndex].value()))
                maxY = max(maxY, np.max(self.yRefStar + self.yOffsetSpinBoxes[refIndex].value()))
                xOffset = self.xOffsetSpinBoxes[refIndex].value()
                x = [i + xOffset for i in range(self.left, self.right + 1)]
                y = [self.yRefStar[i] for i in range(self.left, self.right + 1)]
                y = np.array(y) + self.yOffsetSpinBoxes[refIndex].value()
                self.mainPlot.plot(x, y)
                self.mainPlot.plot(x, y, pen=None, symbol='o',
                                   symbolBrush=(0, 255, 0), symbolSize=dotSize)

                # Plot the continuous smoothed curve through the reference lightcurve
                if len(self.smoothSecondary) > 0:
                    self.mainPlot.plot(x, self.smoothSecondary + self.yOffsetSpinBoxes[refIndex].value(),
                                       pen=pg.mkPen((100, 100, 100), width=4), symbol=None)

        dotColors = [(255, 0, 0), (160, 32, 255), (80, 208, 255), (96, 255, 128),
                     (255, 224, 32), (255, 160, 16), (160, 128, 96), (64, 64, 64), (255, 208, 160), (0, 128, 0)]

        for i, checkBox in enumerate(self.showCheckBoxes):
            if checkBox.isChecked():
                if self.targetCheckBoxes[i].isChecked():
                    continue
                if self.referenceCheckBoxes[i].isChecked():
                    continue
                if i == 0:
                    self.mainPlot.plot(self.LC1 + self.yOffsetSpinBoxes[0].value())
                    minY = min(minY, np.min(self.LC1) + self.yOffsetSpinBoxes[0].value())
                    maxY = max(maxY, np.max(self.LC1) + self.yOffsetSpinBoxes[0].value())
                    x = [i for i in range(left, right)]
                    y = [self.LC1[i] for i in range(left, right)]
                    y = np.array(y) + self.yOffsetSpinBoxes[0].value()
                    self.mainPlot.plot(x, y, pen=None, symbol='o',
                                       symbolBrush=dotColors[i], symbolSize=dotSize)
                elif i == 1:
                    self.mainPlot.plot(self.LC2 + self.yOffsetSpinBoxes[1].value())
                    minY = min(minY, np.min(self.LC2) + self.yOffsetSpinBoxes[1].value())
                    maxY = max(maxY, np.max(self.LC2) + self.yOffsetSpinBoxes[1].value())
                    x = [i for i in range(left, right)]
                    y = [self.LC2[i] for i in range(left, right)]
                    y = np.array(y) + self.yOffsetSpinBoxes[1].value()
                    self.mainPlot.plot(x, y, pen=None, symbol='o',
                                       symbolBrush=dotColors[i], symbolSize=dotSize)
                elif i == 2:
                    self.mainPlot.plot(self.LC3 + self.yOffsetSpinBoxes[2].value())
                    minY = min(minY, np.min(self.LC3) + self.yOffsetSpinBoxes[2].value())
                    maxY = max(maxY, np.max(self.LC3) + self.yOffsetSpinBoxes[2].value())
                    x = [i for i in range(left, right)]
                    y = [self.LC3[i] for i in range(left, right)]
                    y = np.array(y) + self.yOffsetSpinBoxes[2].value()
                    self.mainPlot.plot(x, y, pen=None, symbol='o',
                                       symbolBrush=dotColors[i], symbolSize=dotSize)
                elif i == 3:
                    self.mainPlot.plot(self.LC4 + self.yOffsetSpinBoxes[3].value())
                    minY = min(minY, np.min(self.LC4) + self.yOffsetSpinBoxes[3].value())
                    maxY = max(maxY, np.max(self.LC4) + self.yOffsetSpinBoxes[3].value())
                    x = [i for i in range(left, right)]
                    y = [self.LC4[i] for i in range(left, right)]
                    y = np.array(y) + self.yOffsetSpinBoxes[3].value()
                    self.mainPlot.plot(x, y, pen=None, symbol='o',
                                       symbolBrush=dotColors[i], symbolSize=dotSize)
                else:
                    self.mainPlot.plot(self.extra[i - 4] + self.yOffsetSpinBoxes[i].value())
                    minY = min(minY, np.min(self.extra[i - 4] + self.yOffsetSpinBoxes[i].value()))
                    maxY = max(maxY, np.max(self.extra[i - 4] + self.yOffsetSpinBoxes[i].value()))
                    x = [i for i in range(left, right)]
                    y = [self.extra[i - 4][j] for j in range(left, right)]
                    y = np.array(y) + self.yOffsetSpinBoxes[i].value()
                    self.mainPlot.plot(x, y, pen=None, symbol='o',
                                       symbolBrush=dotColors[i], symbolSize=dotSize)

        self.mainPlot.setYRange(minY, maxY)

        if self.dRegion is not None:
            self.mainPlot.addItem(self.dRegion)
        if self.rRegion is not None:
            self.mainPlot.addItem(self.rRegion)
        if self.bkgndRegions is not []:
            for region in self.bkgndRegions:
                self.mainPlot.addItem(region)

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
            self.userTrimInEffect = True
        else:
            # self.showInfo('All points will be selected (because no trim points specified)')
            self.showMsg('All data points were selected')
            self.left = 0
            self.right = self.dataLen - 1

        for i in range(0, self.left):
            self.yStatus[i] = EXCLUDED
        for i in range(min(self.dataLen, self.right + 1), self.dataLen):
            self.yStatus[i] = EXCLUDED
        for i in range(self.left, min(self.dataLen, self.right + 1)):
            self.yStatus[i] = INCLUDED

        if len(self.smoothSecondary) > 0:
            self.smoothSecondary = []
            if self.smoothingIntervalSpinBox.value() > 0:
                # Start with pristine (original) values for the curve being analyzed.
                self.processTargetSelection(self.targetIndex, redraw=False)
                self.newNormalize()

        self.selectedPoints = {}
        self.newRedrawMainPlot()
        self.doBlockIntegration.setEnabled(False)
        self.mainPlot.autoRange()


# aTODO Remove this test code
# def my_trace_func(frame, event, arg):
#     if not event == 'exception':
#         return
#     print(f'event: {event}\n')
#     print(f'  arg: {arg}\n')
#     print(f'frame: {frame}')
#     return my_trace_func


def main(csv_file_path=None):
    # csv_file_path gets filled in by PyMovie

    os.environ['QT_MAC_WANTS_LAYER'] = '1'  # This line needed when Mac updated to Big Sur

    import traceback
    app = PyQt5.QtWidgets.QApplication(sys.argv)

    print(f'PyOTE Version: {version.version()}')

    if sys.platform == 'linux':
        PyQt5.QtWidgets.QApplication.setStyle('macintosh')
        print(f'os: Linux')
    elif sys.platform == 'darwin':
        PyQt5.QtWidgets.QApplication.setStyle('macintosh')
        print(f'os: MacOS')
    else:
        print(f'os: Windows')
        PyQt5.QtWidgets.QApplication.setStyle('windows')
        app.setStyleSheet("QLabel, QPushButton, QToolButton, QCheckBox, "
                          "QRadioButton, QLineEdit , QTextEdit {font-size: 8pt}")

    # aTODO Remove this test code
    # sys.settrace(my_trace_func)

    # Save the current/proper sys.excepthook object
    saved_excepthook = sys.excepthook

    def exception_hook(exctype, value, tb):
        print('')
        print('=' * 30)
        print(value)
        print('=' * 30)
        print('')

        traceback.print_tb(tb)
        # Call the usual exception processor
        saved_excepthook(exctype, value, tb)
        # Exit if you prefer...
        # sys.exit(1)

    sys.excepthook = exception_hook

    form = SimplePlot(csv_file_path)
    form.show()
    print(app.exec_())
    quit()


if __name__ == '__main__':
    main()
