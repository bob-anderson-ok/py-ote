
"""
Created on Thu May 25 12:26:10 2017

@author: Bob Anderson
"""

import numpy as np
from PyQt5.QtWidgets import QDialog


def convertTimeStringToTime(timeStr):
    """
    We assume the timeStr has the format  [hh:mm:ss.ssss]
    Returns -1.0 if conversion fails, otherwise time as a float
    """
    # noinspection PyBroadException
    try:
        if timeStr[0] != '[':
            return -1.0
        if timeStr[-1] != ']':
            return -1.0
        clippedStr = timeStr[1:-1]  # remove [] from input
        parts = clippedStr.split(':')
        hourSecs = float(parts[0]) * 60 * 60
        minSecs = float(parts[1]) * 60
        secs = float(parts[2])
        return hourSecs + minSecs + secs
    except Exception:
        return -1.0
    

def convertTimeToTimeString(time):
    """
    Convert a time (float) to string --- [hh:mm:ss.ssss]
    """
    hours = int(time // (60 * 60))
    if hours > 23 or hours < 0:
        return ""
    time -= hours * 60 * 60
    minutes = int(time // 60)
    time -= minutes * 60
    seconds = time
    timeStr = '[{:02d}:{:02d}:{:07.4f}]'.format(hours, minutes, seconds)
    return timeStr


def improveTimeStep(outliers, deltaTime):
    validIntervals = [deltaTime[i] for i in range(len(deltaTime)) if i not in outliers]
    # Under some circumstances, OCR can be so bad that no valid intervals can
    # be found.  In that case, we return a timeStep of 0.0 and let the
    # program respond to this invlid value at a higher level.
    if validIntervals:
        betterTimeStep = np.mean(validIntervals)
    else:
        betterTimeStep = 0.0
    return betterTimeStep


def getTimeStepAndOutliers(timestamps, tolerance=0.1):
    """
    The default tolerance is set so that it will work for Tangra
    which only reports 3 of the 4 digits of VTI fractional seconds
    """
    time = [convertTimeStringToTime(item) for item in timestamps]
    deltaTime = np.diff(time)
    timeStep = np.median(deltaTime)
    high = timeStep * (1.0 + tolerance)
    low = timeStep * (1.0 - tolerance)
    outlierIndices = [i for i, dt in enumerate(deltaTime) if dt < low or dt > high]
    numEntries = len(timestamps)
    numOutliers = len(outlierIndices)
    timestampErrorRate = numOutliers / numEntries
    
    improvedTimeStep = improveTimeStep(outlierIndices, deltaTime)
    
    return improvedTimeStep, outlierIndices, timestampErrorRate


def isFrameNumberInData(frameToTest, frame):
    first = float(frame[0])
    last = float(frame[-1])
    return first <= frameToTest <= last


savedF1 = savedHH1 = savedMM1 = savedSS1 = ''
savedF2 = savedHH2 = savedMM2 = savedSS2 = ''
savedNTSCButton = True
savedPALButton = False
savedCustomButton = False
savedCustomText = ''


def saveAllDialogEntries(dialog):
    global savedF1, savedHH1, savedMM1, savedSS1
    global savedF2, savedHH2, savedMM2, savedSS2
    global savedNTSCButton, savedPALButton, savedCustomButton
    global savedCustomText

    savedF1 = dialog.frameNum1.text()
    savedHH1 = dialog.hh1.text()
    savedMM1 = dialog.mm1.text()
    savedSS1 = dialog.ss1.text()

    savedF2 = dialog.frameNum2.text()
    savedHH2 = dialog.hh2.text()
    savedMM2 = dialog.mm2.text()
    savedSS2 = dialog.ss2.text()

    savedNTSCButton = dialog.radioButtonNTSC.isChecked()
    savedPALButton = dialog.radioButtonPAL.isChecked()
    savedCustomButton = dialog.radioButtonCustom.isChecked()

    savedCustomText = dialog.frameDeltaTime.text()


# noinspection PyBroadException,PyUnusedLocal
def manualTimeStampEntry(frame, dialog, flashFrames=None):
    time = []  # Output --- to be computed from timestamp data entered in dialog
    dataEntered = 'error in data entry'

    if flashFrames:
        dialog.frameNum1.setText(str(flashFrames[0]))
        if len(flashFrames) > 1:
            dialog.frameNum2.setText(str(flashFrames[1]))
    else:
        dialog.frameNum1.setText(savedF1)
        dialog.frameNum2.setText(savedF2)

    dialog.hh1.setText(savedHH1)
    dialog.mm1.setText(savedMM1)
    dialog.ss1.setText(savedSS1)

    dialog.hh2.setText(savedHH2)
    dialog.mm2.setText(savedMM2)
    dialog.ss2.setText(savedSS2)

    if savedNTSCButton:
        dialog.radioButtonNTSC.setChecked(savedNTSCButton)
    if savedPALButton:
        dialog.radioButtonPAL.setChecked(savedPALButton)
    if savedCustomButton:
        dialog.radioButtonCustom.setChecked(savedCustomButton)

    dialog.frameDeltaTime.setText(savedCustomText)

    result = dialog.exec_()

    saveAllDialogEntries(dialog)

    if result == QDialog.Accepted:
        nf = ef = frameNum1 = frameNum2 = frameDeltaTime = -1
        hh1 = mm1 = ss1 = hh2 = mm2 = ss2 = -1
        try:
            frameNum1 = float(dialog.frameNum1.text())
        except Exception:
            return '"' + dialog.frameNum1.text() + '" is invalid as frame number input', \
                   time, dataEntered, nf, ef
        try:
            hh1 = int(dialog.hh1.text())
        except Exception:
            return '"' + dialog.hh1.text() + '" is invalid as hh input', \
                   time, dataEntered, nf, ef
        try:
            mm1 = int(dialog.mm1.text())
        except Exception:
            return '"' + dialog.mm1.text() + '" is invalid as mm input', \
                   time, dataEntered, nf, ef
        try:
            ss1 = float(dialog.ss1.text())
        except Exception:
            return '"' + dialog.ss1.text() + '" is invalid as ss.ssss input', \
                   time, dataEntered, nf, ef

        frameNum2text = dialog.frameNum2.text()
        if frameNum2text != '':
            try:
                frameNum2 = float(dialog.frameNum2.text())
            except Exception:
                return '"' + dialog.frameNum2.text() + '" is invalid as frame number input', \
                       time, dataEntered, nf, ef
            try:
                hh2 = int(dialog.hh2.text())
            except Exception:
                return '"' + dialog.hh2.text() + '" is invalid as hh input', \
                       time, dataEntered, nf, ef
            try:
                mm2 = int(dialog.mm2.text())
            except Exception:
                return '"' + dialog.mm2.text() + '" is invalid as mm input', \
                       time, dataEntered, nf, ef
            try:
                ss2 = float(dialog.ss2.text())
            except Exception:
                return '"' + dialog.ss2.text() + '" is invalid as ss.ssss input', \
                       time, dataEntered, nf, ef
        else:
            return 'Both entries must be supplied', \
                   time, dataEntered, nf, ef

        if dialog.frameDeltaTime.text() and not dialog.radioButtonCustom.isChecked():
            return 'You have something entered in custom frame time edit box ' + \
                   'but have not clicked on the radio button to enable use of this value.\n\n' + \
                   'Please clarify your intentions.', \
                   time, dataEntered, nf, ef

        if dialog.radioButtonNTSC.isChecked():
            expectedFrameDeltaTime = 1.001 / 30.0
        elif dialog.radioButtonPAL.isChecked():
            expectedFrameDeltaTime = 1.000 / 25.0
        else:
            try:
                expectedFrameDeltaTime = float(eval(dialog.frameDeltaTime.text(), {}, {}))
            except Exception:
                return '"' + dialog.frameDeltaTime.text() + '" is invalid as timeDelta', \
                       time, dataEntered, nf, ef
            if not isinstance(expectedFrameDeltaTime, float):
                return '"' + dialog.frameDeltaTime.text() + '" is invalid as timeDelta --- not a float', \
                       time, dataEntered, nf, ef
            if not expectedFrameDeltaTime > 0.0:
                return '"' + dialog.frameDeltaTime.text() + '" is invalid as timeDelta --- not > 0', \
                       time, dataEntered, nf, ef

        # Validate data entries

        if frameNum2 <= frameNum1:
            return 'early frame num must be less than late frame num', \
                   time, dataEntered, nf, ef
        if frameNum1 < 0 or hh1 < 0 or mm1 < 0 or ss1 < 0:
            return 'Negative values of frame num are invalid.', \
                   time, dataEntered, nf, ef
        if frameNum2 < 0 or hh2 < 0 or mm2 < 0 or ss2 < 0:
            return 'Negative values of frame num are invalid.', \
                   time, dataEntered, nf, ef
        if not isFrameNumberInData(frameNum1, frame):
            return 'early frame num is not valid: could not be found in the file data', \
                   time, dataEntered, nf, ef
        if not isFrameNumberInData(frameNum2, frame):
            return 'late frame num is not valid: could not be found in the file data', \
                   time, dataEntered, nf, ef

        timeStr1 = 'Manual timestamp info: @ frame {:0.2f} [{:02d}:{:02d}:{:07.4f}]'.format(frameNum1, hh1, mm1, ss1)
        timeStr2 = ' --- @ frame {:0.2f} [{:02d}:{:02d}:{:07.4f}]'.format(frameNum2, hh2, mm2, ss2)
        dataEntered = timeStr1 + timeStr2

        numFramesInSpan = frameNum2 - frameNum1
        t1 = hh1 * 3600 + mm1 * 60 + ss1
        t2 = hh2 * 3600 + mm2 * 60 + ss2
        # Handle midnight crossing
        if t2 < t1:
            t2 += 3600.0 * 24.0
        expectedFramesInSpan = (t2 - t1) / expectedFrameDeltaTime

        # Time to compute timestamps.
        calculatedFrameDeltaTime = (t2 - t1) / (frameNum2 - frameNum1)
        time = timestampsFromFrameDelta(t1, frameNum1, calculatedFrameDeltaTime, frame)

        return 'ok', time, dataEntered, numFramesInSpan, expectedFramesInSpan
    else:
        return 'cancelled', time, dataEntered, -1, -1


def timestampsFromFrameDelta(t1, frameNum1, frameDeltaTime, frame):
    time = []
    secondsIn24Hours = 3600 * 24.0
    for item in frame:
        frameNum = float(item)
        timeOfFrame = t1 + (frameNum - frameNum1) * frameDeltaTime
        if timeOfFrame >= secondsIn24Hours:
            timeOfFrame -= secondsIn24Hours
        elif timeOfFrame < 0:
            timeOfFrame += secondsIn24Hours
        time.append(convertTimeToTimeString(timeOfFrame))

    return time
