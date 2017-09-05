#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
    except:
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
    return np.mean(validIntervals)


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
    ans = False
    for item in frame:
        if float(item) == frameToTest:
            ans = True
            break
    return ans


# noinspection PyBroadException,PyUnusedLocal
def manualTimeStampEntry(frame, dialog):
    time = []  # Output --- to be computed from timestamp data entered in dialog
    dataEntered = 'error in data entry'

    result = dialog.exec_()

    if result == QDialog.Accepted:
        frameNum1 = frameNum2 = frameDeltaTime = -1
        hh1 = mm1 = ss1 = hh2 = mm2 = ss2 = -1
        try:
            frameNum1 = float(dialog.frameNum1.text())
        except:
            return '"' + dialog.frameNum1.text() + '" is invalid as frame number input', time, dataEntered
        try:
            hh1 = int(dialog.hh1.text())
        except:
            return '"' + dialog.hh1.text() + '" is invalid as hh input', time, dataEntered
        try:
            mm1 = int(dialog.mm1.text())
        except:
            return '"' + dialog.mm1.text() + '" is invalid as mm input', time, dataEntered
        try:
            ss1 = float(dialog.ss1.text())
        except:
            return '"' + dialog.ss1.text() + '" is invalid as ss.ssss input', time, dataEntered

        frameNum2text = dialog.frameNum2.text()
        if frameNum2text != '':
            try:
                frameNum2 = float(dialog.frameNum2.text())
            except:
                return '"' + dialog.frameNum2.text() + '" is invalid as frame number input', time, dataEntered
            try:
                hh2 = int(dialog.hh2.text())
            except:
                return '"' + dialog.hh2.text() + '" is invalid as hh input', time, dataEntered
            try:
                mm2 = int(dialog.mm2.text())
            except:
                return '"' + dialog.mm2.text() + '" is invalid as mm input', time, dataEntered
            try:
                ss2 = float(dialog.ss2.text())
            except:
                return '"' + dialog.ss2.text() + '" is invalid as ss.ssss input', time, dataEntered
        else:
            try:
                frameDeltaTime = float(dialog.frameDeltaTime.text())
            except:
                return '"' + dialog.frameDeltaTime.text() + '" is invalid as frameDeltaTime', time, dataEntered

        # Validate data entries
        if frameNum2 != -1:
            if frameNum2 <= frameNum1:
                return 'frame 1 must be less than frame 2', time, dataEntered
            if frameNum1 < 0 or hh1 < 0 or mm1 < 0 or ss1 < 0:
                return 'Negative values in are invalid.', time, dataEntered
            if frameNum2 < 0 or hh2 < 0 or mm2 < 0 or ss2 < 0:
                return 'Negative values in are invalid.', time, dataEntered
            if not isFrameNumberInData(frameNum1, frame):
                return 'frame 1 is not valid: could not be found in the file data', time, dataEntered
            if not isFrameNumberInData(frameNum2, frame):
                return 'frame 2 is not valid: could not be found in the file data', time, dataEntered
        else:
            if frameNum1 < 0 or hh1 < 0 or mm1 < 0 or ss1 < 0:
                return 'Negative values in are invalid.', time, dataEntered
            if not isFrameNumberInData(frameNum1, frame):
                return 'frame 1 is not valid: could not be found in the file data', time, dataEntered
            if frameDeltaTime <= 0:
                return 'frame delta time must be > 0.00', time, dataEntered

        timeStr1 = 'Manual timestamp info: @ frame {:0.1f} [{:02d}:{:02d}:{:07.4f}]'.format(frameNum1, hh1, mm1, ss1)
        if frameNum2 != -1:
            timeStr2 = ' --- @ frame {:0.1f} [{:02d}:{:02d}:{:07.4f}]'.format(frameNum2, hh2, mm2, ss2)
        else:
            timeStr2 = ' --- frameDeltaTime = {:f}'.format(frameDeltaTime)
        dataEntered = timeStr1 + timeStr2

        # Time to compute timestamps.

        if frameDeltaTime > 0:
            t1 = hh1 * 3600 + mm1 * 60 + ss1
            time = timestampsFromFrameDelta(t1, frameNum1, frameDeltaTime, frame)
        else:
            t1 = hh1 * 3600 + mm1 * 60 + ss1
            t2 = hh2 * 3600 + mm2 * 60 + ss2
            # Handle midnight crossing
            if t2 < t1:
                t2 += 3600.0 * 24.0
            frameDeltaTime = (t2 - t1) / (frameNum2 - frameNum1)
            time = timestampsFromFrameDelta(t1, frameNum1, frameDeltaTime, frame)

        return 'ok', time, dataEntered
    else:
        return 'ok', time, 'Manual timestamp entry was cancelled.'


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
