#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 25 12:26:10 2017

@author: Bob Anderson
"""

import numpy as np


def convertTimeStringToTime(timeStr):
    """
    We assume the timeStr has the format  [hh:mm:ss.ssss]
    Returns -1.0 if conversion fails, otherwise time as a float
    """
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
