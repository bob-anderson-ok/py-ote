#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 23 07:24:21 2017

@author: Bob Anderson
"""

import os

tangraNeedsBackgroundSubtraction = True

pymovieSignalColumnCount = 0

pymovieDataColumns = []

pymovieColumnNamePrefix = 'signal'


def readLightCurve(filepath, pymovieColumnType='signal'):
    """
    Reads the intensities and timestamps from Limovie,
    Tangra, PYOTE, or R-OTE csv files.  (PYOTE and R-OTE file formats are equal)
    """
    global pymovieColumnNamePrefix

    pymovieColumnNamePrefix = pymovieColumnType
    if fileCanBeOpened(filepath):
        
        readOk, errMsg, frame, time, value, ref1, ref2, ref3, extra, aperture_names, headers = readAs(filepath)
        if readOk:
            return frame, time, value, ref1, ref2, ref3, extra, aperture_names, headers
        else:
            raise Exception(errMsg)
            
    else:
        raise Exception('File could not be opened')


def fileCanBeOpened(file):
    return os.path.exists(file) and os.path.isfile(file)
                   

def getFileKind(file):
    fileobject = open(file)
    with fileobject:
        line = fileobject.readline()
        if 'Tangra' in line:
            return 'Tangra'
        elif 'Limovie' in line:
            return 'Limovie'
        elif 'PyMovie' in line:
            return'PyMovie'
        elif 'R-OTE' in line or line[0] == '#':  # Matches PyOTE and PyMovie files too!
            return 'R-OTE'
        elif 'RAW' in line:
            return 'raw'
        else:
            return '???'


# noinspection PyUnusedLocal
def tangraParser(line, frame, time, value, ref1, ref2, ref3, extra):
    """
    We only accept Tangra files that have been formatted
    according to the AOTA default which is ---
       UT time field formatted as  [hh:mm:ss.sss]
       We detect the state of background subtraction (either done or needed)
       An example data line:  11,[16:00:14.183],2837.8,100.0,4097.32,200.0
    """
    part = line.split(',')
    if len(part) < 2:
        raise Exception(line + " :is an invalid Tangra file entry.")
    else:
        frame.append(part[0])
        time.append(part[1])

    try:
        for item in part:
            if item == '':
                raise Exception(line + " :cannot be parsed.  Are there empty fields in data lines? Fix them all!")

        if tangraNeedsBackgroundSubtraction:
            value.append(str(float(part[2]) - float(part[3])))
            if len(part) >= 6:
                if part[4]:
                    ref1.append(str(float(part[4]) - float(part[5])))
            if len(part) >= 8:
                if part[6]:
                    ref2.append(str(float(part[6]) - float(part[7])))
            if len(part) >= 10:
                if part[8]:
                    ref3.append(str(float(part[8]) - float(part[9])))
        else:
            value.append(part[2])
            if len(part) >= 4:
                if part[3]:
                    ref1.append(part[3])
            if len(part) >= 5:
                if part[4]:
                    ref2.append(part[4])
            if len(part) >= 6:
                if part[5]:
                    ref3.append(part[5])
    except ValueError:
        raise Exception(line + " :cannot be parsed.  Are there empty fields?")


# noinspection PyUnusedLocal
def limovieParser(line, frame, time, value, ref1, ref2, ref3, extra):
    """
    Limovie sample line ---
        3.5,21381195,21381200,22,27,43.0000,,,,,2737.8,3897.32 ...
    """
    part = line.split(',')
    frame.append(part[0])
    time.append('[' + part[3] + ':' + part[4] + ':' + part[5] + ']')
    value.append(part[10])
    if part[11]:
        ref1.append(part[11])
    if part[12]:
        ref2.append(part[12])


# noinspection PyUnusedLocal
def roteParser(line, frame, time, value, ref1, ref2, ref3, extra):
    """
    R-OTE sample line ---
        1.00,[17:25:39.3415],2737.8,3897.32,675.3,892.12
    """
    part = line.split(',')
    frame.append(part[0])
    time.append(part[1])
    value.append(part[2])
    if len(part) >= 4:
        if part[3]:
            ref1.append(part[3])
    if len(part) >= 5:
        if part[4]:
            ref2.append(part[4])
    if len(part) >= 6:
        if part[5]:
            ref3.append(part[5])


# noinspection PyUnusedLocal
def pymovieParser(line, frame, time, value, ref1, ref2, ref3, extra):
    """
    R-OTE sample line ---
        1.00,[17:25:39.3415],2737.8,3897.32,675.3,892.12
    """
    global pymovieDataColumns

    part = line.split(',')
    frame.append(part[0])
    time.append(part[1])
    dataColumnIndex = 0
    partNum = pymovieDataColumns[dataColumnIndex]
    value.append(part[partNum])
    dataColumnIndex += 1

    if len(part) >= 4 and pymovieSignalColumnCount >= 2:
        partNum = pymovieDataColumns[dataColumnIndex]
        if part[partNum]:
            ref1.append(part[partNum])
        dataColumnIndex += 1
    if len(part) >= 5 and pymovieSignalColumnCount >= 3:
        partNum = pymovieDataColumns[dataColumnIndex]
        if part[partNum]:
            ref2.append(part[partNum])
        dataColumnIndex += 1
    if len(part) >= 6 and pymovieSignalColumnCount >= 4:
        partNum = pymovieDataColumns[dataColumnIndex]
        if part[partNum]:
            ref3.append(part[partNum])
        dataColumnIndex += 1
    if pymovieSignalColumnCount > 4:
        for i in range(6, pymovieSignalColumnCount + 2):
            partNum = pymovieDataColumns[dataColumnIndex]
            if len(part) > i and part[partNum]:
                extra[i-6].append(part[partNum])
            dataColumnIndex += 1


# noinspection PyUnusedLocal
def rawParser(line, frame, time, value, secondary, ref2, ref3, extra):
    part = part = line.split(',')
    frame.append(part[0])
    time.append(part[1])
    value.append(part[2])


def readAs(file):
    global tangraNeedsBackgroundSubtraction
    global pymovieSignalColumnCount, pymovieDataColumns, pymovieColumnNamePrefix

    kind = getFileKind(file)
    
    fileobject = open(file)
    frame = []
    time = []
    value = []
    ref1 = []
    ref2 = []
    ref3 = []
    extra = []
    headers = []
    aperture_names = []
    
    if kind == 'Tangra':
        colHeaderKey = 'FrameNo'
        parser = tangraParser
    elif kind == 'R-OTE':  # PYOTE uses same colHeaderKey
        colHeaderKey = 'FrameNum'
        parser = roteParser
    elif kind == 'PyMovie':
        colHeaderKey = 'FrameNum'
        pymovieSignalColumnCount = 0
        parser = pymovieParser
    elif kind == 'Limovie':
        colHeaderKey = 'No.'
        parser = limovieParser
    elif kind == 'raw':
        colHeaderKey = 'RAW'
        parser = rawParser
    else:
        return False, 'invalid file "kind"', frame, time, value, ref1, ref2, ref3, extra, aperture_names, headers

    with fileobject:
        while True:
            line = fileobject.readline()
            if line:
                if colHeaderKey in line:

                    if kind == 'Tangra':
                        if line.find('SignalMinusBackground') > 0:
                            tangraNeedsBackgroundSubtraction = False
                        else:
                            tangraNeedsBackgroundSubtraction = True

                    if kind == 'PyMovie':
                        # We need to count the number of times 'signal" starts
                        # a column header
                        line = line.rstrip()  # Get rid of possible trailing new line \n
                        parts = line.split(',')
                        pymovieDataColumns = []
                        columnIndex = 0
                        for part in parts:
                            if part.startswith(pymovieColumnNamePrefix):
                                pymovieSignalColumnCount += 1
                                aperture_names.append(part.split('-')[1])
                                pymovieDataColumns.append(columnIndex)
                            columnIndex += 1
                            # If there are more than 4 columns of 'signals', we need to setup
                            # extra to hold those columns
                        for i in range(5, pymovieSignalColumnCount+1):
                            extra.append([])

                    while True:
                        line = fileobject.readline()
                        if line:
                            # noinspection PyBroadException
                            try:
                                parser(line, frame, time, value, ref1, ref2, ref3, extra)
                            except Exception as e:
                                return False, str(e), frame, time, value, \
                                       ref1, ref2, ref3, extra, aperture_names, headers
                        else:
                            return True, kind, frame, time, value, ref1, ref2, ref3, extra, aperture_names, headers
                headers.append(line[:-1])
            else:
                return (False, colHeaderKey + ' not found as first column header', 
                        [], [], [], [], [], [], [], aperture_names, headers)
