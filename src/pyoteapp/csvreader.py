#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 23 07:24:21 2017

@author: Bob Anderson
"""

import os

tangraNeedsBackgroundSubtraction = None

pymovieSignalColumnCount = 0

pymovieTotalDataDict = {}

pymovieDataColumns = []

pymovieColumnNames = []

pymovieColumnNamePrefix = 'signal'

kind = ''


def readLightCurve(filepath, pymovieColumnType='signal', pymovieDict={}):  # noqa
    """
    Reads the intensities and timestamps from Limovie,
    Tangra, PYOTE, or R-OTE csv files.  (PYOTE and R-OTE file formats are equal)
    """
    global pymovieColumnNamePrefix, pymovieTotalDataDict, tangraNeedsBackgroundSubtraction

    pymovieTotalDataDict = pymovieDict
    tangraNeedsBackgroundSubtraction = None

    pymovieColumnNamePrefix = pymovieColumnType
    if fileCanBeOpened(filepath):
        
        readOk, errMsg, frame, time, value, ref1, ref2, ref3, extra, aperture_names, headers = readAs(filepath)
        if readOk:
            return frame, time, value, ref1, ref2, ref3, extra, aperture_names, headers
        else:
            raise ValueError(errMsg)
            
    else:
        raise ValueError('File could not be opened')


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
       UT time field formatted as  [hh:mm:ss.sssssss]
       We detect the state of background subtraction (either done or needed)
       An example data line:  11,[16:00:14.183],2837.8,100.0,4097.32,200.0
    """
    part = line.split(',')

    # New policy for dealing with empty data fields: drop the whole line/frame if timeInfo or LC1 field is empty,
    # otherwise substitute 0
    for i, item in enumerate(part):
        if item == '':
            if i == 1:
                # timeInfo is missing - this is fatal - drop the entire line
                return
            if i == 2:
                # empty field for LC1 (assumed to be target) - treat as dropped frame by dropping the whole line
                return

    if len(part) < 2:
        raise Exception(line + " :is an invalid Tangra file entry.")
    else:
        frame.append(part[0])
        time.append(part[1])

    try:
        # This was removed when we decided to deal with empty fields by dropping the whole frame (see new policy above)
        # for item in part:
        #     if item == '':
        #         raise Exception(line + " :cannot be parsed.  Are there empty fields in data lines? Fix them all!")

        # Deal with empty fields other than timeInfo and LC1 by replacing with '0'
        for i, item in enumerate(part):
            if item == '':
                part[i] = '0'

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
    PyOTE sample line ---
        1.00,[17:25:39.3415],2737.8,3897.32,675.3,892.12
    """
    global pymovieDataColumns, pymovieTotalDataDict

    part = line.split(',')
    part[-1] = part[-1][:-1]  # Remove \n from last column data
    if not pymovieTotalDataDict == {}:
        for i in range(len(part)):
            if i == 1:
                pymovieTotalDataDict[pymovieColumnNames[i]].append(part[i])
            else:
                try:
                    convertedValue = float(part[i])
                    pymovieTotalDataDict[pymovieColumnNames[i]].append(convertedValue)
                except ValueError as e:
                    print(f'At frame {part[0]}: {part[i]} caused exception: {e}')
                    pymovieTotalDataDict[pymovieColumnNames[i]].append(-0.0)

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
    global pymovieTotalDataDict, pymovieColumnNames, kind

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
    pymovieColumnNames = []
    
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
        return False, 'invalid file - not a known light-curve csv file type', \
            frame, time, value, ref1, ref2, ref3, extra, aperture_names, headers

    with fileobject:
        while True:
            line = fileobject.readline()
            # if line.startswith('#'):
            if 'SignalMinusBackground' in line:
                tangraNeedsBackgroundSubtraction = False
            if line:
                if colHeaderKey in line or 'BinNo' in line:  # To deal with Tangra binned files

                    if kind == 'Tangra':
                        if tangraNeedsBackgroundSubtraction is None:
                            if line.find('SignalMinusBackground') > 0:
                                tangraNeedsBackgroundSubtraction = False
                            else:
                                tangraNeedsBackgroundSubtraction = True
                        headers.append('FrameNum,timeInfo,signal-LC1')  # Just enough so that pyote doesn't reject the file

                    if kind == 'PyMovie':
                        # We need to count the number of times 'signal" starts
                        # a column header
                        line = line.rstrip()  # Get rid of possible trailing new line \n
                        parts = line.split(',')
                        pymovieDataColumns = []
                        columnIndex = 0
                        for part in parts:
                            # A PyMovie csv file has the possibilty of having duplicated column names (because
                            # the user has freedom to name columns. PyMovie version 3.7.7 does not allow this,
                            # but older versions did not check, so we also check here.
                            if part in pymovieColumnNames:
                                return (False, part + ' is a duplicated column name - please edit names to resolve',
                                        [], [], [], [], [], [], [], aperture_names, headers)

                            pymovieColumnNames.append(part)
                            if part.startswith(pymovieColumnNamePrefix):
                                pymovieSignalColumnCount += 1
                                aperture_names.append(part.split('-')[1])
                                pymovieDataColumns.append(columnIndex)
                            columnIndex += 1
                            # If there are more than 4 columns of 'signals', we need to setup
                            # extra to hold those columns
                        for i in range(5, pymovieSignalColumnCount+1):
                            extra.append([])
                        if pymovieTotalDataDict is not None:
                            # Build an empty dictionary with keys form pyMovieColumnNames list
                            for key in pymovieColumnNames:
                                pymovieTotalDataDict[key] = []
                            pass

                    while True:
                        line = fileobject.readline()
                        if line:
                            if kind == 'Tangra':
                                line = line.rstrip()  # Get rid of possible trailing new line \n
                                if line[-1] == ',':
                                    line += '0'
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
