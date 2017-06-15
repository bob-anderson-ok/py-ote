#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 23 07:24:21 2017

@author: Bob Anderson
"""

import os


def readLightCurve(filepath):
    """
    Reads the intensities and timestamps from Limovie,
    Tangra, or R-OTE csv files.
    """
    if fileCanBeOpened(filepath):
        
        readOk, errMsg, frame, time, value, secondary, headers = readAs(filepath)
        if readOk:
            return frame, time, value, secondary, headers
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
        elif 'R-OTE' in line or line[0] == '#':
            return 'R-OTE'
        elif 'RAW' in line:
            return 'raw'
        else:
            return '???'


def tangraParser(line, frame, time, value, secondary):
    """
    We only accept Tangra files that have been formatted
    according to the AOTA default which is ---
       UT time field formatted as  [hh:mm:ss.sss]
       and readings reported as signal,background (no background subtraction)
       An example data line:  11,[16:00:14.183],2837.8,100.0,4097.32,200.0
    """
    part = line.split(',')
    frame.append(part[0])
    time.append(part[1])
    value.append(str(float(part[2]) - float(part[3])))
    if len(part) >= 6:
        secondary.append(str(float(part[4]) - float(part[5])))
    

def limovieParser(line, frame, time, value, secondary):
    """
    Limovie sample line ---
        3.5,21381195,21381200,22,27,43.0000,,,,,2737.8,3897.32 ...
    """
    part = line.split(',')
    frame.append(part[0])
    time.append('[' + part[3] + ':' + part[4] + ':' + part[5] + ']')
    value.append(part[10])
    if part[11]:
        secondary.append(part[11])
    

def roteParser(line, frame, time, value, secondary):
    """
    R-OTE sample line ---
        1.00,[17:25:39.3415],2737.8,3897.32
    """
    part = line.split(',')
    frame.append(part[0])
    time.append(part[1])
    value.append(part[2])
    if len(part) >= 4:
        if part[3]:
            secondary.append(part[3])


def rawParser(line, frame, time, value, secondary):
    value.append(line)
    

def readAs(file):
    
    kind = getFileKind(file)
    
    fileobject = open(file)
    headers = []
    time = []
    frame = []
    value = []
    secondary = []
    
    if kind == 'Tangra':
        colHeaderKey = 'FrameNo'
        parser = tangraParser
    elif kind == 'R-OTE':
        colHeaderKey = 'FrameNum'
        parser = roteParser
    elif kind == 'Limovie':
        colHeaderKey = 'No.'
        parser = limovieParser
    elif kind == 'raw':
        colHeaderKey = 'RAW'
        parser = rawParser
    else:
        return False, 'invalid file "kind"', [], [], [], [], []
        
    with fileobject:
        while True:
            line = fileobject.readline()
            if line:
                if colHeaderKey in line:
                    while True:
                        line = fileobject.readline()
                        if line:
                            try:
                                parser(line, frame, time, value, secondary)                               
                            except:
                                return False, 'format error', frame, time, value, secondary, headers
                        else:
                            return True, kind, frame, time, value, secondary, headers
                headers.append(line)
            else:
                return (False, colHeaderKey + ' not found as first column header', 
                        [], [], [], [], headers)
    