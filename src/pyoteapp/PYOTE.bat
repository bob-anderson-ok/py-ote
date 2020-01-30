@echo off
cd C:\Anaconda3

echo Creating run-pyote.py in C:\Anaconda3

@echo from pyoteapp import pyote > run-pyote.py
@echo pyote.main() >> run-pyote.py

echo. 
echo Activating the base Anaconda environment

call C:\Anaconda3\Scripts\activate.bat

echo.
echo Executing the run-pyote.py script

python C:\Anaconda3\run-pyote.py
