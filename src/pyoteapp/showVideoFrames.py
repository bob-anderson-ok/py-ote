import pyqtgraph as pg
import pyqtgraph.examples
from PyQt5 import QtGui
import cv2


def readAviFile(frame_to_read=0, full_file_path=None):
    cap = None
    fourcc = ''

    if full_file_path:

        cap = cv2.VideoCapture(full_file_path, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            return f'{full_file_path} could not be opened!'
        else:
            # Let's get the FOURCC code
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
            fourcc_str = f'{fourcc & 0xff:c}{fourcc >> 8 & 0xff:c}{fourcc >> 16 & 0xff:c}{fourcc >> 24 & 0xff:c}'
            print(f'FOURCC codec ID: {fourcc_str}')

            fps = cap.get(cv2.CAP_PROP_FPS)
            print(f'frames per second: {fps:0.6f}')

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f'There are {frame_count} frames in the file.')

    # Read the specified frame
    success, image, errmsg = readAviFrame(frame_to_show=frame_to_read, cap=cap, fourcc=fourcc)

    if cap is not None:
        cap.release()

    return success, image, errmsg


def readAviFrame(frame_to_show, cap, fourcc):
    image = None

    try:
        if fourcc == 'dvsd':
            success, frame = getDvsdAviFrame(frame_to_show, cap)
            if len(frame.shape) == 3:
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_to_show)
            status, frame = cap.read()
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        return True, image, 'ok'

    except Exception as e1:
        return False, None, f'Problem reading avi file: {e1}'


def getDvsdAviFrame(fr_num, cap):
    # This routine is used for 'dvsd' avi files because they cannot be positioned directly to a specific frame

    success = False
    frame = None

    next_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)

    if fr_num == next_frame:
        success, frame = cap.read()
        return success, frame

    if fr_num > next_frame:
        frames_to_read = fr_num - next_frame + 1
        while frames_to_read > 0:
            frames_to_read -= 1
            success, frame = cap.read()
        return success, frame

    return False, None


def excercise():
    _ = QtGui.QApplication([])

    win = QtGui.QMainWindow()
    win.resize(800, 800)
    imv = pg.ImageView()
    imv.show()

    pyqtgraph.examples.run()


if __name__ == "__main__":
    excercise()
