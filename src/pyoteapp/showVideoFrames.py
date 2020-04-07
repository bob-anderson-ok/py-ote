import pyqtgraph as pg
import pyqtgraph.examples
from PyQt5 import QtGui
import cv2
import pyoteapp.SER
import glob
import astropy.io.fits as pyfits  # Used for reading/writing FITS files
from Adv2.Adv2File import Adv2reader
from Adv2.AdvError import AdvLibException


# def readAavFile(frame_to_read=0, full_file_path=None):
#     try:
#         rdr = Adv2reader(full_file_path)
#     except AdvLibException as adverr:
#         return repr(adverr), None
#     except IOError as ioerr:
#         return repr(ioerr), None
#
#     rdr.closeFile()
#     return 'ok', None


def readAavFile(frame_to_read=0, full_file_path=None):
    try:
        rdr = Adv2reader(full_file_path)
    except AdvLibException as adverr:
        return {"success": False, "image": None, "errmsg": repr(adverr)}
    except IOError as ioerr:
        return {"success": False, "image": None, "errmsg": repr(ioerr)}

    err, image, _, _ = rdr.getMainImageAndStatusData(frameNumber=frame_to_read)
    rdr.closeFile()
    if not err:
        return {"success": True, "image": image, "errmsg": ""}
    else:
        return {"success": False, "image": None, "errmsg": err}


def readAviFile(frame_to_read=0, full_file_path=None):
    cap = None
    fourcc = ''
    fps = None
    frame_count = None

    if full_file_path:

        cap = cv2.VideoCapture(full_file_path, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            errmsg = f'{full_file_path} could not be opened!'
            return {"success": False, "image": None, "errmsg": errmsg,
                    "fourcc": '', "fps": 0.0, "num_frames": 0}
        else:
            # Let's get the FOURCC code
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
            fourcc_str = f'{fourcc & 0xff:c}{fourcc >> 8 & 0xff:c}{fourcc >> 16 & 0xff:c}{fourcc >> 24 & 0xff:c}'
            # print(f'FOURCC codec ID: {fourcc_str}')

            fps = cap.get(cv2.CAP_PROP_FPS)
            # print(f'frames per second: {fps:0.6f}')

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            # print(f'There are {frame_count} frames in the file.')

    # Read the specified frame
    success, image, errmsg = readAviFrame(frame_to_show=frame_to_read, cap=cap, fourcc=fourcc_str)

    if cap is not None:
        cap.release()

    return {"success": success, "image": image, "errmsg": errmsg,
            "fourcc": fourcc_str, "fps": fps, "num_frames": frame_count}


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


def readSerFile(frame_to_read=0, full_file_path=None):

    if full_file_path:

        ser_meta_data, ser_timestamps = pyoteapp.SER.getMetaData(full_file_path)

        # showSerMetaData()
        frame_count = ser_meta_data['FrameCount']
        print(f'There are {frame_count} frames in the SER file.')
        bytes_per_pixel = ser_meta_data['BytesPerPixel']
        print(f'Image data is encoded in {bytes_per_pixel} bytes per pixel')

        try:
            bytes_per_pixel = ser_meta_data['BytesPerPixel']
            image_width = ser_meta_data['ImageWidth']
            image_height = ser_meta_data['ImageHeight']
            little_endian = ser_meta_data['LittleEndian']
            with open(full_file_path, 'rb') as ser_file_handle:
                image = pyoteapp.SER.getSerImage(
                    ser_file_handle, frame_to_read,
                    bytes_per_pixel, image_width, image_height, little_endian
                )
            raw_ser_timestamp = ser_timestamps[frame_to_read]
            parts = raw_ser_timestamp.split('T')
            time_stamp = f'{parts[0]} @ {parts[1]}'
            print(f'Timestamp found: {time_stamp}')
        except Exception as e:
            return {'success': False, 'errmsg': f"{e}", 'image': None, 'timestamp': ''}

        return {'success': True, 'errmsg': '', 'image': image, 'timestamp': time_stamp}


def readFitsFile(frame_to_read=0, full_file_path=None):
    fits_filenames = sorted(glob.glob(full_file_path + '/*.fits'))
    num_frames = len(fits_filenames)
    # print(f'Number of FITS frames: {num_frames}')
    errmsg = ''
    success = False
    time_stamp = ''

    try:
        hdr = pyfits.getheader(fits_filenames[frame_to_read], 0)

        image = pyfits.getdata(fits_filenames[frame_to_read], 0)

        if 'DATE-OBS' in hdr.keys():
            date_time = hdr['DATE-OBS']
            # The form of DATE-OBS is '2018-08-21T05:21:02.4561235' so we can simply 'split' at the T
            parts = date_time.split('T')
            time_stamp = f'{parts[0]} @ {parts[1]}'

        success = True

    except Exception as e3:
        errmsg = f'While reading image data from FITS file: {e3}'
        image = None

    return {'success': success, 'image': image, 'errmsg': errmsg, 'timestamp': time_stamp, 'num_frames': num_frames}


def excercise():
    _ = QtGui.QApplication([])

    win = QtGui.QMainWindow()
    win.resize(800, 800)
    imv = pg.ImageView()
    imv.show()

    pyqtgraph.examples.run()


if __name__ == "__main__":
    excercise()
