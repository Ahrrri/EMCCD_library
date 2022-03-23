import os
import sys
import time
import threading
import numpy as np
import ctypes as ct
import win32event
import PIL.Image

from PyQt5.QtCore import pyqtSignal, QObject, QThread
from .EMCCD_Const import EMCCD_Const
from .EMCCD_drv import *
from ..base import CCD

filename = os.path.abspath(__file__)
dirname = os.path.dirname(filename)

dummy_image = np.array(PIL.Image.open(dirname + "\\dummy_image.png").convert('L'))
print(dummy_image.shape)


class AndorCapabilities(ct.Structure):
    """
    struct AndorCapabilities used for GetCapacities function
    """
    _fields_ = [("ulSize", ct.c_ulong),
                ("ulAcqModes", ct.c_ulong),
                ("ulReadModes", ct.c_ulong),
                ("ulFTReadModes", ct.c_ulong),
                ("ulTriggerModes", ct.c_ulong),
                ("ulCameraType", ct.c_ulong),
                ("ulPixelModes", ct.c_ulong),
                ("ulSetFunctions", ct.c_ulong),
                ("ulGetFunctions", ct.c_ulong),
                ("ulFeatures", ct.c_ulong),
                ("ulPCICard", ct.c_ulong),
                ("ulEMGainCapability", ct.c_ulong)]

    def __init__(self):
        super().__init__()
        self.ulSize = ct.sizeof(AndorCapabilities)

    def print_capabilities(self):
        print(f"All properties are in binary:")
        print(f"ulSize: {self.ulSize:b}")

        # Acquisition mode
        # Overlap Acquisition | Fast Kinetics | Frame Transfer | Kinetic Series |
        # Accumulation Acquisition | Video(Run till abort) | Single Scan(Rightmost bit)
        # IDK about 8th bit set by 1
        print(f"ulAcqModes: {self.ulAcqModes:b}")

        # Read mode
        # Random Track | Multi Track | Full Vertical Binning | Single Track | Sub Image | Full Image(Rightmost bit)
        print(f"ulReadModes: {self.ulReadModes:b}")

        # Read mode compatible with Frame Transfer mode
        # Random Track | Multi Track | Full Vertical Binning | Single Track | Sub Image | Full Image(Rightmost bit)
        print(f"ulFTReadModes: {self.ulFTReadModes:b}")

        # Trigger mode
        # External Charge Shifting | Inverted | External Exposure | Bulb Trigger | External Start
        # Continuous | External FVB EM | External | Internal(Rightmost bit)
        print(f"ulTriggerModes: {self.ulTriggerModes:b}")

        # Camera Type
        print(f"ulCameraType: {self.ulCameraType:b}")

        # ...
        print(f"ulPixelModes: {self.ulPixelModes:b}")
        print(f"ulSetFunctions: {self.ulSetFunctions:b}")
        print(f"ulGetFunctions: {self.ulGetFunctions:b}")
        print(f"ulFeatures: {self.ulFeatures:b}")
        print(f"ulPCICard: {self.ulPCICard:b}")
        print(f"ulEMGainCapability: {self.ulEMGainCapability:b}")


def error_check(func):
    def func_wrapper(*args, **kwargs):
        if "ignore_code" in kwargs.keys():
            ignore_code = kwargs["ignore_code"]
            del kwargs["ignore_code"]
        else:
            ignore_code = []

        if "ignore_error" in kwargs.keys():
            ignore_error = kwargs["ignore_error"]
            del kwargs["ignore_error"]
        else:
            ignore_error = False

        returned_code = func(*args, **kwargs)

        if not ((returned_code == DRV_SUCCESS) or ignore_error or (returned_code in ignore_code)):
            print(f'andor SDK returned Error: {int_to_drv[returned_code]} [{returned_code}]')
        return returned_code

    func_wrapper.__name__ = func.__name__
    return func_wrapper


class EMCCD(QObject, CCD, EMCCD_Const):
    new_image_signal = pyqtSignal(np.ndarray)

    def __init__(self, parent=None, dll_file_path=None):
        QObject.__init__(self)
        self.parent = parent
        if dll_file_path is not None:
            self.DLL_FILE_PATH = dll_file_path
        self._load_dll()

        self._initialize()

        self.buffer_cv = threading.Condition()

        self._andor_capabilities = AndorCapabilities()
        self._acquisition_mode = 'continuous_scan'
        self._avail_acquisition_modes = ['single_scan', 'continuous_scan']
        self._status = None
        self._gain = None
        self._gain_range = None
        self._exposure_time = None
        self._accumulation_cycle_time = None
        self._kinetic_cycle_time = None

        #
        #
        self._default_setting()
        self._ccd_image = np.zeros((self.RAW_IMG_HEIGHT, self.RAW_IMG_WIDTH), dtype=np.int32)
        self._buffer_size = 100
        self._buffer_list = []
        self._img_cnt = 0
        self._image_get_thread = None
        #
        # self._running_thread = None
        # self._cooling_tread = None
        #
        # self._cool_on = False
        # self._cool_status_list = ["stabilized", "cooling", "hot"]
        # self._cool_status = "hot"

    def _load_dll(self):
        """Loads the dll files.
        """
        self.dll = None

        with open(dirname + "/emccd_functions.txt") as f:
            dll_methods = f.read().splitlines()

        for method in dll_methods:
            try:
                # used try-except to skip reserved(not yet implemented) functions
                setattr(self, method, error_check(getattr(self.dll, method)))
            except AttributeError as e:
                print(method, "not yet implemented (Reserved Function. Check SDK for details).")

    def _initialize(self):
        code = DRV_SUCCESS
        if code == DRV_SUCCESS:
            print('Device successfully connected. You need to cool the device before you can start imaging')
        else:
            print('Something went wrong with Initialize()')

    def _default_setting(self):

        # GetDetector
        self.RAW_IMG_WIDTH = dummy_image.shape[1]
        self.RAW_IMG_HEIGHT = dummy_image.shape[0]
        self.RAW_IMG_SIZE = self.RAW_IMG_WIDTH * self.RAW_IMG_HEIGHT
        print(f"DETECTOR RAW IMAGE SIZE: {self.RAW_IMG_WIDTH} * {self.RAW_IMG_HEIGHT} = {self.RAW_IMG_SIZE}")

        # pass GetHardwareVersion

        # GetNumberVSSpeeds. 4

        # GetVSSpeed. 0.6, 1.13, 2.2, 4.33 microseconds per one vertical pixel shift

        # GetFastestRecommendedVSSpeed. recommends the fastest VS speed without adjusting Vertical Clock Voltage
        # To adjust Vertical Clock Voltage, see SetVSAmplitude

        # pass GetSoftwareVersion

        # GetNumberADChannels. more than ond A-D converter. See SetADChannel

        # GetNumberHSSpeeds
        # GetHSSpeed

        # GetCapabilities
        self._andor_capabilities.print_capabilities()

        # self.SetReadMode(4)  # Image Readout Mode
        # self.SetTriggerMode(0)  # use Internal Trigger
        # self.SetShutter(1, 1, self.SHU_CLOSINGTIME,
        #                 self.SHU_OPENINGTIME)  # SetShutter(int typ, int mode, int closingtime, int openingtime)
        # # mode 0:auto, 1: open, 2: close
        # self.SetImage(1, 1, 1, self.RAW_IMG_WIDTH, 1, self.RAW_IMG_HEIGHT)  # Full size
        #
        # # get size of circular buffer
        # circular_buffer_size = ct.c_long(0)
        # self.GetSizeOfCircularBuffer(ct.pointer(circular_buffer_size))
        # self.circular_buffer_size = circular_buffer_size.value
        #
        # # set acquisition mode
        # self.acquisition_mode = "continuous_scan"
        # self.SetAcquisitionMode(5)
        #

        # GetEMGainRange
        # Returns the minimum and maximum values of the current selected EM Gain mode and temperature of the sensor.
        min_gain = ct.c_int(0)
        max_gain = ct.c_int(0)

        self._gain_range = [1, 100]
        gain = ct.c_int(0)
        self._gain = gain.value

        # self.SetNumberAccumulations(1)

        # self._trigger_count = self.DEFAULT_NUMBER_OF_KINETIC_SERIES
        # self.trigger_count = self._trigger_count

        # self._exposure_time = self.DEFAULT_EXPOSURE
        # self.exposure_time = self._exposure_time

    # there is two ways to get images
    # One is straightforward, using Event with win32event.CreateEvent and WaitSingleObject, SetDriverEvent
    # The other one is using alternative ways, such as WaitForAcquisitionTimeOut, offered from Andor SDK 2.
    # See https://ssjune.tistory.com/category/pywin32/win32event
    def run_device(self):
        if True:
            start_acquisition_code = DRV_SUCCESS
            if start_acquisition_code == DRV_SUCCESS:
                _, _, kinetic_cycle_time = self.get_acquisition_timings()
                contiguous_image_array = np.ascontiguousarray(np.zeros(self.RAW_IMG_SIZE, dtype=np.int32))
                self._buffer_list = []
                self._img_cnt = 0
                self.status = DRV_ACQUIRING
                while self.status == DRV_ACQUIRING:
                    print('update' + str(self._img_cnt) + ", buffer " + str(len(self._buffer_list)))
                    QThread.msleep(100)
                    image_buffer = np.copy(dummy_image)
                    self._img_cnt += 1
                    self.ccd_image = np.copy(image_buffer)
                    if not (len(self._buffer_list) < self._buffer_size):
                        while len(self._buffer_list) >= self._buffer_size:
                            self._buffer_list.pop(0)
                    self._buffer_list.append(self.ccd_image)
                    self.new_image_signal.emit(image_buffer)
        else:
            print('Something went wrong with acquisition')

    def stop_device(self):
        self.status = DRV_IDLE
        # self.AbortAcquisition(ignore_code=[DRV_IDLE])

    def close_device(self):
        if self.acquire_thread is not None:
            self.stop_device()
            self.acquire_thread.join()

        # need to notify to wake up waiting threads
        self.buffer_cv.acquire()
        self.buffer_cv.notify_all()
        self.buffer_cv.release()

        self.CoolerOFF(ignore_error=True)
        self.ShutDown()

    def get_image_get_thread(self):
        if self._image_get_thread is not None:
            return self._image_get_thread
        else:
            self._image_get_thread = self._ImageGetThread(self)
            return self._image_get_thread

    class _ImageGetThread(QThread):
        def __init__(self, cam):
            super().__init__()
            self.cam = cam

        def quit(self) -> None:
            self.cam.stop_device()

        def run(self) -> None:
            self.cam.run_device()

    @property
    def sensor_size(self):
        return [self.RAW_IMG_HEIGHT, self.RAW_IMG_WIDTH]

    @property
    def acquisition_mode(self):
        return self._acquisition_mode

    @acquisition_mode.setter
    def acquisition_mode(self, mode):
        """
        You can only select "single_scan" or "continuous_scan".
        """
        assert isinstance(mode, str), "Acquisition mode should be a string."
        if mode not in self._avail_acquisition_modes:
            raise ValueError("Not supported Acquisition mode")
        self._acquisition_mode = mode

        if mode == "single_scan":
            self.SetAcquisitionMode(3)  # kinetic series
        elif mode == "continuous_scan":
            self.SetAcquisitionMode(5)  # run till abort

    @property
    # There is no function such as GetExposureTime. It can only be obtained by GetAcquisitionTimings.
    def exposure_time(self):
        return self._exposure_time

    @exposure_time.setter
    # This function will set the exposure time to the nearest valid value not less than the given value.
    # The actual exposure time used is obtained by GetAcquisitionTimings.
    def exposure_time(self, value):
        self.SetExposureTime(ct.c_float(value))
        self._exposure_time = value

    @property
    def gain(self):
        return self._gain

    @gain.setter
    def gain(self, value):
        value = int(value)
        min_gain = ct.c_int(0)
        max_gain = ct.c_int(0)
        self.GetEMGainRange(ct.byref(min_gain), ct.byref(max_gain))
        self._gain_range = [min_gain.value, max_gain.value]
        if value > max_gain.value:
            value = max_gain.value
        elif value < min_gain.value:
            value = min_gain.value
        self.SetEMCCDGain(value)
        self._gain = value

    @property
    def trigger_count(self):
        return self._trigger_count

    @trigger_count.setter
    def trigger_count(self, count):
        assert isinstance(count, int), "Trigger count should be an int."
        self.SetNumberKinetics(count)
        self._buffer_size = count
        self._trigger_count = count

    @property
    def target_temperature(self):
        return self._target_temperature

    @target_temperature.setter
    def target_temperature(self, temp):
        min_temp = ct.c_int(0)
        max_temp = ct.c_int(0)
        self.GetTemperatureRange(ct.byref(min_temp), ct.byref(max_temp))
        if temp < min_temp:
            temp = min_temp
        elif temp > max_temp:
            temp = max_temp
        self.SetTemperature(temp)
        self._target_temperature = temp

    @property
    def current_temperature(self):
        temperature = ct.c_int(0)
        self.GetTemperature(ct.byref(temperature),
                            ignore_code=[DRV_TEMPERATURE_OFF,
                                         DRV_TEMP_NOT_STABILIZED,
                                         DRV_TEMPERATURE_STABILIZED,
                                         DRV_TEMPERATURE_NOT_REACHED,
                                         DRV_TEMPERATURE_DRIFT,
                                         DRV_ACQUIRING])
        self._current_temperature = temperature.value
        return self._current_temperature

    @property
    def ccd_image(self):
        return self._ccd_image

    @ccd_image.setter
    def ccd_image(self, image):
        self.buffer_cv.acquire()
        self._ccd_image = image
        self.buffer_cv.notify_all()
        self.buffer_cv.release()

    def get_acquisition_timings(self):
        return 1, 1, 1

    @property
    def status(self):
        return self._status

    @status.setter
    def status(self, value):
        self._status = value


if __name__ == "__main__":
    temp = EMCCD()
