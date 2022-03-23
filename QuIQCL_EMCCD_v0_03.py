"""
This is a heritage of HunHuh who left us remaining lots of fineworks

Modification applied by JJH to integrate it with CCD GUI
Revised a little by CBG

This API is based on Andor SDK 2 (which controls CCD cameras) version 2.102.

in win32event library you can create a Win32event.
Use this to connect via SetDriverEvent
and use win32event.WaitForSingleObject to wait EMCCD's signal.
It can be alternated by WaitForAcquisitionTimeOut

pyqtSignal must be declared as class variable, though it actually acts as instance variable.
It is somewhat "instantiated" by super().__init__(), with QObject.

과거의 유산으로부터 가져올 것
save_acquired_data에서 SaveAsTiff, SaveAsPNG 이런거 써보기
그림을 QImage 이런 걸로 안하고 matplotlib으로 그리는 이유?
AndorCapabilities에서 깔끔하게 available feature 추출하기
건질 게 많은 것 같지는 않다.

To do
Crop mode ROI
zoom out
turning off Cooler
external trigger
spooling ...(save)
exposure time을 줄이고 gain을 엄청 늘리기, Fastkinetics
Binning?
Acquisition 중 exposure 바꾸기
Rotation
뒤집었을 때 줌 되어 있으면 이상하게 보이는 문제
"""
import os
import sys
import time
import numpy as np
import ctypes as ct
import _ctypes

from PyQt5.QtCore import pyqtSignal, QObject, QThread
from .EMCCD_Const import EMCCD_Const
from .EMCCD_drv import *
from ..base import CCD
from datetime import datetime

filename = os.path.abspath(__file__)
dirname = os.path.dirname(filename)


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

        code = func(*args, **kwargs)

        if not ((code == DRV_SUCCESS) or ignore_error or (code in ignore_code)):
            print(f'andor SDK returned Error: {int_to_drv[code]} [{code}]')
        return code

    func_wrapper.__name__ = func.__name__
    return func_wrapper


class EMCCD(QObject, CCD, EMCCD_Const):
    new_image_signal = pyqtSignal(np.ndarray)
    acquisition_finished_signal = pyqtSignal()
    temperature_signal = pyqtSignal(int, str)

    def __init__(self, parent=None, dll_file_path=None):
        QObject.__init__(self)
        EMCCD_Const.__init__(self)
        self.parent = parent
        if dll_file_path is not None:
            self.DLL_FILE_PATH = dll_file_path

        self._load_dll()

        self._initialize()

        self._andor_capabilities = None
        self._accumulation_cycle_time = None
        self._acquisition_mode = None
        self._avail_acquisition_modes = ['single_scan', 'continuous_scan']
        self._avail_read_mode = ["FVB", "multi_track", "random_track", "single_track", "image"]
        self._exposure_time = None
        self._gain = None
        self._gain_range = None
        self._status = None
        self._kinetic_cycle_time = None
        self._trigger_count = None

        self._buffer_size = 50
        self._buffer_list = []
        self._img_cnt = 0
        self._image_acquiring_thread = None

        self._temperature_thread = None
        self._temperature_thread_flag = False
        self._cooler_on_flag = False
        self._current_temperature = None
        self._target_temperature = None

        self._default_setting()
        self._ccd_image = np.zeros((self.raw_img_height, self.raw_img_width), dtype=np.int32)
        # self._cool_status_list = ["stabilized", "cooling", "hot"]
        # self._cool_status = "hot"
        self._temperature_thread = self.get_temperature_thread()
        self._temperature_thread.start()

    def __del__(self):
        _ctypes.FreeLibrary(self.dll._handle)
        del self.dll

    def _load_dll(self):
        """Loads the dll files.
        """
        self.dll = ct.WinDLL(self.DLL_FILE_PATH)

        with open(dirname + "/emccd_functions.txt") as f:
            dll_methods = f.read().splitlines()

        for method in dll_methods:
            try:
                # used try-except to skip reserved(not yet implemented) functions
                setattr(self, method, error_check(getattr(self.dll, method)))
            except AttributeError as e:
                print(method, "not yet implemented (Reserved Function. Check SDK for details).")

    def _initialize(self):
        code = self.Initialize(self.DLL_FILE_DIR)
        if code == DRV_SUCCESS:
            print('Device successfully connected. You need to cool the device before you can start imaging')
        else:
            print('Something went wrong with Initialize()')

    def _default_setting(self):

        # GetDetector
        detector_xpixels = ct.c_int(0)
        detector_ypixels = ct.c_int(0)
        self.GetDetector(ct.byref(detector_xpixels), ct.byref(detector_ypixels))
        self.raw_img_width = detector_xpixels.value
        self.raw_img_height = detector_ypixels.value
        self.raw_img_size = self.raw_img_width * self.raw_img_height
        print(f"DETECTOR RAW IMAGE SIZE: {self.raw_img_width} * {self.raw_img_height} = {self.raw_img_size}")
        print(f"Set image to the raw image size")
        self.SetImage(1, 1, 1, self.raw_img_width, 1, self.raw_img_height)  # Full size
        self.SetReadMode(4)  # Image Readout Mode

        # pass GetHardwareVersion

        # GetNumberVSSpeeds. 4
        vertical_shift_speeds = ct.c_int(0)
        self.GetNumberVSSpeeds(ct.byref(vertical_shift_speeds))
        print(f"Detector's number of allowed vertical speeds: {vertical_shift_speeds.value}")

        # GetVSSpeed. 0.6, 1.13, 2.2, 4.33 microseconds per one vertical pixel shift
        vertical_shift_speed = ct.c_float(0)
        self.VERTICAL_SHIFT_SPEEDS = []
        for i in range(vertical_shift_speeds.value):
            self.GetVSSpeed(i, ct.byref(vertical_shift_speed))
            self.VERTICAL_SHIFT_SPEEDS.append(vertical_shift_speed.value)
        print(f"Detector's available VS speeds: {self.VERTICAL_SHIFT_SPEEDS}")

        # GetFastestRecommendedVSSpeed. recommends the fastest VS speed without adjusting Vertical Clock Voltage
        # To adjust Vertical Clock Voltage, see SetVSAmplitude
        # index = ct.c_int(0)
        # speed = ct.c_float(0)
        # self.GetFastestRecommendedVSSpeed(ct.byref(index), ct.byref(speed))
        # print(f"SDK recommends index {index.value}, speed {speed.value} for vertical shift")
        # self.SetVSSpeed(index.value)
        # print(f"Set VSSpeed to {speed.value}")
        self.SetVSAmplitude(3)
        self.SetVSSpeed(0)

        # pass GetSoftwareVersion

        # GetNumberADChannels. More than one A-D converter is possible. See SetADChannel
        channels = ct.c_int(0)
        self.GetNumberADChannels(ct.byref(channels))
        self._number_ad_channels = channels.value
        print(f"{channels.value} A-D converters available")

        # GetNumberHSSpeeds
        horizontal_shift_speeds = ct.c_int(0)
        self.GetNumberHSSpeeds(0, 0, ct.byref(horizontal_shift_speeds))
        print(f"Detector's number of allowed horizontal speeds: {horizontal_shift_speeds.value}")
        # GetHSSpeed
        horizontal_shift_speed = ct.c_float(0)
        self.HORIZONTAL_SHIFT_SPEEDS = []
        for i in range(horizontal_shift_speeds.value):
            self.GetHSSpeed(0, 0, i, ct.byref(horizontal_shift_speed))
            self.HORIZONTAL_SHIFT_SPEEDS.append(horizontal_shift_speed.value)
        print(f"Detector's available HS speeds: {self.HORIZONTAL_SHIFT_SPEEDS}(MHz)")

        # SetHSSpeed
        self.SetHSSpeed(0, 0)
        print(f"Set HS Speed to {self.HORIZONTAL_SHIFT_SPEEDS[0]}")

        # GetCapabilities
        self._andor_capabilities = AndorCapabilities()
        self.GetCapabilities(ct.byref(self._andor_capabilities))
        self._andor_capabilities.print_capabilities()

        # use Internal Trigger
        self.SetTriggerMode(0)

        # type, mode, closingtime, openingtime
        # typ : 0 TTL low signal to open shutter
        # mode : 0 auto, 1 Permanently open 2 Permanently closed
        self.SetShutter(1, 1, self.SHU_CLOSINGTIME, self.SHU_OPENINGTIME)

        # # get size of circular buffer
        circular_buffer_size = ct.c_long(0)
        self.GetSizeOfCircularBuffer(ct.pointer(circular_buffer_size))
        self.circular_buffer_size = circular_buffer_size.value
        print(f'circular buffer size is {circular_buffer_size.value}')

        # set target temperature
        self.target_temperature = -60

        # # set acquisition mode
        self.acquisition_mode = "single_scan"
        # I think accumulation does not have to be anything other than 1

        self.SetNumberAccumulations(1)
        # self.SetAccumulationCycleTime(0)
        # SetEMGain
        self.gain = self.DEFAULT_GAIN
        self.trigger_count = 10
        # SetExposureTime
        self.exposure_time = self.EXPOSURE_TIME

    def close_device(self):
        thread = self.get_image_acquiring_thread()
        if thread.isRunning():
            thread.stop()
            thread.wait()

        thread = self.get_temperature_thread()
        if thread.isRunning():
            thread.stop()
            thread.wait()

        self.ShutDown()
        print("successfully shutdowned")

    # there is two ways to get images
    # One is straightforward, using Event with win32event.CreateEvent and WaitSingleObject, SetDriverEvent
    # The other one is using alternative ways, such as WaitForAcquisitionTimeOut, offered from Andor SDK 2.
    # See https://ssjune.tistory.com/category/pywin32/win32event
    def run_device(self):
        if self.status == DRV_IDLE:
            start_acquisition_code = self.StartAcquisition()
            if start_acquisition_code == DRV_SUCCESS:
                exposure_time, accumulate, kinetic_cycle_time = self.get_acquisition_timings()
                contiguous_image_array = np.ascontiguousarray(np.zeros(self.raw_img_size, dtype=np.int32))
                self._buffer_list = []
                self._img_cnt = 0
                print(
                    f'exposure {exposure_time:.5f}, accumul {accumulate:.5f}, kinetic cycle {kinetic_cycle_time:.5f} seconds')
                while self.status == DRV_ACQUIRING:
                    while self.WaitForAcquisitionTimeOut(int(kinetic_cycle_time * 1000 + 2000),
                                                         ignore_code=[DRV_NO_NEW_DATA]) == DRV_SUCCESS:
                        code = self.GetMostRecentImage(
                            contiguous_image_array.ctypes.data_as(ct.POINTER(ct.c_int32)),
                            ct.c_ulong(self.raw_img_size),
                            ignore_code=[DRV_NO_NEW_DATA])
                        if code == DRV_SUCCESS:
                            image_buffer = contiguous_image_array.reshape(
                                (self.raw_img_width, self.raw_img_height))
                            self._img_cnt += 1
                            self._ccd_image = np.copy(image_buffer)
                            if not (len(self._buffer_list) < self._buffer_size):
                                while len(self._buffer_list) >= self._buffer_size:
                                    self._buffer_list.pop(0)
                            self._buffer_list.append(self._ccd_image)
                            self.new_image_signal.emit(self._ccd_image)
                self.acquisition_finished_signal.emit()
            else:
                print('Something went wrong with acquisition')

        else:
            print('Could not start acquisition because drive was not IDLE.')

    def stop_device(self):
        code = self.AbortAcquisition(ignore_code=[DRV_IDLE])
        if code == DRV_SUCCESS:
            print("Aborted successfully. Wait for the thread... ", end="")
        else:
            print("Something went wrong with AbortAcquisition")

    def cooler_on(self):
        if not self._cooler_on_flag:
            self._cooler_on_flag = True
            print(f'CoolerON returned {int_to_drv[self.CoolerON()]}')

    def cooler_off(self):
        if self._cooler_on_flag:
            self._cooler_on_flag = False
            self.CoolerOFF()

    def _temperature_thread_on(self):
        while self._temperature_thread_flag:
            current_temperature, code = self.current_temperature
            self.temperature_signal.emit(current_temperature, int_to_drv[code])
            QThread.sleep(3)

    def _temperature_thread_off(self):
        self._temperature_thread_flag = False
        if self._cooler_on_flag:
            self._cooler_on_flag = False
            self.CoolerOFF()

    def get_image_acquiring_thread(self):
        if self._image_acquiring_thread is not None:
            return self._image_acquiring_thread
        else:
            self._image_acquiring_thread = self._ImageAcquireThread(self)
            return self._image_acquiring_thread

    def get_temperature_thread(self):
        if self._temperature_thread is not None:
            return self._temperature_thread
        else:
            self._temperature_thread = self._TemperatureThread(self)
            return self._temperature_thread

    class _ImageAcquireThread(QThread):
        def __init__(self, cam):
            super().__init__()
            self.cam = cam

        def stop(self):
            self.cam.stop_device()
            self.quit()

        def run(self) -> None:
            self.cam.run_device()

    class _TemperatureThread(QThread):
        def __init__(self, cam):
            super().__init__()
            self.cam = cam

        def stop(self) -> None:
            self.cam._temperature_thread_off()
            self.quit()

        def run(self) -> None:
            # This does not mean cooler on. This is Cooling Thread on.
            self.cam._temperature_thread_flag = True
            self.cam._temperature_thread_on()

    def get_acquisition_timings(self):
        exposure = ct.c_float(0.0)
        accumulation_cycle_time = ct.c_float(0.0)
        kinetic_cycle_time = ct.c_float(0.0)
        self.GetAcquisitionTimings(ct.pointer(exposure), ct.pointer(accumulation_cycle_time),
                                   ct.pointer(kinetic_cycle_time))
        self._exposure_time = exposure.value
        self._accumulation_cycle_time = accumulation_cycle_time.value
        self._kinetic_cycle_time = kinetic_cycle_time.value
        return exposure.value, accumulation_cycle_time.value, kinetic_cycle_time.value

    @property
    def sensor_size(self):
        return [self.raw_img_width, self.raw_img_height]

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
    def ccd_image(self):
        return self._ccd_image

    @property
    def current_temperature(self):
        temperature = ct.c_int(0)
        code = self.GetTemperature(ct.byref(temperature),
                                   ignore_code=[DRV_TEMPERATURE_OFF,
                                                DRV_TEMP_NOT_STABILIZED,
                                                DRV_TEMPERATURE_STABILIZED,
                                                DRV_TEMPERATURE_NOT_REACHED,
                                                DRV_TEMPERATURE_DRIFT,
                                                DRV_ACQUIRING])
        self._current_temperature = temperature.value
        return self._current_temperature, code

    @property
    # There is no function such as GetExposureTime. It can only be obtained by GetAcquisitionTimings.
    def exposure_time(self):
        return self._exposure_time

    @exposure_time.setter
    # This function will set the exposure time to the nearest valid value not less than the given value.
    # The actual exposure time used is obtained by GetAcquisitionTimings.
    def exposure_time(self, value):
        self.SetExposureTime(ct.c_float(0.001 * value))  # input is in seconds
        self._exposure_time = value  # self._exposure_time is in milliseconds

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
        print(f'gain range is {min_gain.value} to {max_gain.value}')
        self._gain = value

    @property
    def read_mode(self):
        return self._read_mode

    @read_mode.setter
    def read_mode(self, mode):
        pass

    @property
    def status(self):
        status = ct.c_int(0)
        self.GetStatus(ct.pointer(status))
        self._status = status.value
        return status.value

    @property
    def target_temperature(self):
        return self._target_temperature

    @target_temperature.setter
    def target_temperature(self, temp):
        min_temp = ct.c_int(0)
        max_temp = ct.c_int(0)
        self.GetTemperatureRange(ct.byref(min_temp), ct.byref(max_temp))
        if temp < min_temp.value:
            temp = min_temp.value
        elif temp > max_temp.value:
            temp = max_temp.value
        self.SetTemperature(temp)
        self._target_temperature = temp

    @property
    def trigger_count(self):
        return self._trigger_count

    @trigger_count.setter
    def trigger_count(self, count):
        assert isinstance(count, int), "Trigger count should be an int."
        self.SetNumberKinetics(count)
        # self._buffer_size = count
        self._trigger_count = count


if __name__ == "__main__":
    temp = EMCCD()
