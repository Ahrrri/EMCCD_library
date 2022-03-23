import ctypes as ct


class EMCCD_Const(object):
    ############## dll directory ##############
    DLL_FILE_PATH = "C:\\Program Files\\Andor SDK\\atmcd64d.dll"  # for 64 bit
    # DLL_FILE_PATH = "\\172.22.22.101\\qc_user\\Device Control Scripts\\Andor EMCCD iXON Ultra 888\\DLL\\ATMCD64D.DLL"
    DLL_FILE_DIR = "C:\\Program Files\\Andor SDK"  # for 64 bit

    # shutter opening and closing time
    SHU_OPENINGTIME = 27  # ms
    SHU_CLOSINGTIME = 27  # ms

    DEFAULT_GAIN = 20
    TARGET_TEMPERATURE = -60

    TRIGGER_COUNT = 20
    EXPOSURE_TIME = 20
    """
    ############## emccd parameters ############


    # RAW_IMG_WIDTH = 1024
    # RAW_IMG_HEIGHT = 1024
    RAW_IMG_WIDTH = 512
    RAW_IMG_HEIGHT = 512
    RAW_IMG_SIZE = RAW_IMG_WIDTH * RAW_IMG_HEIGHT

    MAX_GAIN = 1000  # max gain of emccd: 1000
    MIN_GAIN = 2  # min gain of emccd: 2

    DEFAULT_EXPOSURE = 0.1  # s
    MAX_EXPOSURE = 10.0  # s (max exposure of emccd itself is 32768)
    MIN_EXPOSURE = 0.0  # s

    DEFAULT_NUMBER_OF_ACCUMULATION = 1
    MAX_NUMBER_OF_ACCUMULATION = 10000
    MIN_NUMBER_OF_ACCUMULATION = 1

    DEFAULT_ACCUMULATION_CYCLE_TIME = 0.1  # s
    MAX_ACCUMULATION_CYCLE_TIME = 10  # s
    MIN_ACCUMULATION_CYCLE_TIME = 0  # s

    DEFAULT_NUMBER_OF_KINETIC_SERIES = 500
    MAX_NUMBER_OF_KINETIC_SERIES = 10000
    MIN_NUMBER_OF_KINETIC_SERIES = 1

    DEFAULT_KINETIC_CYCLE_TIME = 0.1  # s
    MAX_KINETIC_CYCLE_TIME = 10  # s
    MIN_KINETIC_CYCLE_TIME = 0  # s

    DEFAULT_EXPOSURE_MICRO_SEC = 1000.0  # us
    MAX_EXPOSURE_MICRO_SEC = 1000000.0  # us
    MIN_EXPOSURE_MICRO_SEC = 0.0  # us

    DEFAULT_EXPOSED_ROWS = 100
    MAX_EXPOSED_ROWS = RAW_IMG_HEIGHT
    MIN_EXPOSED_ROWS = 1

    DEFAULT_HORIZONTAL_BIN = 1
    MAX_HORIZONTAL_BIN = RAW_IMG_HEIGHT
    MIN_HORIZONTAL_BIN = 1

    DEFAULT_VERTICAL_BIN = 1
    MAX_VERTICAL_BIN = RAW_IMG_WIDTH
    MIN_VERTICAL_BIN = 1

    DEFAULT_OFFSET = 0
    MAX_OFFSET = RAW_IMG_HEIGHT
    MIN_OFFSET = 0
    """