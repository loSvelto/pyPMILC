from pyopencl import device_type, channel_type
from enum import Enum
from numpy import uint8
from numpy import uint16


class DeviceType (Enum):
    """

    """

    GPU = device_type.GPU
    CPU = device_type.CPU


class DataType (Enum):

    PIXEL = uint8
    PIXEL_MAX = 255
    ERR = uint16
    CL_PIXEL = channel_type.UNSIGNED_INT8
    CL_ERR = channel_type.UNSIGNED_INT16
