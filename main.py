# Port from ParallelMILC (OpenCL 1.2) to pyParallelMILC (pyOpenCL 2.0)
# Stefano Ricchiuti (stefano_ricchiuti@hotmail.com)

import os
import numpy
from ParallelMILC import ParallelMILC
import MILC
import Utils
from Constants import DeviceType
from time import time

os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'


if __name__ == "__main__":

    filename = "testset\\256\MR_liver_t1.img"

    image_data_p = Utils.load_image(filename, 256, 256)
    image_data = Utils.load_image(filename, 256, 256).astype(int)

    # result1 = MILC.prediction_errors(image_data)

    try:
        example = ParallelMILC(DeviceType.GPU.value)
    except Exception as ex:
        print(ex)
        input()
        exit()

    result = example.parallel_prediction_errors(image_data_p)
    # time2 = time()
    # print("Execution time: ", time2 - time1, "s")
    Utils.save_errors("errors", result)
    print(result)
    errors = Utils.load_errors("errors.npy")
