# Port from ParallelMILC (OpenCL 1.2) to pyParallelMILC (pyOpenCL 2.0)
# Stefano Ricchiuti (stefano_ricchiuti@hotmail.com)

import os, sys
from ParallelMILC import ParallelMILC
import MILC
import Utils
from Constants import DeviceType
from time import time

os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'


if __name__ == "__main__":

    if not len(sys.argv) == 5:
        Utils.message_pause_exit("USAGE: pyPMILC (-c|-d) <inputImageFile> <outputImageFiles> <width> " +
                                 "<height> <# slices|?> [seq|sequential|forceCPU]\n")

    filename = sys.argv[1]
    errors_filename = sys.argv[2]
    width = int(sys.argv[3])
    height = int(sys.argv[4])
    slices = None if sys.argv[5] == '?' else int(sys.argv[5])

    image_data_p = Utils.load_image(filename, width, height)
    image_data = Utils.load_image(filename, width, height).astype(int)

    # result1 = MILC.prediction_errors(image_data)

    try:
        example = ParallelMILC(DeviceType.GPU.value)
    except Exception as ex:
        Utils.message_pause_exit(ex)
    time1 = time()
    result = example.parallel_prediction_errors(image_data_p)
    time2 = time()
    print("Execution time: ", time2 - time1, "s")
    Utils.save_errors(errors_filename, result)
    errors = Utils.load_errors(errors_filename + ".npy")
