import pyopencl as cl
import numpy
from Constants import DataType


class ParallelMILC:

    def __init__(self, device_type=cl.device_type.GPU):
        """Initialise an OpenCL context bounded to this instance and loads the main kernels

        Keyword arguments:
        device_type --  preferred device type (DeviceType type from Constants.py)
                        default is GPU
        """

        # Get all the available platforms
        platforms = cl.get_platforms()
        # Raise exception if no OpenCL platform is found
        if not platforms:
            raise Exception("No OpenCL platform available")

        # Get a device of the specified type from the first platform available
        devices = platforms[0].get_devices(device_type)
        if not devices:
            raise Exception("No OpenCL device of the type ", device_type, " is available")

        self.ctx = cl.Context(
            dev_type=device_type,
            properties=[(cl.context_properties.PLATFORM, platforms[0])])

        self.device_type = device_type
        self.queue = cl.CommandQueue(self.ctx)

        # Load and build the kernel. Filename is hardcoded as it isn't supposed to get changed by users
        self._load_program("part1.cl")

    def _load_program(self, filename):

        f = open(filename, 'r')

        # Additional preprocessor macros
        defines = ""
        # defines = """
        # #define DEV_TYPE 2
        # """

        file_str = "".join(defines).join(f.readlines())
        self.program = cl.Program(self.ctx, file_str).build()

    def parallel_prediction_errors(self, image):

        mf = cl.mem_flags
        err_format = cl.ImageFormat(channel_order=cl.channel_order.R, channel_type=DataType.CL_ERR.value)

        source_image = cl.image_from_array(self.ctx, image)

        original_shape = numpy.shape(image)
        cl_shape = list(reversed(original_shape))

        output_image = cl.Image(
            self.ctx,
            mf.WRITE_ONLY,
            err_format,
            shape=cl_shape
        )

        sampler = cl.Sampler(self.ctx, False, cl.addressing_mode.CLAMP, cl.filter_mode.NEAREST)

        self.program.image_test(self.queue, original_shape, None, source_image, output_image, sampler)

        output_data = numpy.empty(shape=cl_shape, dtype=DataType.ERR.value)
        cl.enqueue_read_image(self.queue, output_image, (0, 0, 0), cl_shape, output_data)

        return output_data.reshape(original_shape)
