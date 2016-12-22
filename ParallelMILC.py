import pyopencl as cl
import numpy
from Constants import DataType


class ParallelMILC:

    def __init__(self, device_type=cl.device_type.GPU):
        """Initialise an OpenCL context bounded to this instance and loads the main kernels

        Keyword arguments:
        device_type --  preferred device type (DeviceType type from Constants.py) - default is GPU
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
        self.mf = cl.mem_flags

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
        """ Get the MILC prediction errors for a 3D image by means of OpenCL accelerated computation

            Keyword arguments:
            image --  a 3D numpy array (bitmap image)

            Return:
            a 3D numpy array of the same shape of "image", containing the prediction errors
        """

        mf = cl.mem_flags
        # Define the image format for the prediction errors
        err_format = cl.ImageFormat(channel_order=cl.channel_order.R, channel_type=DataType.CL_ERR.value)

        # Define the input image from the numpy 3D array
        source_image = cl.image_from_array(self.ctx, image)

        original_shape = numpy.shape(image)
        cl_shape = list(reversed(original_shape))  # inverted shape (pyOpenCL bug?)

        # output image
        output_image = cl.Image(
            self.ctx,
            mf.WRITE_ONLY,
            err_format,
            shape=cl_shape
        )

        # sampler. pixels out of range have a value of '0'
        sampler = cl.Sampler(self.ctx, False, cl.addressing_mode.CLAMP, cl.filter_mode.NEAREST)

        # enqueue kernel
        self.program.image_test(self.queue, original_shape, None, source_image, output_image, sampler)

        # read the resulting image into a numpy array
        output_data = numpy.empty(shape=cl_shape, dtype=DataType.ERR.value)
        cl.enqueue_read_image(self.queue, output_image, (0, 0, 0), cl_shape, output_data)

        return output_data.reshape(original_shape)

    def get_batch_size (self, head, size):
        """
        Computes the size of the group given the index of the first element
        and the total size.
        """
        _head = [head[0], head[1], head[2]]
        slices = size[2] - head[2]
        batch_size = 0

        if _head[1] > 0:
            length = _head[0] - _head[1] + 1
            if size[2] <= (_head[1] + 1):
                return int((slices * (2 * length + slices - 1))/2)
            else:
                n = _head[1] + 1
                batch_size = (n * (2 * length + n - 1))/2
                slices -= _head[1] + 1
                _head[1] = 0
                _head[0] -= 1

        length = _head[0] + 1
        n = length if length < slices else slices

        batch_size += (n * (2 * length - n + 1))/2
        return int(batch_size)

    def core_algo (self, errors):
        head = numpy.array([1, 2, 3])
        head_buffer = cl.Buffer(self.ctx, self.mf.COPY_HOST_PTR|self.mf.READ_ONLY, hostbuf=head)
        head_map = cl.enqueue_map_buffer(self.queue, buf=head_buffer, dtype=numpy.int, flags=cl.map_flags.WRITE, shape=(3), offset=0)[0]

        batchSize = 1
        globalDim = [batchSize]

        size = numpy.shape(errors)
        cycles = size[0] + size[1] + size[2]

