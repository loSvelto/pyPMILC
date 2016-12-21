import numpy
from Constants import DataType


def load_image(filename, height, width, depth=None):

    image_data = numpy.fromfile(filename, DataType.PIXEL.value)
    if not depth:
        depth = int(image_data.size / height / width)
    image_data = numpy.reshape(image_data, (height, width, depth))

    return image_data


def save_errors(filename, errors):

    numpy.save(file=filename, arr=errors)


def load_errors(filename):

    return numpy.load(filename)


def message_pause_exit (message) :
    print(message, "Press any key to exit the program...")
    input()
    exit()
