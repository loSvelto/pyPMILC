import numpy
from Constants import DataType


def _check_pixel (value, max):

    if value > max or value < 0:
        return 0
    else:
        return value


def _linearized_median_predictor (image, index):

    x = index[0]
    y = index[1]

    pixelA = _check_pixel(image[x-1,y,0] if x > 0 else 0, DataType.PIXEL_MAX.value)
    pixelB = _check_pixel(image[x,y-1,0] if y > 0 else 0, DataType.PIXEL_MAX.value)
    pixelC = _check_pixel(image[x-1,y-1,0] if (x > 0 and y > 0) else 0, DataType.PIXEL_MAX.value)

    prediction = _check_pixel(int(((2 * (pixelA + pixelB)) / 3) - int((pixelC / 3))), DataType.PIXEL_MAX.value)

    return prediction


def _distances_based_linearized_median_predictor(image, index):

    x = index[0]
    y = index[1]
    z = index[2]

    pixelA0 = _check_pixel(image[x-1, y, z] if x > 0 else 0, DataType.PIXEL_MAX.value)
    pixelB0 = _check_pixel(image[x, y-1, z] if y > 0 else 0, DataType.PIXEL_MAX.value)
    pixelC0 = _check_pixel(image[x-1, y-1, z] if x > 0 and y > 0 else 0, DataType.PIXEL_MAX.value)

    pixelA1 = _check_pixel(image[x-1,y,z-1] if x > 0 else 0, DataType.PIXEL_MAX.value)
    pixelB1 = _check_pixel(image[x,y-1,z-1] if y > 0 else 0, DataType.PIXEL_MAX.value)
    pixelC1 = _check_pixel(image[x-1,y-1,z-1] if x > 0 and y > 0 else 0, DataType.PIXEL_MAX.value)

    pixelX1 = _check_pixel(image[x,y,z - 1], DataType.PIXEL_MAX.value)

    deltaA = pixelA0 - pixelA1
    deltaB = pixelB0 - pixelB1
    deltaC = pixelC0 - pixelC1

    deltaAB = deltaA + deltaB

    predicted = _check_pixel(int(pixelX1 + int(((deltaAB + deltaAB - deltaC) / 3))), DataType.PIXEL_MAX.value)

    return predicted


def _mapError(error):

    uError = abs(error)
    errorToWrite = 2 * uError

    if error < 0:
        errorToWrite -= 1

    return errorToWrite


def prediction_errors (image):

    errors = numpy.empty_like(image)

    for index, x in numpy.ndenumerate(image):
        prediction = 0
        if index[2] > 0:
            prediction = _distances_based_linearized_median_predictor(image, index)
        else:
            prediction = _linearized_median_predictor(image, index)
        errors[index] = _mapError(x - prediction)
        # errors[index] = prediction

    return errors
