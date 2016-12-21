/*
#ifndef DEV_TYPE
#define DEV_TYPE 1
#endif
*/

unsigned char checkPixel(int value) {
	if (value > 255)
		return (unsigned char)255;

	if (value < 0)
		return (unsigned char)0;

	return (unsigned char)value;
}

unsigned short mapError(int error) {
	int uError = abs(error);
	unsigned short errorToWrite = 2 * uError;

	if (error < 0)
		errorToWrite -= 1;

	return errorToWrite;
}

int unmapError(unsigned short mappedError) {
	int trueError = 0;
	int sign = 1;

	if ((mappedError % 2) == 0) {
		trueError = mappedError / 2;
		sign = 0;
	} else {
		trueError = (mappedError + 1) / 2;
		sign = 1;
	}

	trueError *= ((sign) ? -1 : 1);

	return trueError;
}

/*bool setError(__global unsigned short *img, int2 coord, int slice, int width, int height, unsigned short value) {
	if (!verifyCoordinates(coord, width, height))
		return false;

	img[(slice * width * height) + (coord.y * width + coord.x)] = value;

	return true;
}
#if DEV_TYPE == 1
#endif
#if DEV_TYPE == 2
*/

unsigned char linearized_median_predictor(read_only image3d_t srcImg, sampler_t sampler, int4 coord)
{
	unsigned char pixelA = checkPixel(read_imageui(srcImg, sampler, (int4)(0, coord.y, coord.z - 1, 0)).x);
	unsigned char pixelB = checkPixel(read_imageui(srcImg, sampler, (int4)(0, coord.y - 1, coord.z, 0)).x);
	unsigned char pixelC = checkPixel(read_imageui(srcImg, sampler, (int4)(0, coord.y - 1, coord.z - 1, 0)).x);

	int predicted = (int)((2 * (pixelA + pixelB)) / 3) - (pixelC / 3);

	return checkPixel(predicted);
}

unsigned char distances_based_linearized_median_predictor(read_only image3d_t srcImg, sampler_t sampler, int4 coord)
{
	unsigned char pixelA0 = checkPixel(read_imageui(srcImg, sampler, (int4)(coord.x, coord.y , coord.z - 1, 0)).x);
	unsigned char pixelB0 = checkPixel(read_imageui(srcImg, sampler, (int4)(coord.x, coord.y - 1 , coord.z, 0)).x);
	unsigned char pixelC0 = checkPixel(read_imageui(srcImg, sampler, (int4)(coord.x, coord.y - 1 , coord.z - 1, 0)).x);

	unsigned char pixelA1 = checkPixel(read_imageui(srcImg, sampler, (int4)(coord.x - 1, coord.y, coord.z - 1, 0)).x);
	unsigned char pixelB1 = checkPixel(read_imageui(srcImg, sampler, (int4)(coord.x - 1, coord.y - 1, coord.z, 0)).x);
	unsigned char pixelC1 = checkPixel(read_imageui(srcImg, sampler, (int4)(coord.x - 1, coord.y - 1, coord.z - 1, 0)).x);

	unsigned char pixelX1 = checkPixel(read_imageui(srcImg, sampler, (int4)(coord.x - 1, coord.y, coord.z, 0)).x);

	int deltaA = pixelA0 - pixelA1;
	int deltaB = pixelB0 - pixelB1;
	int deltaC = pixelC0 - pixelC1;

	int deltaAB = deltaA + deltaB;

	int predicted = pixelX1 + ((deltaAB + deltaAB - deltaC) / 3);

	return checkPixel(predicted);
}

__kernel void image_test(read_only image3d_t srcImg, write_only image3d_t dest, sampler_t sampler)
{
    unsigned int x = get_global_id(0);
    unsigned int y = get_global_id(1);
    unsigned int z = get_global_id(2);
    unsigned short error = 0;

    int4 coord = (int4)(z, y, x, 0);

    unsigned char myVal = checkPixel(read_imageui(srcImg, sampler, coord).x);
    unsigned char predicted = 0;

    if (z > 0)
    {
        predicted = distances_based_linearized_median_predictor(srcImg, sampler, coord);
    }
    else
    {
        predicted = linearized_median_predictor(srcImg, sampler, coord);
    }

    error = mapError (myVal - predicted);

    write_imageui(dest, (int4)(coord), error);

}

//#endif