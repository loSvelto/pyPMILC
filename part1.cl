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

    if (z > 0) {
        predicted = distances_based_linearized_median_predictor(srcImg, sampler, coord);
    }
    else {
        predicted = linearized_median_predictor(srcImg, sampler, coord);
    }

    error = mapError (myVal - predicted);

    write_imageui(dest, (int4)(coord), error);

}

void translate (int gid, constant int head[3], int *_3D)
{
	int x = head[0],
		 y = head[1],
		 z = 0;

	if (y > 0)
	{
		int len = x - y + 1;
		int maxSeries = ((y + 1) * (2 * len + y)) / 2;
		if (gid < maxSeries)
		{
			int b = 2 * len - 1;
			int delta = (b * b) + (8 * gid);
			float sqr = sqrt((float)delta);
			z = floor((float)((sqr - b) / 2));
			int displ = gid - (z * (2 * len + (z - 1))) / 2;
			_3D[0] = head[0] - displ;
			_3D[1] = head[1] - z + displ;
			_3D[2] = head[2] + z;
			return;
		}
		else
		{
			gid = gid - maxSeries;
			x = head[0] - 1;
			z = head[1] + 1;
		}
	}
	int len = x + 1;
	int b = 2 * len + 1;
	int delta = (b * b) - (8 * gid);
	float sqr = sqrt((float)delta);
	int _z = floor((float)((b - sqr) / 2));
	int displ = gid - (_z * (2 * len - (_z - 1))) /2;
	_3D[0] = x - _z - displ;
	_3D[1] = displ;
	_3D[2] = _z + z + head[2];
	return;
}

unsigned char getPrediction_gpu (image3d_t image,
		sampler_t sampler,
		int i, int j, int k)
{
	int sx = 0;
	unsigned char sa = 0, sb = 0, sc = 0,
					 _sx = 0, _sa = 0, _sb = 0, _sc = 0;

	sa = (read_imageui(image, sampler, (int4)(i-1, j, k, 0))).x;
	sb = (read_imageui(image, sampler, (int4)(i, j-1, k, 0))).x;
	sc = (read_imageui(image, sampler, (int4)(i-1, j-1, k, 0))).x;
	_sx = (read_imageui(image, sampler, (int4)(i, j, k-1, 0))).x;
	_sa = (read_imageui(image, sampler, (int4)(i-1, j, k-1, 0))).x;
	_sb = (read_imageui(image, sampler, (int4)(i, j-1, k-1, 0))).x;
	_sc = (read_imageui(image, sampler, (int4)(i-1, j-1, k-1, 0))).x;

	if (k == 0)
		sx = (int)((2 * (sa + sb)) / 3) - (sc / 3);
	else
		sx = (int)_sx + ((2 * ((sa - _sa) + (sb - _sb)) - (sc - _sc)) / 3);

	unsigned char prediction = checkPixel(sx);
	return prediction;
}

kernel void executeParallelDecompression (constant int head[3],
		 read_only image3d_t errors,
		 read_only image3d_t image,
		 write_only image3d_t output,
		 sampler_t sampler,
		 int width, int height, int numOfSlices)
{
	int pos[3] = {0, 0, 0};
	int gid = get_global_id(0);
	translate(gid, head, pos);
	int i = pos[0];
	int j = pos[1];
	int k = pos[2];

    int4 coord = (int4)(i, j, k, 0);

	unsigned char predicted = getPrediction_gpu(image, sampler, i, j, k);
	unsigned short mappedError = (read_imageui(errors, sampler, coord)).x;
	unsigned char value = predicted + unmapError(mappedError);

	write_imageui(output, coord, value);
}

kernel void map_test (constant int head[3], write_only image3d_t output) {
    unsigned short head0 = head[0];
    unsigned short head1 = head[1];
    unsigned short head2 = head[2];
    write_imageui(output, (int4)(0, 0, 0, 0), head0);
    write_imageui(output, (int4)(0, 0, 1, 0), head1);
    write_imageui(output, (int4)(0, 0, 2, 0), head2);
}

//#endif