#include "lab1.h"
static const unsigned W = 640;
static const unsigned H = 640;
static const unsigned NFRAME = 360;
#include <cmath>                                                                                    
#define X_CENTER_1 0   // x pixel position of center                                                 
#define Y_CENTER_1 0   // y pixel position of center                                                 
#define X_CENTER_2 640   // x pixel position of center                                                 
#define Y_CENTER_2 640   // y pixel position of center                                                 
#define RADIUS 160   // approximate radius of circular wave train, in pixels                        
#define WAVELENGTH 80  // wavelength of ripples, in pixels   
#define TRAINWIDTH 3.4 // approximate width of wave train, in wavelengths                              
#define SUPERPHASE 1.5 // phase vel. / group vel. (irrelevant for stills)                              
#define PI 3.14159265                                                                                   
// returns a number from -1.0 to 1.0                                                               
double depth(int x, int y, int t) {
  double rate = sqrt(t)*20;
  rate = rate > 320 ? 320 : rate;
  double dx1 = x - (X_CENTER_1 + rate) ; // or int, if the center coords are ints
  double dy1 = y - (Y_CENTER_1 + rate);      
  double dx2 = x - (X_CENTER_2 - rate) ; // or int, if the center coords are ints
  double dy2 = y - (Y_CENTER_2 - rate);  
  double r1 = (sqrt(dx1*dx1+dy1*dy1)-RADIUS)/WAVELENGTH ; 
  double r2 = (sqrt(dx2*dx2+dy2*dy2)-RADIUS)/WAVELENGTH ;
  double k1 = r1 - (1-SUPERPHASE)*RADIUS/WAVELENGTH ;
  double a1 = 1 / (1.0 + (r1/TRAINWIDTH)*(r1/TRAINWIDTH));
  double k2 = r2 - (1-SUPERPHASE)*RADIUS/WAVELENGTH ;
  double a2 = 1 / (1.0 + (r2/TRAINWIDTH)*(r2/TRAINWIDTH));
  return (a1 * sin(k1*2*PI - t/2) + a2 * sin(k2*2*PI - t/2))/2;             
}

struct Lab1VideoGenerator::Impl {
	int t = 0;
	unsigned char map[H][W];
};

Lab1VideoGenerator::Lab1VideoGenerator(): impl(new Impl) {
}

Lab1VideoGenerator::~Lab1VideoGenerator() {}

void Lab1VideoGenerator::get_info(Lab1VideoInfo &info) {
	info.w = W;
	info.h = H;
	info.n_frame = NFRAME;
	// fps = 24/1 = 24
	info.fps_n = 24;
	info.fps_d = 1;
};


void Lab1VideoGenerator::Generate(uint8_t *yuv) {
	for(int i = 0; i < H; i++){
		for(int j = 0; j < W; j++){
			impl->map[i][j] = 175 + depth(i, j, impl->t) * 75;
		}
	}
	cudaMemcpy(yuv, impl->map, sizeof(impl->map), cudaMemcpyHostToDevice);
	cudaMemset(yuv+W*H, 200, W*H/4);
	cudaMemset(yuv+W*H+W*H/4, 90, W*H/4);
	++(impl->t);
}
