//
// SST Pattern Test
//

#include <algorithm>
#include <opencv2/opencv.hpp>
#include <netcdf.h>
#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include <sstream>
#include <omp.h>
#include <unistd.h>	// access(3)
#include <cmath>

#include "matrix.h"

#include <Eigen/Dense>
using namespace Eigen;
#include "connectedcomponents.h"

using namespace cv;

#define USED(x)	((void)(x))
#define SQ(x)    ((x)*(x))
#define SGN(A)   ((A) > 0 ? 1 : ((A) < 0 ? -1 : 0 ))
#define nelem(x) (sizeof(x)/sizeof((x)[0]))
#define SAVENC(X)	if(DEBUG)savenc("../data/thermalfronts/" #X ".nc", (X))
#define SAVENCF(X, Y)	if(DEBUG)savenc("../data/thermalfronts/" #Y ".nc", (X))
#define SAVENCN(X, Y)	if(DEBUG)savenc(Y, (X))
#define CHECKMAT(M, T)	CV_Assert((M).type() == (T) && (M).isContinuous())
#define CLAMP(A,L,H)    ((A)<=(L) ? (L) : (A)<=(H) ? (A) : (H))
#define isfinite(x)	(std::isfinite(x))

#define GRAD_THRESH 0.2
#define GRAD_LOW 0.1
#define DELTARANGE_THRESH 0.3
#define DELTAMAG_LOW (GRAD_LOW/2.0)
#define LAM2_THRESH	-0.05
#define SST_LOW 270
#define SST_HIGH 309
#define DELTA_LOW -1
#define DELTA_HIGH 4
#define OMEGA_LOW -5
#define OMEGA_HIGH 3
#define ANOMALY_HIGH 10
#define ANOMALY_LOW -10
#define ANOMALY_THRESH -8
#define ALBEDO_LOW 3
#define ALBEDO_HIGH 4
#define EDGE_THRESH 1
#define STD_THRESH 0.5

#define TQ_STEP 1
#define DQ_STEP 0.5	// 0.5 doesn't work on some examples
#define OQ_STEP 0.5
#define AQ_STEP 1

#define TQ_HIST_STEP 1
#define DQ_HIST_STEP 0.25
#define OQ_HIST_STEP OQ_STEP

template <typename T>
cv::Mat_<T> acspo_to_opencv(acspo::matrix<T> &mat)
{
    return cv::Mat_<T>(mat.rows(), mat.cols(), mat.ptr());
}

template <typename T>
acspo::matrix<T> opencv_to_acspo(cv::Mat_<T> &mat)
{
    return acspo::matrix<T>(mat.rows, mat.cols, (T *)mat.data).clone();
}

enum {
	DEBUG = 1,
	
	LUT_INVALID = -1,
	LUT_OCEAN = 0,
	LUT_CLOUD = 1,
	
	LUT_LAT_SPLIT = 4,
	
	COMP_INVALID = -1,	// invalid component
	COMP_SPECKLE = -2,	// component that is too small
	
	FRONT_SIDE_DIST = 7,	// multiplied to gradient dx/dy to obtain front sides
	MIN_FRONT_SIZE = 50,//100,
	MIN_FRONT_FRAGMENT = 10,
    MIN_CLUSTER_SIZE = 100
};

enum {
	MaskInvalid       = (1<<0),    		// or Valid
	MaskDay           = (1<<1),         // or Night
	MaskLand          = (1<<2),         // or Ocean
	MaskTwilightZone  = (1<<3),         // or No Twilight Zone
	MaskGlint         = (1<<4),         // or No Sun Glint
	MaskIce           = (1<<5),         // or No Ice
	
	MaskCloudOffset   = 6,              // first bit of cloud mask
	MaskCloud         = ((1<<7)|(1<<6)),
	MaskCloudClear    = 0,              // 0 - Clear
	MaskCloudProbably = (1<<6),         // 1 - Probably cloudy
	MaskCloudSure     = (1<<7),         // 2 - Confidently  cloudy
	MaskCloudInvalid  = ((1<<7)|(1<<6)),  // 3 - Irrelevant to SST (which includes land, ice and invalid pixels)
};

enum {
	LAND = -11,
	ICE_TEST = -10,
	COLD_CLOUD = -9,
	BT11_08_TEST = -8,
	NN_TEST = -7,
	BT12_TEST = -6,
	TEST_LAPLACIAN = -5,
	RATIO_TEST=-4,
	EIG_TEST = -3,
	TEST_UNIFORMITY = -2,
	TEST_CLOUD_BOUNDARY = -1,
	TEST_GRADMAG_LOW = 0,
	TEST_LOCALMAX,
	FRONT_GUESS,
};

enum {
	VIIRS_SWATH_SIZE = 16,
	MODIS_SWATH_SIZE = 10,
};

// utils.cc
acspo::matrix<float> row_neighbor_diffs(const acspo::matrix<float> &src);

void eprintf(const char *fmt, ...);
void logprintf(const char *fmt, ...);
char* estrdup(const char *s);
void *emalloc(size_t n);
const char *type2str(int type);
void diffcloudmask(const Mat &_old, const Mat &_new, Mat &_rgb);
std::string outpath(const char *inpath, const char *suffix);
double meann(const float *x, int n);
double maxn(const float *x, int n);
double corrcoef(const float *x, const float *y, int n);
void generate_sst_histogram_4d(const Mat1f &sst, const Mat1f &diff_11_12, const Mat_<schar> &mask, Mat1f &hist);
void histogram_check_4d(const Mat1f &sst, const Mat1f &diff_11_12, Mat_<schar> &frontmask, const Mat1f hist);
void histogram_check_3d(const Mat1f &sst,  Mat_<schar> &frontmask, const Mat1f hist, int flag2);
void generate_sst_histogram_3d(const Mat1f &sst,  const Mat_<schar> &mask, Mat1f &hist, int flag);
acspo::matrix<uchar> get_landborders(const acspo::matrix<uchar> &land_mask, int kernel_size);
void apply_land_mask(const acspo::matrix<uchar> &landmask, acspo::matrix<float> &clear);
void generate_cloud_histogram_3d(const acspo::matrix<float> &d1, const acspo::matrix<float> &d2, const acspo::matrix<float> &d3, const float *lows, const float *highs, const float *steps, const acspo::matrix<uchar> &mask, Mat1f &hist);
void check_cloud_histogram_3d(const acspo::matrix<float> &d1, const acspo::matrix<float> &d2, const acspo::matrix<float> &d3, const float *lows, const float *highs, const float *steps, const acspo::matrix<uchar> &mask, const Mat1f &hist, int flag, acspo::matrix<schar> &frontmask);

// io.cc
char *fileprefix(const char *path);
int	readvar(int ncid, const char *name, Mat&);
void ncfatal(int n, const char *fmt, ...);
void createvar(int ncid, const char *varname, const char *varunits, const char *vardescr, const Mat &data);
void writevar(int ncid, const char *varname, const Mat &data);
void savenc(const char*, const Mat&, bool compress=false);
void loadnc(const char*, Mat&);

// filters.cc
acspo::matrix<uchar> rectdilate(const acspo::matrix<uchar> &src, unsigned int size);
acspo::matrix<float> medfilter(const acspo::matrix<float> &src, unsigned int size);
std::tuple<acspo::matrix<float>, acspo::matrix<float> > gradient(acspo::matrix<float> &src);
acspo::matrix<float> stdfilter(const acspo::matrix<float> &src, int ksize);
acspo::matrix<float> localmax(const acspo::matrix<float> &dxx, const acspo::matrix<float> dyy, const acspo::matrix<float> dxy);
acspo::matrix<float> nanlogfilter(const acspo::matrix<float> &src, const int size, const float sigma, const float factor);
acspo::matrix<float> compute_eigenvals(const acspo::matrix<float> &bt08, const acspo::matrix<float> &bt11, const acspo::matrix<float> &bt12, const acspo::matrix<uchar> &border_mask);
acspo::matrix<uchar> nanmask(const acspo::matrix<float> &src);

struct ACSPOFile {
private:
	template <class T> void _readvar(const char *name, Mat_<T> &data);
public:
	int ncid;
	ACSPOFile(const char *path, int omode);
	void close();
	template <class T> acspo::matrix<T> readvar(const char *name);
};

template <typename T> 
acspo::matrix<T> ACSPOFile::readvar(const char *name)
{
    Mat_<T> data_cv;
    _readvar(name, data_cv);
    return opencv_to_acspo(data_cv);
}
