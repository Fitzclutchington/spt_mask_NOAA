//
// Image processing filters
//

#include "spt.h"

template <typename T>
acspo::matrix<T> get_window(const acspo::matrix<T> &src, unsigned int i, unsigned int j, unsigned int rows, unsigned int cols)
{
    unsigned int src_rows = src.rows(), src_cols = src.cols();
    unsigned int off_rows = (rows-1)/2, off_cols = (cols-1)/2;

    assert(rows <= src_rows && cols <= src_cols);
    assert(rows % 2 == 1 && cols % 2 == 1);

    acspo::matrix<T> win(rows, cols);

    for (unsigned int wi = 0; wi < rows; wi++) {
        for (unsigned int wj = 0; wj < cols; wj++) {
            int si = i+wi-off_rows;
            int sj = j+wj-off_cols;

            if (si < 0) {
                si = -si-1;
            } else if (si > src_rows-1) {
                si = 2*src_rows-si-1;
            }

            if (sj < 0) {
                sj = -sj-1;
            } else if (sj > src_cols-1) {
                sj = 2*src_cols-sj-1;
            }

            win(wi, wj) = src(si, sj);
        }
    }

    return win;
}

acspo::matrix<uchar> rectdilate(const acspo::matrix<uchar> &src, unsigned int size)
{
    unsigned int rows = src.rows(), cols = src.cols();

    assert(size % 2 == 1);
    assert(size <= rows && size <= cols);

    acspo::matrix<uchar> dst(src.size());

    #pragma omp parallel for collapse(2)
    for (unsigned int i = 0; i < rows; i++) {
        for (unsigned int j = 0; j < cols; j++) {

            auto win = get_window(src, i, j, size, size);
            unsigned int win_elem = win.elem();

            uchar val = src(i, j);

            for (unsigned int k = 0; k < win_elem; k++) {
                if (win(k) > dst(i, j)) {
                    val = win(k);
                }
            }

            dst(i, j) = val;
        }
    }

    return dst;
}

acspo::matrix<float> medfilter(const acspo::matrix<float> &src, unsigned int size)
{
    unsigned int rows = src.rows(), cols = src.cols();

    assert(size % 2 == 1);
    assert(size <= rows && size <= cols);

    acspo::matrix<float> dst(src.size());

    #pragma omp parallel for collapse(2)
    for (unsigned int i = 0; i < rows; i++) {
        for (unsigned int j = 0; j < cols; j++) {
            if (std::isnan(src(i, j))) {
                dst(i, j) = NAN;
                continue;
            }

            auto win = get_window(src, i, j, size, size);

            dst(i, j) = med(win);
        }
    }

    return dst;
}

static acspo::matrix<float> filter2d(const acspo::matrix<float> &src, const acspo::matrix<float> &kern)
{
    unsigned int rows = src.rows(), cols = src.cols();
    unsigned int kern_rows = kern.rows(), kern_cols = kern.cols(), kern_elem = kern.elem();

    assert(kern_rows % 2 == 1 && kern_cols % 2 == 1);
    assert(kern_rows <= rows && kern_cols <= cols);

    acspo::matrix<float> dst(src.size());

    #pragma omp parallel for collapse(2)
    for (unsigned int i = 0; i < rows; i++) {
        for (unsigned int j = 0; j < cols; j++) {
            if (std::isnan(src(i, j))) {
                dst(i, j) = NAN;
                continue;
            }

            auto win = get_window(src, i, j, kern_rows, kern_cols);

            dst(i, j) = 0;

            for (unsigned int k = 0; k < kern_elem; k++) {
                if (!std::isnan(win(k))) {
                    dst(i, j) += kern(k)*win(k);
                }
            }
        }
    }

    return dst;
}

static acspo::matrix<float> sepfilter2d(const acspo::matrix<float> &src, const std::vector<float> &kern_x, const std::vector<float> &kern_y)
{
    const unsigned int rows = kern_x.size(), cols = kern_y.size();

    acspo::matrix<float> kern(rows, cols);

    #pragma omp parallel for collapse(2)
    for (unsigned int i = 0; i < rows; i++) {
        for (unsigned int j = 0; j < cols; j++) {
            kern(i, j) = kern_y[i]*kern_x[j];
        }
    }

    return filter2d(src, kern);
}

std::tuple<acspo::matrix<float>, acspo::matrix<float> > gradient(acspo::matrix<float> &src)
{
	static const std::vector<float> h = { 0.036420, 0.248972, 0.429217, 0.248972, 0.036420 };
	static const std::vector<float> hn = { -0.036420, -0.248972, -0.429217, -0.248972, -0.036420 };
	static const std::vector<float> hp = { 0.108415, 0.280353, 0, -0.280353, -0.108415 };

	auto dx = sepfilter2d(src, h, hp);
	// We negate h here to fix the sign of dy.
	auto dy = sepfilter2d(src, hp, hn);

    return std::make_tuple(dx, dy);
}

// Separable blur implementation that can handle images containing NaN.
// OpenCV blur does not correctly handle such images.
//
// src -- source image
// dst -- destination image (output)
// ksize -- kernel size
//
acspo::matrix<float> nanblur(const acspo::matrix<float> &src, int ksize)
{
	std::vector<float> kern(ksize, 1.0/(float)ksize);
	return sepfilter2d(src, kern, kern);
}

// Find local maximum.
// Prototype: matlab/localmax.m
//
// sstmag -- SST gradient magnitude
// high, low -- (output)
// sigma -- standard deviation
//
acspo::matrix<float> localmax(const acspo::matrix<float> &dxx, const acspo::matrix<float> dyy, const acspo::matrix<float> dxy)
{
    const unsigned int rows = dxx.rows(), cols = dxx.cols();

    assert(dyy.size() == dxx.size());
    assert(dxy.size() == dxx.size());

    acspo::matrix<float> lam2(rows, cols);
	
    #pragma omp parallel for
	for (unsigned i = 0; i < rows*cols; i++) {
		float a = dxx(i)+dyy(i);
        float b = dxx(i)-dyy(i);
		float c = sqrt(b*b+4.0*dxy(i)*dxy(i));
		lam2(i) =  0.5*(a-c);
	}

    return lam2;
}

// Standard deviation filter, implemented as
//	dst = sqrt(blur(src^2) - blur(src)^2)
//
// src -- source image
// dst -- destination image (output)
// ksize -- kernel size
//
acspo::matrix<float> stdfilter(const acspo::matrix<float> &src, int ksize)
{
    acspo::matrix<float> dst;
    {
        auto b = nanblur(src, ksize);
        auto bs = nanblur(src*src, ksize);
	    dst = bs-b*b;
    }
	// avoid sqrt of nagative number
    dst.assign(0, dst < 0);

	return sqrt(dst);
}

// This function is equivalent to fspecial('log', n, sigma) in MATLAB.
//
// n -- width/height of kernel
// sigma -- standard deviation of the Gaussian
// dst -- the kernel (output)
//
static acspo::matrix<float> logkernel(int n, double sigma)
{
    // We need double precision internally to compute the kernel.
    acspo::matrix<double> dst(n, n);
	int h = (n-1)/2;
	double ss = sigma*sigma;
	
    #pragma omp parallel for collapse(2)
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			double x = j-h;
		    double y = i-h;
			dst(i, j) = std::exp(-(x*x+y*y)/(2.0*ss));
		}
	}

	double total = sum(dst);
	
    #pragma omp parallel for collapse(2)
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			double x = j-h;
		    double y = i-h;
			dst(i, j) *= x*x+y*y-2.0*ss;
		}
	}

	dst /= ss*ss*total;
	dst -= mean(dst);	// make sum of filter equal to 0

    return dst.convert<float>();
}

// Create a mask of where source image is NaN.
//
// _src -- source image
// _dst -- NaN mask (output)
//
acspo::matrix<uchar> nanmask(const acspo::matrix<float> &src)
{
    unsigned int rows = src.rows(), cols = src.cols();
	auto dst = acspo::matrix<uchar>::zeros(rows, cols);
    #pragma omp parallel for
	for(unsigned int i = 0; i < rows*cols; i++) {
		if (std::isnan(src(i))) {
			dst(i) = 255;
		}
	}
    return dst;
}

// Apply Laplacian of Gaussian (LoG) filter to an image containing NAN.
//
// src -- source image
// size -- width/height of LoG kernel
// sigma -- standard deviation of the Gaussian used in LoG kernel
// factor -- kernel is multiplied by this number before application
// dst -- destination image (output)
//
acspo::matrix<float> nanlogfilter(const acspo::matrix<float> &src, const int size, const float sigma, const float factor)
{
	auto kern = logkernel(size, sigma);
    kern *= factor;
    return filter2d(src, kern);
}

acspo::matrix<float> compute_eigenvals(const acspo::matrix<float> &bt08, const acspo::matrix<float> &bt11, const acspo::matrix<float> &bt12, const acspo::matrix<uchar> &border_mask)
{
  int y,x,i,j;
  int y_delta = 1;
  int x_delta = 1;
  int height = bt08.rows();
  int width = bt08.cols();

  acspo::matrix<float> eigen(height,width);
  
  int min_num = (2*y_delta +1) *(2*x_delta + 1)/2;

  #pragma omp parallel for collapse(2) private(y,x,i,j)
  for(y=y_delta;y<height-y_delta;y++){
    for(x=x_delta;x<width-x_delta;x++){
      int count_dim;
      float bt08_sum,bt11_sum,bt12_sum,count,window_sum,row_sum, res_mean;
      float temp_bt08;
      float temp_bt11;
      float temp_bt12;
      std::vector<float> valid_bt08;
      std::vector<float> valid_bt11;
      std::vector<float> valid_bt12;
      float bt08_mean;
      float bt11_mean;
      float bt12_mean;
      Vector3f ones(1,1,1);
      Vector3f e1;
      MatrixXf r;
      Matrix3f A;

      if(isfinite(bt11(y,x))){
        // calc first window
        // we know that first left are nans so we don't calculate left inds     
        bt08_sum=bt11_sum=bt12_sum=0;
        valid_bt08.clear();
        valid_bt11.clear();
        valid_bt12.clear();
        for(i=-y_delta;i<y_delta+1;i++){
          for(j=-x_delta;j<x_delta+1;j++){              
           
	          //t = ((((cur_ind+k)%FILTER_TIME_SIZE)+FILTER_TIME_SIZE) % FILTER_TIME_SIZE);
	          temp_bt08 = bt08(y+i,x+j);
	          temp_bt11 = bt11(y+i,x+j);
	          temp_bt12 = bt12(y+i,x+j);

	          if(!std::isnan(temp_bt08) && !std::isnan(temp_bt11) && !std::isnan(temp_bt12)){
	            valid_bt08.push_back(temp_bt08);
	            valid_bt11.push_back(temp_bt11);
	            valid_bt12.push_back(temp_bt12);

	            bt08_sum+= temp_bt08;
	            bt11_sum+=temp_bt11;
	            bt12_sum+=temp_bt12;
	          }
          }
        }
  
        //if numberof pixels in window is greater tan threshold
        // calculate the mean of the norm of the pixels
        // projected into the second eigenvector
        count = valid_bt08.size();
        count_dim = valid_bt08.size();
        //printf("count = %f\n",count);
        //printf("min_num %d\n",min_num);
        if(count > min_num){
        
          //printf("in count\n");
          bt08_mean =bt08_sum/count;
          bt11_mean =bt11_sum/count;
          bt12_mean =bt12_sum/count;

          MatrixXf window(count_dim,3);
          for(i = 0; i < count; ++i){
            window(i,0) = valid_bt08[i] - bt08_mean;
            window(i,1) = valid_bt11[i] - bt11_mean;
            window(i,2) = valid_bt12[i] - bt12_mean;
          }
          
          A = (window.transpose()*window);
          e1 = A*(A*ones);
          e1/=sqrt(e1.transpose()*e1);
          r = window - (window*e1)*e1.transpose();
          window_sum =0;

          for(i = 0; i < 3; ++i){
          	row_sum = 0;
          	for(j = 0; j < count; ++j){
          		row_sum += r(j,i) * r(j,i);
          	}
          	row_sum = sqrt(row_sum);
          	window_sum += row_sum;
          }
          res_mean = window_sum/ 3.0;

          eigen(y,x) = res_mean;
        }
      }
    }
  }

  return eigen;
}
