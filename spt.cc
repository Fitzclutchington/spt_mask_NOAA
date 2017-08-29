//
// SST Pattern Test
//

#include "spt.h"
#include "fastBilateral.hpp"

// thernal fronts and their sides
enum {
	FRONT_INVALID = -1,
	FRONT_INIT = 0,	// initial fronts
	FRONT_THIN,	// thinned fronts
	FRONT_LEFT,	// left side
	FRONT_RIGHT,	// right side
};



void
isnanmat(const Mat1f &src, Mat1b &dst)
{
	dst.create(src.size());
	for(size_t i = 0; i < src.total(); i++){
		dst(i) = std::isnan(src(i)) ? 255 : 0;
	}
}



// Connected components wrapper that limits the minimum size of components.
//
// mask -- the image to be labeled
// connectivity -- 8 or 4 for 8-way or 4-way connectivity respectively
// lim -- limit on the minimum size of components
// cclabels -- destination labeled image (output)
//
static int
connectedComponentsWithLimit(const Mat &mask, int connectivity, int lim, Mat1i &cclabels)
{
	Mat stats, centoids;
	int ncc = connectedComponentsWithStats(mask, cclabels, stats, centoids, connectivity, CV_32S);
	
	// Remove small connected components and rename labels to be contiguous.
	// Also, set background label 0 (where mask is 0) to -1.
	std::vector<int> ccrename(ncc);
	ccrename[0] = COMP_INVALID;
	int newlab = 0;
	for(int lab = 1; lab < ncc; lab++){
		if(stats.at<int>(lab, CC_STAT_AREA) >= lim){
			ccrename[lab] = newlab++;
		}else{
			ccrename[lab] = COMP_SPECKLE;
		}
	}
	ncc = newlab;
	for(size_t i = 0; i < mask.total(); i++){
		cclabels(i) = ccrename[cclabels(i)];
	}

	return ncc;
}




// Thin fronts based on SST gradient.
// Prototype: matlab/front_thinning.m
//
// frontsimg -- front labels
// dy_sst -- gradient in Y direction
// dx_sst -- gradient in X direction
// mag_grad_sst -- gradient magnitude
// thinnedf -- front labels containing thinned fronts (output)
//
static void
thinfronts(const Mat_<schar> &frontsimg, const Mat1f &dy_sst, const Mat1f &dx_sst, const Mat1f &mag_grad_sst,
	const Mat1f &logmag, Mat_<schar> &thinnedf)
{
	thinnedf.create(frontsimg.size());
	thinnedf = FRONT_INVALID;
	int i = 0;
	int alpha_step = 2;
	for(int y = 0; y < frontsimg.rows; y++){
		for(int x = 0; x < frontsimg.cols; x++){
			if(frontsimg(i) == FRONT_INIT){
				double dy = dy_sst(i) / mag_grad_sst(i);
				double dx = dx_sst(i) / mag_grad_sst(i);

				int maxy = y;
				int maxx = x;
				float maxg = logmag(maxy, maxx);
				for(int alpha = -alpha_step; alpha <= alpha_step; alpha++){
					int yy = round(y + alpha*dx);
					int xx = round(x - alpha*dy);
					
					if(0 <= yy && yy < frontsimg.rows
					&& 0 <= xx && xx < frontsimg.cols
					&& logmag(yy, xx) > maxg){
						maxy = yy;
						maxx = xx;
						maxg = logmag(yy, xx);
					}
				}
				if(frontsimg(maxy, maxx) == FRONT_INIT){
					thinnedf(maxy, maxx) = FRONT_THIN;
				}
			}
			i++;
		}
	}
}



enum {
	ACSPOTestsEasycloudMask = 0xc6
};

// Attempt to connect broken fronts by using cos-similarity of gradient vectors.
// Each pixels within a window is compared to the pixel at the center of the
// window.
// Prototype: matlab/front_connect.m
//
// fronts -- fronts containing only initial fronts (FRONT_INIT) (intput & output)
// dx_sst -- gradient in x direction
// dy_sst -- gradient in y direction
// mag_grad_sst -- SST gradient magnitude
// easyclouds -- guaranteed cloud based on various thresholds
// lam2 -- local max
//
static void
connectfronts(Mat_<schar> &fronts, const Mat1f &dx_sst, const Mat1f &dy_sst, const Mat1f &sst,
	const Mat1f &mag_grad_sst, const Mat1f &logmag, const Mat1b &easyclouds, const Mat1f &lam2)
{
	Mat_<schar> thinnedf;
	
	const int W = 11;	// window width/height
	const int mid = fronts.cols*(W/2) + (W/2);	// pixel at center of window
	
	Mat1b rejected(fronts.size());
	rejected = 0;
	
	Mat1f tmp;
	exp(100*(lam2+0.01), tmp);
	Mat1f llam = 1.0/(1+tmp);
	
	Mat1b valid =  (fronts != FRONT_INIT) & (mag_grad_sst > GRAD_LOW) & ~easyclouds;
	
	for(int iter = 0; iter < 10; iter++){
		int i = 0;
		// For each pixel with full window, where the pixel
		// is top left corner of the window
		for(int y = 0; y < fronts.rows-W+1; y++){
			for(int x = 0; x < fronts.cols-W+1; x++){
				if(fronts(i + mid) == FRONT_INIT){
					double cdy_sst = dy_sst(i + mid);
					double cdx_sst = dx_sst(i + mid);
					double csst = sst(i + mid);
					double max = 0;
					int k = i;
					int argmax = i + mid;
					for(int yy = y; yy < y+W; yy++){
						for(int xx = x; xx < x+W; xx++){
							if(valid(k) != 0 && fabs(sst(k) - csst) < 0.5 && rejected(k) == 0){
								// cos-similarity
								//double sim = llam(k) * (dy_sst(k)*cdy_sst + dx_sst(k)*cdx_sst);
								double sim = (dy_sst(k)*cdy_sst + dx_sst(k)*cdx_sst);
								if(sim > max){
									max = sim;
									argmax = k;
								}
							}
							k++;
						}
						k += fronts.cols - W;
					}
					fronts(argmax) = FRONT_INIT;
				}
				i++;
			}
		}
		thinfronts(fronts, dy_sst, dx_sst, mag_grad_sst, logmag, thinnedf);
		
		for(size_t i = 0; i < thinnedf.total(); i++){
			switch(thinnedf(i)){
			default:
				if(fronts(i) == FRONT_INIT){
					fronts(i) = FRONT_INVALID;
					rejected(i) = 255;
				}
				break;
			case FRONT_THIN:
				fronts(i) = FRONT_INIT;
				break;
			}
		}
	}
}

static void
createsptmask(const Mat1b &newacspo, const Mat_<schar> &frontsimg, Mat1b &sptmask)
{
 	sptmask = 0;

 	// add fronts to sptmask
 	for(size_t i = 0; i < sptmask.total(); i++){
 		sptmask(i) = (newacspo(i)&MaskCloud) >> MaskCloudOffset;
 		if((newacspo(i)&MaskCloud) == MaskCloudClear && frontsimg(i) == FRONT_INIT){
 			sptmask(i) |= (1<<2);
 		}
 	}
}

// Write spt into NetCDF dataset ncid as variable named "spt_mask".
//
static void
writesptmask(int ncid, const Mat1b &sptmask)
{
	int n, varid, ndims, dimids[2];
	nc_type xtype;
	size_t len;
	
	const char varname[] = "spt_mask";
	const char varunits[] = "none";
	const char vardescr[] = "SPT mask packed into 1 byte: bits1-2 (00=clear; 01=probably clear; 10=cloudy; 11=clear-sky mask undefined); bit3 (0=no thermal front; 1=thermal front)";

	// chunk sizes used by acspo_mask
	const size_t chunksizes[] = {1024, 3200};
	
	// It's not possible to delete a NetCDF variable, so attempt to use
	// the variable if it already exists. Create the variable if it does not exist.
	n = nc_inq_varid(ncid, varname, &varid);
	if(n != NC_NOERR){
		n = nc_inq_dimid(ncid, "scan_lines_along_track", &dimids[0]);
		if(n != NC_NOERR){
			ncfatal(n, "nc_inq_dimid failed");
		}
		n = nc_inq_dimid(ncid, "pixels_across_track", &dimids[1]);
		if(n != NC_NOERR){
			ncfatal(n, "nc_inq_dimid failed");
		}
		
		n = nc_def_var(ncid, varname, NC_UBYTE, nelem(dimids), dimids, &varid);
		if(n != NC_NOERR){
			ncfatal(n, "nc_def_var failed");
		}
		n = nc_def_var_chunking(ncid, varid, NC_CHUNKED, chunksizes);
		if(n != NC_NOERR){
			ncfatal(n, "nc_def_var_chunking failed");
		}
		n = nc_def_var_deflate(ncid, varid, 0, 1, 1);
		if(n != NC_NOERR){
			ncfatal(n, "setting deflate parameters failed");
		}
		
		n = nc_put_att_text(ncid, varid, "UNITS", nelem(varunits)-1, varunits);
		if(n != NC_NOERR){
			ncfatal(n, "setting attribute UNITS failed");
		}
		n = nc_put_att_text(ncid, varid, "Description", nelem(vardescr)-1, vardescr);
		if(n != NC_NOERR){
			ncfatal(n, "setting attribute Description failed");
		}
	}
	
	// Varify that the netcdf variable has correct type and dimensions.
	n = nc_inq_var(ncid, varid, nullptr, &xtype, &ndims, dimids, nullptr);
	if(n != NC_NOERR){
		ncfatal(n, "nc_inq_var failed");
	}
	if(xtype != NC_UBYTE){
		eprintf("variable type is %d, want %d\n", xtype, NC_UBYTE);
	}
	if(ndims != 2){
		eprintf("variable dims is %d, want 2\n", ndims);
	}
	for(int i = 0; i < 2; i++){
		n = nc_inq_dimlen(ncid, dimids[i], &len);
		if(n != NC_NOERR){
			ncfatal(n, "nc_inq_dimlen failed");
		}
		if(len != static_cast<size_t>(sptmask.size[i])){
			eprintf("dimension %d is %d, want %d\n", i, len, sptmask.size[i]);
		}
	}
	

	// Write data into netcdf variable.
	n = nc_put_var_uchar(ncid, varid, sptmask.data);
	if(n != NC_NOERR){
		ncfatal(n, "nc_putvar_uchar failed");
	}
}


static void
remove_speckles(const acspo::matrix<schar> &front_temp, acspo::matrix<schar> &frontmask, int flag)
{
	acspo::matrix<uchar> mask_temp(frontmask.size());

	// create binary mask where temporary mask == flag
	mask_temp.assign(0, front_temp != flag);
	mask_temp.assign(1, front_temp == flag);

	// erode to remove speckles
	mask_temp = recterode(mask_temp,5);
	// dilate to bring back edges
	mask_temp = rectdilate(mask_temp,5);

	// insert flags into frontmask
	frontmask.assign(flag, mask_temp!=0);
}


static acspo::matrix<schar>
maskfronts(const acspo::matrix<float> &sst, const acspo::matrix<float> &mag_grad_sst, const acspo::matrix<float> &bt08, const acspo::matrix<float> &bt11, const acspo::matrix<float> &mag_grad_bt11, const acspo::matrix<float> &bt12, const acspo::matrix<float> &mag_grad_bt12, const acspo::matrix<float> &eigen, const acspo::matrix<float> &laplacian_sst, const acspo::matrix<float> &lam2, const acspo::matrix<float> &medianSST, const acspo::matrix<uchar> ice_mask, const acspo::matrix<uchar> &land_mask, const acspo::matrix<uchar> &border_mask)
{
    float delta_n = 0.1;
	float T_low = 271.15;
	float delta_Lam = 0.01;
	float thresh_mag = 0.1;

	float thresh_L = 0.8;
	float eigen_thresh = 2;
	float median_thresh = 0.5;
	float mag_ratio_thresh = 0.5;
	float std_thresh = 0.5;

    acspo::matrix<schar> front_mask(sst.size());

	auto magdiff_bt12 = mag_grad_sst - mag_grad_bt12 ;
 	auto magdiff_bt11 = mag_grad_sst - mag_grad_bt11 ;

	auto median_diff = abs(medianSST - sst);

    auto std_median = stdfilter(medianSST, 7);
    auto std_sst = stdfilter(sst, 7);

	front_mask.assign(FRONT_GUESS, TEST_GRADMAG_LOW, mag_grad_sst > thresh_mag);

	front_mask.assign(EIG_TEST,             eigen > eigen_thresh);
	front_mask.assign(TEST_LOCALMAX,        lam2 > -delta_Lam && front_mask == FRONT_GUESS);
	front_mask.assign(BT12_TEST,            sst < bt12);
	front_mask.assign(TEST_UNIFORMITY,      median_diff > median_thresh);
	front_mask.assign(TEST_CLOUD_BOUNDARY,  magdiff_bt12 < -delta_n);
	front_mask.assign(TEST_LAPLACIAN,       laplacian_sst > thresh_L);
	front_mask.assign(COLD_CLOUD,           sst < T_low);
	front_mask.assign(RATIO_TEST,           mag_grad_sst > thresh_mag && (magdiff_bt11/mag_grad_sst) > mag_ratio_thresh);	
	front_mask.assign(ICE_TEST,             ice_mask != 0);
	front_mask.assign(LAND,                 land_mask != 0);
    front_mask.assign(BT11_08_TEST,         bt11 < bt08);

    return front_mask;
}


class SPT {
	int ncid;

    acspo::matrix<uchar> cloud_mask;
    acspo::matrix<uchar> ice_mask;
    acspo::matrix<uchar> land_mask;
    acspo::matrix<uchar> usst_mask;
	acspo::matrix<uchar> border_mask;

	acspo::matrix<float> dx_sst;
	acspo::matrix<float> dy_sst;
	acspo::matrix<float> bt12;
	acspo::matrix<float> bt11;
    acspo::matrix<float> mag_grad_bt11;
    acspo::matrix<float> mag_grad_bt12;
	acspo::matrix<float> bt03;
	acspo::matrix<float> bt08;
	acspo::matrix<float> mag_grad_sst;
	acspo::matrix<float> sst;
	acspo::matrix<float> sst_ref;
	acspo::matrix<float> logmag;
	acspo::matrix<float> laplacian_sst;
	acspo::matrix<float> lam2;
	acspo::matrix<float> eigen;
	acspo::matrix<float> medSST;
	acspo::matrix<uchar> nan_mask;

public:
	SPT(ACSPOFile &f);
	void run();
};

SPT::SPT(ACSPOFile &f)
{
	ncid = f.ncid;
	
    logprintf("reading data ...\n");

	sst = f.readvar<float>("sst_regression");
	sst_ref = f.readvar<float>("sst_reynolds");
	bt11 = f.readvar<float>("brightness_temp_chM15");
	bt12 = f.readvar<float>("brightness_temp_chM16");
    bt03 = f.readvar<float>("brightness_temp_chM12");
    bt08 = f.readvar<float>("brightness_temp_chM14");

    {
	    auto acspo_mask = f.readvar<uchar>("acspo_mask");
        // ice is bit 6
    	ice_mask = acspo_mask & (1 << 5);
        // land is bit 3
	    land_mask = acspo_mask & (1 << 2);
        // cloud is bits 7 and 8
	    cloud_mask = acspo_mask & ((1 << 6) | (1 << 7));
    }

    {
	    auto acspo_tests = f.readvar<uchar>("individual_clear_sky_tests_results");
        // sst uniformity filter (usst) is bit 7
	    usst_mask = acspo_tests & (1 << 6);
    }

	logprintf("computing mag_grad_sst, etc....\n");
	
    medSST = medfilter(sst, 5);

    {
        acspo::matrix<float> dx_bt11, dy_bt11;
	    std::tie(dx_bt11, dy_bt11) = gradient(bt11);
        mag_grad_bt11 = hypot(dx_bt11, dy_bt11);
    }

    {
        acspo::matrix<float> dx_bt12, dy_bt12;
	    std::tie(dx_bt12, dy_bt12) = gradient(bt12);
        mag_grad_bt12 = hypot(dx_bt12, dy_bt12);
    }

    {
        std::tie(dx_sst, dy_sst) = gradient(sst);
        mag_grad_sst = hypot(dx_sst, dy_sst);
    }

    {
        acspo::matrix<float> dxx_sst, dyy_sst, dxy_sst;
        std::tie(dxx_sst, dxy_sst) = gradient(dx_sst);
        std::tie(dxy_sst, dyy_sst) = gradient(dy_sst);
        laplacian_sst = hypot(dxx_sst, dyy_sst);
    }

	logprintf("Laplacian of Gaussian...\n");

    logmag = nanlogfilter(mag_grad_sst, 17, 2, -17);
	logmag.assign(0, logmag < 0);

    {
        acspo::matrix<float> dxx_mag_grad_sst, dyy_mag_grad_sst, dxy_mag_grad_sst;
        {
            acspo::matrix<float> dx_mag_grad_sst, dy_mag_grad_sst;
            std::tie(dx_mag_grad_sst, dy_mag_grad_sst) = gradient(mag_grad_sst);
            std::tie(dxx_mag_grad_sst, dxy_mag_grad_sst) = gradient(dx_mag_grad_sst);
            std::tie(dxy_mag_grad_sst, dyy_mag_grad_sst) = gradient(dy_mag_grad_sst);
        }
        lam2 = localmax(dxx_mag_grad_sst, dyy_mag_grad_sst, dxy_mag_grad_sst);
    }

	border_mask = get_landborders(land_mask, 7);

	apply_land_mask(land_mask, bt03);
	apply_land_mask(land_mask, bt11);
	apply_land_mask(land_mask, bt12);
	apply_land_mask(land_mask, sst);
	
	eigen = compute_eigenvals(bt03, bt11, bt12, border_mask);

    nan_mask = nanmask(sst);
}


void
SPT::run()
{
	auto front_mask = maskfronts(sst, mag_grad_sst, bt08, bt11, mag_grad_bt11, bt12, mag_grad_bt12, eigen, laplacian_sst, lam2, medSST, ice_mask, land_mask, border_mask);

	acspo::matrix<uchar> easyclouds(sst.size());
	easyclouds.assign(1, 0, front_mask < 0);

    acspo::matrix<uchar> ind_ocean(sst.size());
    ind_ocean.assign(1, 0, (nan_mask == 0) && (easyclouds == 0) && (cloud_mask == 0));

    acspo::matrix<uchar> ind_test(sst.size());
    ind_test.assign(1, 0, (nan_mask == 0) && (easyclouds == 0) && (cloud_mask != 0));
	
    {
	    double bt11_low,bt11_high;
	    Point min_loc, max_loc;

        auto bt11_cv = acspo_to_opencv(bt11);
        auto ind_ocean_cv = acspo_to_opencv(ind_ocean);
	    minMaxLoc(bt11_cv, &bt11_low, &bt11_high, &min_loc, &max_loc, ind_ocean_cv);
	    float lows[3] = { (float)bt11_low, -1, 0};
	    float highs[3] = {(float)bt11_high, 5, 5};
	    float steps[3] = {0.2, 0.05, 0.05};

        Mat1f hist_cv;

        generate_cloud_histogram_3d(bt11-bt11_low, bt11-bt12+1, bt11-bt08, lows, highs, steps, ind_ocean, hist_cv);
	    check_cloud_histogram_3d(bt11-bt11_low, bt11-bt12+1, bt11-bt08, lows, highs, steps, ind_test, hist_cv, NN_TEST, front_mask);
    }

	easyclouds.assign(1, 0, front_mask < 0);

    {
        static const int ref_thresh = -8;

        acspo::matrix<int> cluster_labels(sst.size());
        auto cluster_labels_cv = acspo_to_opencv(cluster_labels);

        acspo::matrix<uchar> disagree(sst.size());
        auto disagree_cv = acspo_to_opencv(disagree);

        disagree.assign(255, 0, cloud_mask != 0 && easyclouds == 0);

	    connectedComponentsWithLimit(disagree_cv, 4, MIN_CLUSTER_SIZE, cluster_labels_cv);
	    easyclouds.assign(1, cluster_labels == COMP_SPECKLE);

        auto sst_cv = acspo_to_opencv(sst);
        auto sst_ref_cv = acspo_to_opencv(sst_ref);

        double cluster_low, cluster_high;
	    Point min_loc, max_loc;

	    minMaxLoc(cluster_labels_cv, &cluster_low, &cluster_high, &min_loc, &max_loc);
	    for (unsigned int i = 1; i < cluster_high; ++i) {
		    Scalar tempVal = mean( (sst_cv - sst_ref_cv), (cluster_labels_cv==i) );
		    if(tempVal[0] < ref_thresh){
			    easyclouds.assign(1, cluster_labels == i);
			    cluster_labels.assign(-1, cluster_labels == i);
		    }
	    }
    }

	acspo::matrix<schar> frontsimg(sst.size());
	frontsimg.assign(FRONT_INIT, FRONT_INVALID, front_mask == FRONT_GUESS);

	{
        auto frontsimg_cv = acspo_to_opencv(frontsimg);
        auto dx_sst_cv = acspo_to_opencv(dx_sst);
        auto dy_sst_cv = acspo_to_opencv(dy_sst);
        auto sst_cv = acspo_to_opencv(sst);
        auto mag_grad_sst_cv = acspo_to_opencv(mag_grad_sst);
        auto logmag_cv = acspo_to_opencv(logmag);
        auto easyclouds_cv = acspo_to_opencv(easyclouds);
        auto lam2_cv = acspo_to_opencv(lam2);
	    //TODO: pass logmag to thinfronts
	    connectfronts(frontsimg_cv, dx_sst_cv, dy_sst_cv, sst_cv, mag_grad_sst_cv, logmag_cv, easyclouds_cv, lam2_cv);
    }

    {
	    auto landmask_D = rectdilate(land_mask, 7);
	    frontsimg.assign(FRONT_INVALID, landmask_D != 0);
    }

	acspo::matrix<int> labels;
    {
        Mat1i labels_cv;
        auto frontsimg_cv = acspo_to_opencv(frontsimg);
	    connectedComponentsWithLimit(frontsimg_cv == FRONT_INIT, 8, MIN_FRONT_SIZE, labels_cv);
        labels = opencv_to_acspo(labels_cv);
    }

	labels.assign(FRONT_INVALID, easyclouds != 0);
	frontsimg.assign(FRONT_INIT, FRONT_INVALID, labels > 0);

    acspo::matrix<uchar> new_cloud_mask(sst.size());
    new_cloud_mask.assign(255, 0, cloud_mask != 0 && easyclouds != 0);

    {
        auto new_cloud_mask_cv = acspo_to_opencv(new_cloud_mask);
        auto frontsimg_cv = acspo_to_opencv(frontsimg);
        Mat1b sptmask(new_cloud_mask_cv.size());
        createsptmask(new_cloud_mask_cv, frontsimg_cv, sptmask);
	    writesptmask(ncid, sptmask);
    }
	
    {
        auto sst_diff = sst-sst_ref;
        //auto sea_pixels = (land_mask == 0 && ice_mask == 0 && nan_mask == 0).nnz();

        //auto old_clear = 100.0*(float)(nan_mask == 0 && cloud_mask == 0).nnz()/(float)sea_pixels;
        //auto new_clear = 100.0*(float)(nan_mask == 0 && new_cloud_mask == 0).nnz()/(float)sea_pixels;
        //auto old_mean = mean(sst_diff, cloud_mask == 0 && nan_mask == 0);
        //auto new_mean = mean(sst_diff, new_cloud_mask == 0 && nan_mask == 0);
        logprintf("Old Mask\n");
        logprintf("--------\n");
        //logprintf("clear sky pct = %f\n", old_clear);
        //logprintf("mean(sst-ref) = %f\n", old_mean);
        logprintf("\n");
        logprintf("New Mask\n");
        logprintf("--------\n");
        //logprintf("clear sky pct = %f\n", new_clear);
        //logprintf("mean(sst-ref) = %f\n", new_mean);
    }

}


int
main(int argc, const char **argv)
{
	if(argc < 2){
		eprintf("usage: %s granule\n", argv[0]);
	}
	auto path = argv[1];
	
	logprintf("granule: %s\n", path);
	ACSPOFile f(path, NC_WRITE);
	
	auto spt = SPT(f);

	spt.run();

	f.close();
	logprintf("done\n");
	return 0;
}
