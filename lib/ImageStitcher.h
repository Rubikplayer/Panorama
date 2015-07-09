#include <iostream>
#include <string>
#include <cstring>
#include <vector>
#include <opencv2/opencv.hpp>
#include "RansacMatcher.h"

class ImageStitcher
{
public:
	ImageStitcher( std::string &path, int frame_start, int frame_end, int resize_scale, int distance );
	~ImageStitcher();

 	// step 1: read and resize images
	bool readOneImage( cv::Mat &img, int frame );
	bool readImages( std::vector<cv::Mat> &images );
	bool resizeImages( std::vector<cv::Mat> &images );

	// step 2: project images on cylinder
	bool getCylinderMaps( cv::Mat &xmap, cv::Mat &ymap, cv::Size src_size, float f );
	bool projectImagesToCylinder( std::vector<cv::Mat> &images, cv::Mat &mask );
	
	// step 3: estimate homographies
	bool estimateHomographies( std::vector<cv::Mat> &images, std::vector<cv::Mat> &homographies );
	bool matchTwoImages( cv::Mat& image1, cv::Mat& image2, RobustMatcher rmatcher,
 						cv::Mat& fundamental, cv::Mat& homography );

	// step 4: stitch images
	bool stitchImages( std::vector<cv::Mat> &images, cv::Mat &mask, std::vector<cv::Mat> &homographies,
 						cv::Mat &result );
	cv::Size estimateCanvasSize( std::vector<cv::Mat> &homographiesToCenter, cv::Size img_size, int extra = 10 );

	bool warpImages( const std::vector<cv::Mat> &images, const cv::Mat &mask,
					 const std::vector<cv::Mat> &homographiesToCenter, cv::Size result_size,
					 std::vector<cv::Mat> &images_warped, std::vector<cv::Mat> &masks_warped );	

	bool mergeImages( const std::vector<cv::Mat> &images_warped, 
       					const std::vector<cv::Mat> &masks_warped,
       					cv::Mat &result );	

	bool getGaussianMask( cv::Mat& mask, cv::Size mask_size, float sigma );
	// bool processHomographies( std::vector<cv::Mat> &homographies, cv::Size canvas_size );

	// main process that combine all steps above
	cv::Mat process();

	// utility functions
	void display();
	void printMat( cv::Mat &image, std::string filename );

private:
	std::string image_path;
	std::string path_pattern;
	int frame_start, frame_end;
	int frame_num;
	int resize_scale;   
	int distance; 

};