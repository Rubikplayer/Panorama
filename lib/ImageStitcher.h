#include <iostream>
#include <string>
#include <cstring>
#include <vector>
#include <opencv2/opencv.hpp>

class ImageStitcher
{
public:
	ImageStitcher( std::string &path, int frame_start, int frame_end, int resize_scale, int distance );
	~ImageStitcher();

	bool readOneImage( cv::Mat &img, int frame );
	bool readImages( std::vector<cv::Mat> &images );
	bool resizeImages( std::vector<cv::Mat> &images );
	bool projectImagesToCylinder( std::vector<cv::Mat> &images );


	// bool matchImages( cv::Mat& image1, cv::Mat& image2, RobustMatcher rmatcher,
 //                cv::Mat& fundamental, cv::Mat& homography );
	int mergeImages( std::vector<cv::Mat> homographyToCenter,
					 std::vector<cv::Mat> warpedImages,
					 std::vector<cv::Mat> warpedMasks, 
					 cv::Mat& result );

	cv::Mat process();

	// utility functions
	bool getCylinderMaps( cv::Mat &xmap, cv::Mat &ymap, cv::Size src_size, float f );

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