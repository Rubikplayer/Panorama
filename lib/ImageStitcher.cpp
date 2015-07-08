#include <iostream>
#include <fstream>
#include <cstdio>
#include <string>
#include <cstring>
#include <vector>
#include <stdexcept>
#include <cmath>
#include <opencv2/opencv.hpp>
#include "ImageStitcher.h"

// ----------------------------------------------------- //
//                 Class: ImageStitcher
// ----------------------------------------------------- //

ImageStitcher::ImageStitcher( std::string &path, int start, int end, int r, int d )
{
	image_path = path;
	frame_start = start;
	frame_end = end; 
	resize_scale = r;
	distance = d; 
	frame_num = frame_end - frame_start + 1;

	// check exceptions
	// end < start?

	// parse image path
	size_t first_sharp = image_path.find_first_of( "#" );
	size_t last_sharp = image_path.find_last_of( "#" );
	if ( first_sharp == std::string::npos || last_sharp == std::string::npos )
	{
		std::cerr << "ImageStitcher::ImageStitcher(): expect input path to contain '#'" << std::endl;
	}

	size_t sharp_width = last_sharp - first_sharp + 1;
	std::cout << first_sharp << " " << last_sharp << std::endl;
	path_pattern = image_path.substr( 0, first_sharp ) + "%0" + std::to_string( sharp_width )
	               + "d" + image_path.substr( last_sharp+1, std::string::npos );

}


// ----------------------------------------------------- //


ImageStitcher::~ImageStitcher()
{	
}


// ----------------------------------------------------- //


bool
ImageStitcher::readOneImage( cv::Mat &img, int frame )
{
	if ( frame < frame_start || frame > frame_end )
	{
		throw std::invalid_argument( "ImageStitcher::readImage(): frame out of range" );
	}

	char filename[80];
	sprintf( filename, path_pattern.c_str(), frame );
	img = cv::imread( filename );

	if ( !img.data )
	{
		std::string message = std::string( "ImageStitcher::readImage(): fail to read image: " ) + filename;
		throw std::logic_error( message.c_str() );		
	} 
	
	std::cout << "loaded: " << filename << std::endl;
	return true;
}


// ----------------------------------------------------- //


bool
ImageStitcher::readImages( std::vector<cv::Mat> &images )
{
	// expect images to be empty
	// images.resize( frame_num );

	for ( int i = 0; i < frame_num; ++i )
	{
		int cur_frame = frame_start + i;
		cv::Mat cur_image;
		readOneImage( cur_image, cur_frame );
		images.push_back( cur_image );

		// cv::namedWindow( "input", cv::WINDOW_AUTOSIZE );
		// cv::imshow( "input", cur_image );
		// cv::waitKey( 0 );
	}
	return true;
}


// ----------------------------------------------------- //


bool 
ImageStitcher::resizeImages( std::vector<cv::Mat> &images )
{
	for ( int i = 0; i < frame_num; ++i )
	{
		// std::cout << "frame " << i;
		// std::cout << " " << images[i].cols << "-" << images[i].rows;

		cv::resize( images[i], images[i],
		 			cv::Size( images[i].cols / resize_scale, images[i].rows / resize_scale ) );
		
		// std::cout << " to " << images[i].cols << "-" << images[i].rows << std::endl;
		// cv::namedWindow( "input", cv::WINDOW_AUTOSIZE );
		// cv::imshow( "input", images[i] );
		// cv::waitKey( 20 );
	}

	return true;
}


// ----------------------------------------------------- //


bool
ImageStitcher::projectImagesToCylinder( std::vector<cv::Mat> &images )
{
    for ( int i = 0; i < images.size(); ++i )
    {
        cv::Mat xmap, ymap;
        // generate warp maps
        getCylinderMaps( xmap, ymap, images[i].size(), static_cast<float>( distance ) );
        
        std::cout << i << " r" << images[i].rows << "-c" << images[i].cols << std::endl;
        // warp the images
        // out = Mat::zeros( dst_size.height, dst_size.width, images_resized[0].type());
        cv::remap( images[i], images[i], xmap, ymap,
              	   cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0,0,0) );

        std::cout << i << " r" << images[i].rows << "-c" << images[i].cols << std::endl;

        cv::namedWindow( "image", cv::WINDOW_AUTOSIZE );
        cv::imshow( "image", images[i] ); 
        cv::waitKey( 0 );

        printMat( xmap, std::string( "/tmp/xmap.txt" ) );
        printMat( ymap, std::string( "/tmp/ymap.txt" ) );

    }
    return true;
}


// ----------------------------------------------------- //


bool
ImageStitcher::getCylinderMaps( cv::Mat &xmap, cv::Mat &ymap, cv::Size src_size, float f )
{
	float cx = static_cast<float>( src_size.height ) / 2;
    float cy = static_cast<float>( src_size.width ) / 2;
    float s  = 1; // scaling factor, usually set equal to f (Szeliski (2010) P439)
    
    float miniXToCenterSq = ( static_cast<float>( static_cast<int>(cx) ) - cx )
 					        * ( static_cast<float>( static_cast<int>(cx) ) - cx );
    float xmin = s * atan( (-cx) / f ) * f;
    float xmax = s * atan( ( cx) / f ) * f;
    float ymin = s * (-cy) / sqrt( f * f + 0 ) * f;
    float ymax = s * ( cy) / sqrt( f * f + 0 ) * f;
    
    int new_cols = static_cast<int>( ceil( xmax - xmin ) );
    int new_rows = static_cast<int>( ceil( ymax - ymin ) );

    std::cout << "new size in cylinder map: r"<< new_rows << "-c" << new_cols << std::endl;


    // int new_rows = static_cast<int>(ymax) - static_cast<int>(ymin);
    // int new_cols = static_cast<int>(xmax+0.5) - static_cast<int>(xmin+0.5);
    //    int x_offset = static_cast<int>(xmin/2);
    
    xmap.create( new_rows, new_cols, CV_32FC1 );
    ymap.create( new_rows, new_cols, CV_32FC1 );
    
    float cx2 = static_cast<float>(new_cols) / 2;
    float cy2 = static_cast<float>(new_rows) / 2;
    
    for ( int r = 0; r < new_rows; r++ )
    {
        for ( int c = 0; c < new_cols; c++ )
        {
            // coordinate in original image
            xmap.at<float>(r,c) = f * tan( (c-cx2) / s ) + cx;
            ymap.at<float>(r,c) = (r-cy2) / s / cos( (c-cx2)/s ) + cy;
            // ymap.at<float>(r,c) = f * (r-cy2) / s / cos( (c-cx2)/s ) + cy;
            
        } // end for col
    } // end for row
    
    return true;

}



// ----------------------------------------------------- //


void
ImageStitcher::display()
{
	std::cout << "input path:    " << image_path << std::endl;
	std::cout << "frame range:   " << frame_start << "-" << frame_end << std::endl;
	std::cout << "resize scale:  " << resize_scale << std::endl;
	std::cout << "distance:      " << distance << std::endl;
	std::cout << "path patttern: " << path_pattern << std::endl;
}


// ----------------------------------------------------- //

void 
ImageStitcher::printMat( cv::Mat &image, std::string filename )
{
	std::ofstream of( filename, std::ofstream::out );
	of << "image with size: r" << image.rows << "-c" << image.cols << std::endl;

	for ( int r = 0; r < image.rows; r++ )
    {
        for ( int c = 0; c < image.cols; c++ )
        {
            of << image.at<float>(r,c) << " ";
        }
        of << std::endl;
    }

	of.close();

}


