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
#include "RansacMatcher.h"


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
ImageStitcher::projectImagesToCylinder( std::vector<cv::Mat> &images, cv::Mat &mask )
{
	// same map for all images since they all have same size
	cv::Mat xmap, ymap;
    getCylinderMaps( xmap, ymap, images[0].size(), static_cast<float>( distance ) );

    cv::Mat orig_mask;
    float sigma = static_cast<float>( images[0].cols / 2 );
    getGaussianMask( orig_mask, images[0].size(), 50 );
    cv::remap( orig_mask, mask, xmap, ymap, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0,0,0) );

    for ( int i = 0; i < images.size(); ++i )
    {
        // std::cout << i << " r" << images[i].rows << "-c" << images[i].cols << std::endl;
        cv::remap( images[i], images[i], xmap, ymap,
              	   cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0,0,0) );

        // std::cout << i << " r" << images[i].rows << "-c" << images[i].cols << std::endl;

        // cv::namedWindow( "image", cv::WINDOW_AUTOSIZE );
        // cv::imshow( "image", images[i] ); 
        // cv::waitKey( 0 );

        // printMat( xmap, std::string( "/tmp/xmap.txt" ) );
        // printMat( ymap, std::string( "/tmp/ymap.txt" ) );

    }
    return true;
}


// ----------------------------------------------------- //


bool
ImageStitcher::getCylinderMaps( cv::Mat &xmap, cv::Mat &ymap, cv::Size src_size, float f )
{
	float cx = static_cast<float>( src_size.width ) / 2;
    float cy = static_cast<float>( src_size.height) / 2;
    float s  = f; // scaling factor, usually set equal to f (Szeliski (2010) P439)
    
    
    float miniXToCenterSq = (static_cast<float>(static_cast<int>(cx)) - cx )
    						* (static_cast<float>(static_cast<int>(cx)) - cx );
    float xmin = s * atan( (-cx) / f ) + cx;
    float xmax = s * atan( ( cx) / f ) + cx;
    float ymin = s * (-cy) / sqrt( f * f + miniXToCenterSq ) + cy;
    float ymax = s * ( cy) / sqrt( f * f + miniXToCenterSq ) + cy;
   
    
    int new_rows = static_cast<int>(ymax) - static_cast<int>(ymin);
    int new_cols = static_cast<int>(xmax+0.5) - static_cast<int>(xmin+0.5);
    //    int x_offset = static_cast<int>(xmin/2);
    
    xmap.create( new_rows, new_cols, CV_32FC1 );
    ymap.create( new_rows, new_cols, CV_32FC1 );
    
    float cx2 = static_cast<float>(new_cols) / 2;
    float cy2 = static_cast<float>(new_rows) / 2;
    
    for ( int r=0; r<new_rows; r++ )
    {
        for ( int c=0; c<new_cols; c++ )
        {
            // coordinate in original image
            xmap.at<float>(r,c) = f * tan( (c-cx2) / s ) + cx;
            ymap.at<float>(r,c) = f * (r-cy2) / s / cos( (c-cx2)/s ) + cy;

        } // end for col
    } // end for row

    
    return true;

}


// ----------------------------------------------------- //

bool 
ImageStitcher::estimateHomographies( std::vector<cv::Mat> &images, std::vector<cv::Mat> &homographies )
{
	RobustMatcher rmatcher;
	rmatcher.setConfidenceLevel( 0.98 );
    rmatcher.setMinDistance( 1.0 );
    rmatcher.setRatio( 0.65f );
    cv::Ptr<cv::FeatureDetector> pfd = new cv::SurfFeatureDetector( 10 );
    rmatcher.setFeatureDetector( pfd );

    // std::vector<cv::Mat> homographies;

    for ( int i = 0; i < images.size()-1; i++ )
    {
        std::cout << std::endl << "matching image " << i << " and " << i+1 << std::endl;
        cv::Mat F, H;
        matchTwoImages( images[i], images[i+1], rmatcher, F, H );
        homographies.push_back( H );
    }

    // for ( int i = 0; i < images.size(); i++ )
    // {
    //     std::cout << std::endl << "Preparing H[" << i << "]" << std::endl;
    //     cv::Mat H = Mat::eye(3, 3, orig_homographies[0].type());
    //     if ( i < center ) 
    //     {
    //         for ( int j = i; j < center; j++ ) 
    //         {
    //             H = orig_homographies[j] * H;
    //             std::cout << "   multiplying H[" << j << "]" << std::endl;
    //         }
    //     } 
    //     else if ( i > center ) 
    //     {
    //         for ( int j = i; j > center; j-- ) {
    //             H = orig_homographies[j-1].inv() * H;
    //             std::cout << "   multiplying inv H[" << j-1 << "]" << std::endl;
    //         }
    //     } 
    //     else 
    //     {
    //         // do nothing: H = identity
    //     }
        
    //     // // translate to center ( PUT IN MERGING PART )
    //     // cv::Mat translate = cv::Mat::eye( 3, 3, homographies[0].type() );
    //     // translate.at<double>( 0,2 ) = static_cast<double>( ( result_size.width - images[0].cols ) / 2 );
    //     // translate.at<double>( 1,2 ) = static_cast<double>( ( result_size.height - images[0].rows ) / 2 );
        
    //     homographies.push_back( H ); // translate * H
    // }

    // std::cout << "H[0]: " << homographies[0] << std::endl;
    // cv::Mat I0_warped;
    // cv::warpPerspective( images[0], I0_warped, homographies[0], images[0].size(),
    //                     cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0,0,0));

    // cv::namedWindow( "image 0", cv::WINDOW_AUTOSIZE );
    // cv::imshow( "image 0", images[0] );

    // cv::namedWindow( "image 1", cv::WINDOW_AUTOSIZE );
    // cv::imshow( "image 1", images[1] );
    // // cv::waitKey( 0 );

    // cv::namedWindow( "image 0 warped", cv::WINDOW_AUTOSIZE );
    // cv::imshow( "image 0 warped", I0_warped );
    // cv::waitKey( 0 );

	return true;
}


// ----------------------------------------------------- //


bool 
ImageStitcher::matchTwoImages( cv::Mat& image1, cv::Mat& image2, RobustMatcher rmatcher,
 					cv::Mat& fundamental, cv::Mat& H )
{
	// match the two images
    std::vector<cv::DMatch> matches;
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    fundamental = rmatcher.match( image1, image2, matches, keypoints1, keypoints2 );
    
//    // draw matches
//    Mat matched_image;
//    drawMatches(image1, keypoints1, image2, keypoints2, matches, matched_image, Scalar(255,255,255));
//    
//    namedWindow("matched image");
//    imshow("matched image", matched_image);
//    waitKey();
    
//    Mat image_to_show1, image_to_show2;
//    image1.copyTo(image_to_show1);
//    image2.copyTo(image_to_show2);
    
    // convert keypoints into Point2f
    std::vector<cv::Point2f> points1, points2;
    for ( std::vector<cv::DMatch>::const_iterator it = matches.begin();
         it != matches.end(); ++it )
    {
        float x = keypoints1[it->queryIdx].pt.x;
        float y = keypoints1[it->queryIdx].pt.y;
        points1.push_back( cv::Point2f(x,y) );
        // circle(image_to_show1, Point2f(x,y), 3, Scalar(255,255,255), 2);
        
        x = keypoints2[it->trainIdx].pt.x;
        y = keypoints2[it->trainIdx].pt.y;
        points2.push_back( cv::Point2f(x,y) );
        // circle(image_to_show2, Point2f(x,y), 3, Scalar(255,255,255), 2);
    }
    
    std::cout << "selected feature points: " << points1.size() << " and " << points2.size() << std::endl;
    
    // find homographies
    std::vector<uchar> inliers( points1.size(), 0 );
    H = cv::findHomography( cv::Mat(points1), cv::Mat(points2), inliers, CV_RANSAC, 1);
    
    // cout << "homographies matrix = " << H << endl;
	return true;
}


// ----------------------------------------------------- //


bool
ImageStitcher::stitchImages( std::vector<cv::Mat> &images, cv::Mat &mask, std::vector<cv::Mat> &homographies,
 							 cv::Mat &result )
{

	// process homographies based on pivot image
	std::vector<cv::Mat> homographiesToCenter;

    int center = static_cast<int>(images.size()) / 2;
	for ( int i = 0; i < images.size(); i++ )
    {
        std::cout << std::endl << "Preparing H[" << i << "]" << std::endl;
        cv::Mat H = Mat::eye(3, 3, homographies[0].type());
        if ( i < center ) 
        {
            for ( int j = i; j < center; j++ ) 
            {
                H = homographies[j] * H;
                std::cout << "   multiplying H[" << j << "]" << std::endl;
            }
        } 
        else if ( i > center ) 
        {
            for ( int j = i; j > center; j-- ) {
                H = homographies[j-1].inv() * H;
                std::cout << "   multiplying inv H[" << j-1 << "]" << std::endl;
            }
        } 
        else 
        {
            // do nothing: H = identity
        }
        
        homographiesToCenter.push_back( H ); // translate * H
    }

    // estimate canvas size
    cv::Size result_size = estimateCanvasSize( homographiesToCenter, images[0].size(), 0 );
    std::cout << "result size (r,c): " << result_size.height << " - " << result_size.width << std::endl;

    // translate to center ( PUT IN MERGING PART )
    cv::Mat translate = cv::Mat::eye( 3, 3, homographiesToCenter[0].type() );
    translate.at<double>( 0,2 ) = static_cast<double>( ( result_size.width - images[0].cols ) / 2 );
    translate.at<double>( 1,2 ) = static_cast<double>( ( result_size.height - images[0].rows ) / 2 );

    for ( int i = 0; i < images.size(); i++ )
    {
    	homographiesToCenter[i] = translate * homographiesToCenter[i];
    }

    std::vector<cv::Mat> masks_warped;
    std::vector<cv::Mat> images_warped;

    warpImages( images, mask, homographiesToCenter, result_size, images_warped, masks_warped );

    mergeImages( images_warped, masks_warped, result );

    return true;
}


// ----------------------------------------------------- //


cv::Size
ImageStitcher::estimateCanvasSize( std::vector<cv::Mat> &homographiesToCenter, cv::Size img_size, int extra )
{
	cv::Mat img_corners = ( cv::Mat_<double>(3,4) << 0,                 0, img_size.width-1,  img_size.width-1,
    	        									 0, img_size.height-1,                0, img_size.height-1,
    	        									 1,                 1,                1,                 1 );

	cv::Mat edge1 = homographiesToCenter[0] * img_corners;
	cv::Mat edge2 = homographiesToCenter[frame_num-1] * img_corners;

    cv::Mat x1( edge1, cv::Rect( 0,0,4,1 ) );
    cv::Mat y1( edge1, cv::Rect( 0,1,4,1 ) );

    cv::Mat x2( edge2, cv::Rect( 0,0,4,1 ) );
    cv::Mat y2( edge2, cv::Rect( 0,1,4,1 ) );

    // std::cout << "x: " << x1 << " & " << x2 << std::endl;
    // std::cout << "y: " << y1 << " & " << y2 << std::endl << std::endl;

    double min1, max1, min2, max2;

    cv::minMaxLoc( x1, &min1, &max1 );
    cv::minMaxLoc( x2, &min2, &max2 );
    double x_min = cv::min( min1, min2 );
    double x_max = cv::max( max1, max2 );

    cv::minMaxLoc( y1, &min1, &max1 );
    cv::minMaxLoc( y2, &min2, &max2 );
    double y_min = cv::min( min1, min2 );
    double y_max = cv::max( max1, max2 );

    // std::cout << "[xmin, xmax, ymin, ymax]: " << x_min << " " << x_max << " " << y_min << " " << y_max << std::endl;

    cv::Size result_size;
    result_size.height = static_cast<int>( y_max - y_min + 2 * extra );
    result_size.width = static_cast<int>( x_max - x_min + 2 * extra );

    // std::cout << "result size (r,c): " << result_size.height << " - " << result_size.width << std::endl;

	return result_size;
}


// ----------------------------------------------------- //
bool 
ImageStitcher::warpImages( const std::vector<cv::Mat> &images, const cv::Mat &mask,
						 const std::vector<cv::Mat> &homographiesToCenter, cv::Size result_size,
						 std::vector<cv::Mat> &images_warped, std::vector<cv::Mat> &masks_warped )
{
	for ( int i = 0; i < images.size(); i++ )
	{
        cv::Mat cur_image, cur_mask;

        cv::warpPerspective( images[i], cur_image, homographiesToCenter[i], result_size,
                      		 cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0,0,0) );
        cv::warpPerspective( mask, cur_mask, homographiesToCenter[i], result_size,
                         	 cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0,0,0) );

        images_warped.push_back( cur_image );
        masks_warped.push_back(cur_mask);
        
		// cv::namedWindow( "warped", cv::WINDOW_AUTOSIZE );
		// cv::imshow( "warped", cur_image );

		// cv::namedWindow( "warped mask", cv::WINDOW_AUTOSIZE );
		// cv::imshow( "warped mask", cur_mask );
		// cv::waitKey( 0 );

        
    }
	return true;
}

bool
ImageStitcher::mergeImages( const std::vector<cv::Mat> &images_warped, 
							const std::vector<cv::Mat> &masks_warped,
							cv::Mat &result )
{
	// merge images
    result.create( images_warped[0].rows, images_warped[0].cols, images_warped[0].type() );
    for ( int c = 0; c < result.cols; c++ )
    {
        for ( int r = 0; r < result.rows; r++ )
        {
            // blend images using the isInImage markers
            double sum = 0;
            for ( int i = 0; i < images_warped.size(); i++ )
                sum += masks_warped[i].at<double>(r,c);
                
                
            if ( images_warped[0].channels() == 1)
            {  // if one channel in uchar
                if ( sum == 0 )
                    result.at<uchar>(r,c) = 0;
                else
                {
                    double gray=0;
                    for ( int i=0; i<images_warped.size(); i++ )
                    {
                        double weight = masks_warped[i].at<double>(r,c) / sum;
                        gray += static_cast<double>(images_warped[i].at<uchar>(r,c)) * weight;
                    }
                    result.at<uchar>(r,c) = static_cast<uchar>( gray );

                }
            }
            else if ( images_warped[0].channels() == 3 )
            {   // if three channels in Vec3b (uchar 3components)
                if ( sum == 0 )
                {
                    result.at<cv::Vec3b>(r,c)[0] = 0;
                    result.at<cv::Vec3b>(r,c)[1] = 0;
                    result.at<cv::Vec3b>(r,c)[2] = 0;
                }
                else
                {
                    double blue=0, green=0, red=0;
                    for ( int i=0; i<images_warped.size(); i++ )
                    {
                        double weight = masks_warped[i].at<double>(r,c) / sum;
                        // BGR order
                        blue  += static_cast<double>(images_warped[i].at<cv::Vec3b>(r,c)[0]) * weight;
                        green += static_cast<double>(images_warped[i].at<cv::Vec3b>(r,c)[1]) * weight;
                        red   += static_cast<double>(images_warped[i].at<cv::Vec3b>(r,c)[2]) * weight;
                    }
                    result.at<cv::Vec3b>(r,c)[0] = static_cast<uchar>( blue );
                    result.at<cv::Vec3b>(r,c)[1] = static_cast<uchar>( green );
                    result.at<cv::Vec3b>(r,c)[2] = static_cast<uchar>( red );

                }
            } // end if channels
            
        }
    } // end for all pixels 

    return 1;

	return true;
}


// ----------------------------------------------------- //


// bool
// ImageStitcher::processHomographies( std::vector<cv::Mat> &homographies, cv::Size canvas_size )
// {
// 	return true;
// }

bool
ImageStitcher::getGaussianMask( cv::Mat& mask, cv::Size mask_size, float sigma )
{
    // assume mask is 1-channel image
    if ( mask.channels() != 1 ) {
        cout << "Error in getGaussianMask(): mask should be 1-channel." << endl;
        return false;
    }
    // create as double type
    mask.create( mask_size.height, mask_size.width, CV_64F );
    double cx = static_cast<double>( mask_size.width ) / 2;
    double cy = static_cast<double>( mask_size.height ) / 2;

    for ( int c=0; c<mask.cols; c++ ) {
        for ( int r=0; r<mask.rows; r++ ) {
            mask.at<double>( r,c ) = exp( -((c-cx)*(c-cx)+(r-cy)*(r-cy)) / 2 / sigma / sigma );
        }
    }
    
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


