#include <iostream>
#include <string>
#include <cstring>
#include <vector>
#include <opencv2/opencv.hpp>
#include "ImageStitcher.h"
#include "RansacMatcher.h"

int safemain( int argc, char *argv[] )
{
	
	// parse arguments
	std::string inPath;       // -i <arg> 
	std::string outPath;      // -o <arg>
	int frameStart, frameEnd; // -f <arg1> <arg2>
	int resizeScale;          // -r <arg>
	int distance;             // -d <arg>

	std::map<std::string, int> checklist;
	checklist["-i"] = 1;
	checklist["-o"] = 1;
	checklist["-f"] = 2;
	checklist["-r"] = 1;
	checklist["-d"] = 1;

	for ( size_t i = 1; i < argc; ++i )
	{
		if ( checklist.find( argv[i] ) != checklist.end() )
		{
			if ( strcmp( argv[i], "-i" ) == 0 )
			{
				inPath = std::string( argv[++i] );
			}

			if ( strcmp( argv[i], "-o" ) == 0 )
			{
				outPath = std::string( argv[++i] );
			}

			if ( strcmp( argv[i], "-f" ) == 0 )
			{
				frameStart = atoi( argv[++i] );
				frameEnd = atoi( argv[++i] );
			}
			
			if ( strcmp( argv[i], "-r" ) == 0 )
			{
				resizeScale = atoi( argv[++i] );
			}

			if ( strcmp( argv[i], "-d" ) == 0 )
			{
				distance = atoi( argv[++i] );
			}
		}
	}

	ImageStitcher stitcher( inPath, frameStart, frameEnd, resizeScale, distance );
	stitcher.display();

	cv::Mat result = stitcher.process();

	// write the results
	cv::namedWindow( "Panorama", cv::WINDOW_AUTOSIZE );
	cv::imshow( "Panorama", result );
	cv::waitKey( 0 );

	cv::imwrite( outPath, result );
	std::cout << "output saved to: " << outPath << std::endl;


	return 0;
}

int main( int argc, char *argv[] )
{	
	try
	{
		return safemain( argc, argv );
	}
	catch( std::exception &e )
	{
		std::cerr << "ERROR: " << e.what() << std::endl;
	}
	return -1;

}


