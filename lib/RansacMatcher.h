//
//  RansacMatcher.h
//  RansacMatch
//
//  Created by Tianye Li on 6/17/14.
//  Copyright (c) 2014 Tianye Li. All rights reserved.
//

#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>

using namespace cv;
using namespace std;

//void drawEpilines( const Mat& image1, const vector<KeyPoint> keypoints1,
//                 const Mat& image2, const vector<KeyPoint> keypoints2,
//                 const vector<DMatch> matches, Mat& output);

class RobustMatcher
{
    
private:
    
    // pointer to the feature point detector object
    Ptr<FeatureDetector> detector;
    // pointer to the feature descriptor extractor object
    Ptr<DescriptorExtractor> extractor;
    // max ratio between 1st and 2nd NN
    float ratio;
    // if true, will refine the F matrix
    bool refineF;
    // min distance to epipolar
    double distance;
    // confidence level
    double confidence;
    
public:
    
    RobustMatcher() : ratio(0.65f), refineF(true), confidence(0.99), distance(3.0)
    {
        // SURF is the default feature
        detector = new SurfFeatureDetector();
        extractor = new SurfDescriptorExtractor();
    }
    
    // set the feature detector
    void setFeatureDetector( Ptr<FeatureDetector>& detect )
    {
        detector = detect;
    }
    
    // set the descriptor extractor
    void setDescriptorExtractor( Ptr<DescriptorExtractor>& desc )
    {
        extractor = desc;
    }
    
    // set the min distance to epipolar in RANSAC
    void setMinDistance( double d )
    {
        distance = d;
    }
    
    // set confidence level in RANSAC
    void setConfidenceLevel( double c )
    {
        confidence = c;
    }
    
    // set the NN ratio
    void setRatio( float r )
    {
        ratio = r;
    }
    
    // set if you want to recalculate the F matrix
    void refineFundamental( bool flag )
    {
        refineF = flag;
    }
    
    // clear matches for which NN ratio is greater than threshold
    // return the number of removed points
    // (corresponding entries being cleared, i.e. size will be 0)
    int ratioTest( vector<vector<DMatch>>& matches )
    {
        int removed = 0;
        
        // for all matches
        for ( vector<vector<DMatch>>::iterator matchIterator = matches.begin(); matchIterator != matches.end();
             ++matchIterator )
        {
            
            // if 2 NN has been identified
            if (matchIterator->size() > 1) {
                // check distance ratio
                if ( (*matchIterator)[0].distance / (*matchIterator)[1].distance > ratio )
                {
                    matchIterator->clear(); // remove match
                    removed++;
                }
                
            } else // doesn't have 2 NN
            {
                matchIterator->clear(); // remove match
                removed++;
            }
        } // end for
        
        return removed;
        
    } // end function
    
    
    // insert symmetrical matches in symMatches vector
    void symmetryTest( const vector<vector<DMatch>>& matches1,
                      const vector<vector<DMatch>>& matches2,
                      vector<DMatch>& symMatches )
    {
        // for all matches image 1 -> image 2
        for ( vector<vector<DMatch>>::const_iterator matchIterator1 = matches1.begin();
             matchIterator1 != matches1.end(); ++matchIterator1 )
        {
            if (matchIterator1->size() < 2)
                continue; // ignore deleted matches
            
            // for all matches image 2 -> image 1
            for ( vector<vector<DMatch>>::const_iterator matchIterator2 = matches2.begin();
                 matchIterator2 != matches2.end();
                 ++matchIterator2)
            {
                if (matchIterator2->size() < 2)
                    continue; // ignore deleted matches
                
                // match symmetry test
                if ( ( (*matchIterator1)[0].queryIdx
                      == (*matchIterator2)[0].trainIdx ) &&
                    ( (*matchIterator2)[0].queryIdx
                     == (*matchIterator1)[0].trainIdx ) )
                {
                    // add symmetrical match
                    symMatches.push_back( DMatch( (*matchIterator1)[0].queryIdx,
                                                 (*matchIterator1)[0].trainIdx,
                                                 (*matchIterator1)[0].distance) );
                    break; // go to next match (1->2)
                } // end if
                
            } // end for all matches (2->1)
            
        } // end for all matches (1->2)
    } // end function
    
    
    // identify good matches using RANSAC
    // return fundamental matrix
    Mat ransacTest( const vector<DMatch>& matches,
                   const vector<KeyPoint>& keypoints1,
                   const vector<KeyPoint>& keypoints2,
                   vector<DMatch>& outMatches)
    {
        // convert keypoints into Point2f
        vector<Point2f> points1, points2;
        for (vector<DMatch>::const_iterator it = matches.begin();
             it != matches.end(); ++it)
        {
            float x = keypoints1[it->queryIdx].pt.x;
            float y = keypoints1[it->queryIdx].pt.y;
            points1.push_back( Point2f(x,y) );
            
            x = keypoints2[it->trainIdx].pt.x;
            y = keypoints2[it->trainIdx].pt.y;
            points2.push_back( Point2f(x,y) );
        }
        
        // compute F matrix using RANSAC
        vector<uchar> inliers( points1.size(), 0 );
        Mat fundamental = findFundamentalMat( Mat(points1), Mat(points2),
                                             inliers,          // if inlier
                                             CV_FM_RANSAC,     // method
                                             distance,         // distance to epipolar line
                                             confidence );     // confidence
        
        // extract survivors (inliers)
        vector<uchar>::const_iterator itIn = inliers.begin();
        vector<DMatch>::const_iterator itM = matches.begin();
        // for all matches
        for ( ; itIn != inliers.end(); ++itIn, ++itM) {
            if (*itIn) { // if valid match
                outMatches.push_back(*itM);
            }
        }
        
        cout << "symMatches size (RANSAC) = " << outMatches.size() << endl;
        
        if (refineF)
        {
            // the F matrix will be recomputed with all accepted matches
            
            points1.clear();
            points2.clear();
            for ( vector<DMatch>::const_iterator it = outMatches.begin();
                 it != outMatches.end(); ++it)
            {
                float x = keypoints1[it->queryIdx].pt.x;
                float y = keypoints1[it->queryIdx].pt.y;
                points1.push_back( Point2f(x,y) );
                x = keypoints2[it->trainIdx].pt.x;
                y = keypoints2[it->trainIdx].pt.y;
                points2.push_back( Point2f(x,y) );
            }
            
            // compute the fundamental matrix using all accepted points
            fundamental = findFundamentalMat( Mat(points1), Mat(points2), CV_FM_8POINT );
            
        } // end if refineF = true
        
        return fundamental;
        
    } // end function
    
    
    // match feature points using symmetry test and RANSAC
    // return fundamental matrix
    Mat match( Mat& image1, Mat& image2,
              vector<DMatch>& matches,
              vector<KeyPoint>& keypoints1, vector<KeyPoint>& keypoints2)
    {
        // 1. detect SURF features (and descriptors)
        detector->detect( image1, keypoints1 );
        detector->detect( image2, keypoints2 );
        
        Mat descriptor1, descriptor2;
        extractor->compute( image1, keypoints1, descriptor1 );
        extractor->compute( image2, keypoints2, descriptor2 );

        cout << keypoints1.size() << " keypoints found in image 1." << endl;
        cout << keypoints2.size() << " keypoints found in image 2." << endl;
        
        cout << "Descriptor size = " << descriptor1.rows << " by " << descriptor1.cols << endl;
        
        // 2. match the two image descriptors
        
        BFMatcher matcher(NORM_L2);
        vector<vector<DMatch>> matches1, matches2;
        
        // match 1->2 using kNN (k=2)
        matcher.knnMatch( descriptor1, descriptor2, matches1, 2 );
        // match 2->1 using kNN (k=2)
        matcher.knnMatch( descriptor2, descriptor1, matches2, 2 );
        
        cout << "match 1->2 size = " << matches1.size() << endl;
        cout << "match 2->1 size = " << matches2.size() << endl;
        
        // 3. remove matches for which NN ratio is greater than threshold
        
        // clean 1->2 matches
        int removed = ratioTest(matches1);
        cout << "match 1->2 size (ratio test) = " << matches1.size()-removed << endl;
        
        // clean 2->1 matches
        removed = ratioTest(matches2);
        cout << "match 2->1 size (ratio test) = " << matches2.size()-removed << endl;
        
        
        // 4. remove non-symmetrical matches
        vector<DMatch> symMatches;
        symmetryTest(matches1, matches2, symMatches);
        cout << "symMatches size (symmetry test) = " << symMatches.size() << endl;
        
        // 5. validate matches using RANSAC
        Mat fundamental = ransacTest(symMatches,
                                     keypoints1, keypoints2, matches);
        
        // return the final result
        return fundamental;
        
    } // end function
    
    
}; // end class defination



