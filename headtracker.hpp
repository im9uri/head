//
//  headtracker.h
//  OpenCV Lab 4
//
//  Created by HyunJun Kim on 2015. 6. 14..
//  Copyright (c) 2015년 HyunJun Kim. All rights reserved.
//

#ifndef __HEAD_TRACKER__
#define __HEAD_TRACKER__

#include "opencv2/objdetect.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"

#include <iostream>

using namespace cv;
using namespace std;

class HeadTracker {
    
private:
    
    

    float hranges[2];
    const float* phranges = hranges;
    
    bool headDetected = false;
    bool backprojMode = false;
    
    Rect target;
    
    Mat frame;
    Mat hsv, hue, mask, histogram, histogramImage;
    
    void drawHistogram();
    void updateFrame();
    
public:
    VideoCapture capture;
    int hsize;
    int vmin, vmax;
    int smin;
    Mat image;
    Mat backproj;
    RotatedRect trace;
    
    HeadTracker();
    HeadTracker(int cameraNumber);
    bool detectHead(CascadeClassifier& cascade);
    bool track();

};

#endif
