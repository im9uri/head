//
//  headtracker.cpp
//  OpenCV Lab 4
//
//  Created by HyunJun Kim on 2015. 6. 14..
//  Copyright (c) 2015년 HyunJun Kim. All rights reserved.
//

#include "headtracker.hpp"

HeadTracker::HeadTracker() {
    HeadTracker(0);
}

HeadTracker::HeadTracker(int cameraNumber) {

    capture.open(cameraNumber);
    
    if (!capture.isOpened()) {
        cout << "카메라를 여는 데 실패하였습니다." << endl;
        exit(-1);
    }
    
    capture >> frame;
    frame.copyTo(image);
    
    vmin = 10;
    vmax = 256;
    smin = 30;
    
    hsize = 16;
    hranges[0] = 0;
    hranges[1] = 180;
    
}

void HeadTracker::drawHistogram() {
    
    hsize = 16;
    
    histogramImage = Scalar::all(0);
    int binW = histogramImage.cols / hsize;
    Mat buf(1, hsize, CV_8UC3);
    for( int i = 0; i < hsize; i++ )
        buf.at<Vec3b>(i) = Vec3b(saturate_cast<uchar>(i*180./hsize), 255, 255);
    cvtColor(buf, buf, COLOR_HSV2BGR);
    
    for( int i = 0; i < hsize; i++ ) {
        int val = saturate_cast<int>(histogram.at<float>(i)*histogramImage.rows/255);
        rectangle( histogramImage, Point(i*binW,histogramImage.rows),
                  Point((i+1)*binW,histogramImage.rows - val),
                  Scalar(buf.at<Vec3b>(i)), -1, 8 );
    }
}

void HeadTracker::updateFrame() {
    
    if(!capture.isOpened()){
        capture.open(0);
    }
    
    capture >> frame;
    if (frame.empty())
        return;
    frame.copyTo(image);
    
    // 프레임에서 HSV 이미지를 취득한다.
    cvtColor(image, hsv, COLOR_BGR2HSV);
    
    // 이미지를 보정한다.
    int _vmin = vmin, _vmax = vmax;
    inRange(hsv, Scalar(0, smin, MIN(_vmin,_vmax)), Scalar(180, 256, MAX(_vmin, _vmax)), mask);
    
    // HSV 이미지에서 Hue 채널을 분리한다.
    const int channelMask[] = {0, 0};
    hue.create(hsv.size(), hsv.depth());
    mixChannels(&hsv, 1, &hue, 1, channelMask, 1);
}

bool HeadTracker::detectHead(CascadeClassifier& cascade) {
    
    headDetected = false;
    
    vector<Rect> heads;
    
    Rect biggestRect;
    size_t biggestindex = 0;
    
    updateFrame();
    
   
    cascade.detectMultiScale(image, heads, 1.1, 2, 0, Size(30, 30));
    
    // 얼굴이 발견되지 않으면 리턴한다.
    if (heads.size() < 1)
        return false;
    
    // 가장 큰 얼굴을 찾는다.
    for (size_t i = 0; i < heads.size(); ++i) {
        if(heads[i].width > biggestRect.width) {
            biggestRect = heads[i];
            biggestindex = i;
        }
    }
    target = heads[biggestindex];
    target &= Rect(0, 0, image.cols, image.rows);

    
    // 얼굴을 분석한다.
    Mat roi(hue, target), maskroi(mask, target);
    float hranges[] = {0,180};
    const float* phranges = hranges;
    cout << phranges[1] << endl;
    calcHist(&roi, 1, 0, maskroi, histogram, 1, &hsize, &phranges);
    normalize(histogram, histogram, 0, 255, NORM_MINMAX);
    
    drawHistogram();
    
    headDetected = true;
    
    return true;
}

bool HeadTracker::track() {
    
    if (!headDetected) {
        return false;
    }
    
    updateFrame();
    
    calcBackProject(&hue, 1, 0, histogram, backproj, &phranges);
    backproj &= mask;
    trace = CamShift(backproj, target, TermCriteria(TermCriteria::EPS | TermCriteria::COUNT, 10, 1));
    
    if (target.area() <= 1) {
        int cols = backproj.cols, rows = backproj.rows, r = (MIN(cols, rows) + 5)/6;
        target = Rect(target.x - r, target.y - r, target.x + r, target.y + r) & Rect(0, 0, cols, rows);
    }
    
    return true;
}
