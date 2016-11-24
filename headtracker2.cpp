#include "opencv2/objdetect.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"

#include <iostream>

using namespace std;
using namespace cv;

const string cascadePath = "./haarcascade_frontalface_alt.xml";
const string windowName = "Tracker 3D";

enum Mode {
    TRACK, AUTO_DETECT, MANUAL_DETECT, BACK_PROJECTION
};

void createCube(Point3f c, int size);
void updateWorld(Point3f& view);
void createWorld();

struct Line {
    Point3f p1;
    Point3f p2;
    Scalar color;
};

struct Triangle {
    Point3f p1;
    Point3f p2;
    Point3f p3;
    Scalar color;
};

vector<Line> lines;
vector<Triangle> triangles;
Point3f camera;
int depth = 600;

float screenRatio = 1280/720;
float screenWidth = 1000;
float screenHeight;

void captureVideo();
bool detectHead(CascadeClassifier& cascade);
RotatedRect track();

VideoCapture capture;
Mat capturedFrame;
Rect trackingRegion;
RotatedRect trackedRegion;
float hueRange[] = {0, 180};
Mat histogram;
int histogramSize = 24;
const float* histogramRange = hueRange;

Mat hsv, hue, mask, backproj, image;

int vmin = 10, vmax = 256, smin = 50;

bool pause = false;
bool trackingAvailable = false;

Mode mode = AUTO_DETECT;

bool selectingRegion = false;
bool regionSelected = false;
Rect selection;
Point origin;

Mat canvas;


static void onMouse(int event, int x, int y, int, void*) {
    
    if (selectingRegion) {
        selection.x = MIN(x, origin.x);
        selection.y = MIN(y, origin.y);
        selection.width = std::abs(x - origin.x);
        selection.height = std::abs(y - origin.y);
        selection &= Rect(0, 0, capturedFrame.cols, capturedFrame.rows);
    }
    
    switch (event) {
        case EVENT_LBUTTONDOWN:
            selectingRegion = true;
            mode = MANUAL_DETECT;
            origin = Point(x,y);
            selection = Rect(x,y,0,0);
            break;
        case EVENT_LBUTTONUP:
            selectingRegion = false;
            mode = TRACK;
            if (selection.width > 0 && selection.height > 0)
                regionSelected = true;
            break;
    }
}

int main(int argc, const char** argv) {
    
    CascadeClassifier cascade;
    
    if(!cascade.load(cascadePath)) {
        cerr << "cascade 파일 불러오기 실패" << endl;
        return -1;
    }
    
    // 카메라 세팅
    capture.open(0);
    if (!capture.isOpened()) {
        cout << "카메라를 여는 데 실패하였습니다." << endl;
        return -1;
    }
    screenRatio = capture.get(CV_CAP_PROP_FRAME_HEIGHT) / capture.get(CV_CAP_PROP_FRAME_WIDTH);
    screenHeight = screenWidth * screenRatio;
    capture.set(CV_CAP_PROP_FRAME_WIDTH, screenWidth);
    capture.set(CV_CAP_PROP_FRAME_HEIGHT, screenHeight);
    
    // UI 세팅
    namedWindow(windowName, 0 );
    setMouseCallback(windowName, onMouse, 0);
    createTrackbar("Maximum Value", windowName, &vmin, 256, 0 );
    createTrackbar("Minimum Value", windowName, &vmax, 256, 0 );
    createTrackbar("Saturation", windowName, &smin, 256, 0 );
    
    // 스크린 세팅
    createWorld();
    canvas.create(screenHeight, screenWidth, CV_8UC3);
    
    while(true) {
        
        if (!pause || capturedFrame.empty())
            captureVideo();
        
        switch (mode) {
            case TRACK:
                if (trackingAvailable && !pause) {
                    trackedRegion = track();
                }
                
                if (trackedRegion.size.area() >= 1) {
                    
                }
                
                camera.x = screenWidth - trackedRegion.center.x;
                camera.y = trackedRegion.center.y;
                
                 depth = 600 - sqrt(abs(camera.x - screenWidth / 2));
                
                camera.z = depth;
                
                updateWorld(camera);
                imshow(windowName, canvas);
                break;
            case AUTO_DETECT:
                trackingAvailable = false;
                if (detectHead(cascade)) {
                    trackingAvailable = true;
                    cout << "face detected" << endl;
                    mode = TRACK;
                }
                break;
            case MANUAL_DETECT:
                if (selection.width > 0 && selection.height > 0) {
                    Mat roi(capturedFrame, selection);
                    bitwise_not(roi, roi);
                }
                imshow(windowName, capturedFrame);
                break;
            case BACK_PROJECTION:
                if (trackingAvailable) {
                    RotatedRect region = track();
                    cvtColor(backproj, capturedFrame, COLOR_GRAY2BGR );
                    ellipse(capturedFrame, region, Scalar(0,255, 0), 3, LINE_AA);
                }
                imshow(windowName, capturedFrame);
                break;
            default:
                break;
        }
        
        char key = (char) waitKey(1);
        if (key == 27)
            break;
        
        switch (key) {
            case 'b':
                mode = BACK_PROJECTION;
                pause = false;
                cout << "back projection mode" << endl;
                break;
            case 'a':
                mode = AUTO_DETECT;
                pause = false;
                cout << "auto detection mode" << endl;
                break;
            case 'm':
                mode = MANUAL_DETECT;
                cout << "manual detection mode" << endl;
                break;
            case 'p':
                pause = !pause;
                break;
            case 't':
                mode = TRACK;
                break;
            default:
                //mode = TRACK;
                break;
        }
        
    }
    return 0;
}


Line createLine(Point3f p1, Point3f p2, Scalar color) {
    Line line;
    line.p1 = p1;
    line.p2 = p2;
    line.color = color;
    return line;
}

Triangle createTriangle(Point3f p1, Point3f p2, Point3f p3, Scalar color) {
    Triangle triangle;
    triangle.p1 = p1;
    triangle.p2 = p2;
    triangle.p3 = p3;
    triangle.color = color;
    return triangle;
}


void createCube(Point3f c, int size) {
    
    int s = size / 2;
    
    Point3f vertices[8];
    
    for (int i = 0; i < 8; i++) {
        vertices[i].x = c.x + (i < 4 ? s : - s);
        vertices[i].y = c.y + ((int)(i / 2) % 2 == 0 ? s : - s);
        vertices[i].z = c.z + (((i > 0 && 3 > i) || (i > 4 && 7 > i)) ? -s : s);
    }
    
    for (int i = 0; i < 4; i++) {
        lines.push_back(createLine(vertices[i], vertices[i == 3 ? 0 : i + 1], Scalar(0,0,0)));
        lines.push_back(createLine(vertices[i + 4], vertices[i == 3 ? 4 : i + 5], Scalar(0,0,0)));
        lines.push_back(createLine(vertices[i], vertices[i + 4], Scalar(0,0,0)));
    }
    
}

float rnd() {
    return (float)(rand() % 101) / (float)100;
}

void createWorld() {
    
    int w = screenWidth, h = screenHeight;
    
    for (int i = 0; i < 10; i++) {
        createCube(Point3f(w * rnd() , h * rnd(), -1000 * rnd()), 300 * rnd());
    }
}


// z = 0 평면으로 사영한다.
void project(Point3f& eye, Point3f& tgt, Point2f& proj) {
    
    float aez = abs(eye.z);
    float aet = abs(tgt.z);
    
    // Ze 와 Zp 로 내분한 점이 사영점이다.
    proj.x = (eye.x * aet + tgt.x * aez) / (aez + aet);
    proj.y = (eye.y * aet + tgt.y * aez) / (aez + aet);
}

void updateWorld(Point3f& view) {
    
    canvas = Scalar(255, 255, 255);
    
    Point2f proj1;
    Point2f proj2;
    
    for (size_t i = 0; i < lines.size(); i++) {
        Line line = lines.at(i);
        project(view, line.p1, proj1);
        project(view, line.p2, proj2);
        
        cv::line(canvas, proj1, proj2, line.color);
    }
}



void captureVideo() {
    
    if (!capture.isOpened())
        capture.open(0);
    
    capture >> capturedFrame;
    if (capturedFrame.empty())
        return;
    
    // 프레임에서 HSV 이미지를 취득한다.
    cvtColor(capturedFrame, hsv, COLOR_BGR2HSV);
    
    // 이미지를 보정한다.
    int _vmin = vmin, _vmax = vmax;
    inRange(hsv, Scalar(0, smin, MIN(_vmin,_vmax)), Scalar(180, 256, MAX(_vmin, _vmax)), mask);
    
    // HSV 이미지에서 Hue 채널을 분리한다.
    const int channelMask[] = {0, 0};
    hue.create(hsv.size(), hsv.depth());
    mixChannels(&hsv, 1, &hue, 1, channelMask, 1);
}

bool detectHead(CascadeClassifier& cascade) {
    
    vector<Rect> heads;
    
    Rect biggestRegion;
    size_t biggestindex = 0;
    
    cascade.detectMultiScale(capturedFrame, heads, 1.1, 2, 0|CASCADE_SCALE_IMAGE, Size(30, 30));
    
    // 얼굴이 발견되지 않으면 리턴한다.
    if (heads.size() < 1)
        return false;
    
    // 가장 큰 얼굴을 찾는다.
    for (size_t i = 0; i < heads.size(); ++i) {
        if(heads[i].width > biggestRegion.width) {
            biggestRegion = heads[i];
            biggestindex = i;
        }
    }
    trackingRegion = heads[biggestindex];
    trackingRegion &= Rect(0, 0, capturedFrame.cols, capturedFrame.rows);
    
    // 얼굴을 분석한다.
    Mat roi(hue, trackingRegion), maskroi(mask, trackingRegion);

    calcHist(&roi, 1, 0, maskroi, histogram, 1, &histogramSize, &histogramRange);
    normalize(histogram, histogram, 0, 255, NORM_MINMAX);
    
    return true;
}

RotatedRect track() {
    
    calcBackProject(&hue, 1, 0, histogram, backproj, &histogramRange);
    backproj &= mask;
    
    RotatedRect region = CamShift(backproj, trackingRegion, TermCriteria(TermCriteria::EPS | TermCriteria::COUNT, 10, 1));
    
    if (trackingRegion.area() <= 1) {
        int cols = backproj.cols, rows = backproj.rows, r = (MIN(cols, rows) + 5)/6;
        trackingRegion = Rect(trackingRegion.x - r, trackingRegion.y - r, trackingRegion.x + r, trackingRegion.y + r) & Rect(0, 0, cols, rows);
    }
    
    return region;
}
