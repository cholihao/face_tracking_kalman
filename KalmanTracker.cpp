///////////////////////////////////////////////////////////////////////////////
// KalmanTracker.cpp: KalmanTracker Class Implementation Declaration

#include "KalmanTracker.h"
#include <iostream>
#include <fstream>
#include <iomanip> // to format image names using setw() and setfill()


int KalmanTracker::kf_count = 0;

void KalmanTracker::init_kf(StateType stateMat)
{
    kf = NNKalmanFilter(7,4);
    Z = MatrixXf(4,1);
    Z.setZero();

    VectorXf X(7, 1); //statePre
    X << VectorXf::Zero(7,1);

    VectorXf X0(7, 1); //statePre
    X0 << VectorXf::Zero(7,1);

    MatrixXf A(7, 7); //transitionMatrix;
    A << 1, 0, 0, 0, 1, 0, 0,
      0, 1, 0, 0, 0, 1, 0,
      0, 0, 1, 0, 0, 0, 1,
      0, 0, 0, 1, 0, 0, 0,
      0, 0, 0, 0, 1, 0, 0,
      0, 0, 0, 0, 0, 1, 0,
      0, 0, 0, 0, 0, 0, 1;

    MatrixXf Q(7, 7); //Q << 1e-2, 1e-2, 1e-2, 1e-2;
    Q << MatrixXf::Identity(7,7);
    Q = Q * 1e-2;
    cout << Q << endl << endl;

    MatrixXf H(4, 7); //measurementMatrix
    H << MatrixXf::Identity(4, 7);
    cout << H << endl << endl;

    MatrixXf R(4, 4); //R << 1e-1, 1e-1, 1e-1, 1e-1;
    R << MatrixXf::Identity(4,4);
    R = R * 1e-1;
    cout << R << endl << endl;

    MatrixXf P0(7, 7);
    P0.setIdentity();
    cout << P0 << endl << endl;

    MatrixXf P(7, 7);
    P << MatrixXf::Zero(7, 7);

    MatrixXf K(7, 4);
    K << MatrixXf::Zero(7, 4);

    //
    /* Initialize the Filter*/
    kf.setFixed(X, A, H, Q, R, P, K);
    kf.setInitial(X0, P0);

    // initialize state vector with bounding box in [cx,cy,s,r] style
    kf.X0(0,0) = stateMat.x + stateMat.width / 2;
    kf.X0(1,0) = stateMat.y + stateMat.height / 2;
    kf.X0(2,0) = stateMat.area();
    kf.X0(3,0) = stateMat.width / stateMat.height;
}

/*
// initialize Kalman filter
void KalmanTracker::init_kf(StateType stateMat)
{
	int stateNum = 7;
	int measureNum = 4;

	kf = NKalmanFilter(stateNum, measureNum, 0);

	measurement = Mat::zeros(measureNum, 1, CV_32F);

	kf.transitionMatrix = (Mat_<float>(stateNum, stateNum) <<
		1, 0, 0, 0, 1, 0, 0,
		0, 1, 0, 0, 0, 1, 0,
		0, 0, 1, 0, 0, 0, 1,
		0, 0, 0, 1, 0, 0, 0,
		0, 0, 0, 0, 1, 0, 0,
		0, 0, 0, 0, 0, 1, 0,
		0, 0, 0, 0, 0, 0, 1);

	setIdentity(kf.measurementMatrix); //H
	setIdentity(kf.processNoiseCov, Scalar::all(1e-2)); //Q
	setIdentity(kf.measurementNoiseCov, Scalar::all(1e-1)); //R
	setIdentity(kf.errorCovPost, Scalar::all(1)); //P0

	std::cout << kf.measurementMatrix << std::endl << std::endl;
	std::cout << kf.processNoiseCov << std::endl << std::endl;
	std::cout << kf.measurementNoiseCov << std::endl << std::endl;
	std::cout << kf.errorCovPost << std::endl << std::endl;
	// initialize state vector with bounding box in [cx,cy,s,r] style
	//std::cout << stateMat.x << " " << stateMat.width/2 << std::endl;
	//std::cout << stateMat.y << " " << stateMat.height/2 << std::endl;
	//std::cout << stateMat.area() << std::endl;
	//std::cout << stateMat.width << " " << stateMat.height << std::endl;

	kf.statePost.at<float>(0, 0) = stateMat.x + stateMat.width / 2;
	kf.statePost.at<float>(1, 0) = stateMat.y + stateMat.height / 2;
	kf.statePost.at<float>(2, 0) = stateMat.area();
	kf.statePost.at<float>(3, 0) = stateMat.width / stateMat.height;

	std::cout << kf.statePost.at<float>(0, 0) << std::endl;
	std::cout << kf.statePost.at<float>(1, 0) << std::endl;
	std::cout << kf.statePost.at<float>(2, 0) << std::endl;
	std::cout << kf.statePost.at<float>(3, 0) << std::endl;
}
*/

// Predict the estimated bounding box.
StateType KalmanTracker::predict()
{
	// predict
	//Mat p = kf.predict();
	kf.predict();
	m_age += 1;

	if (m_time_since_update > 0)
		m_hit_streak = 0;
	m_time_since_update += 1;

	//std::cout << p.at<float>(0, 0) << " " << p.at<float>(1, 0)  << " " << p.at<float>(2, 0) << " " << p.at<float>(3, 0) << std::endl;
	//std::cout << kf.X(0, 0) << " " << kf.X(1, 0)  << " " << kf.X(2, 0) << " " << kf.X(3, 0) << std::endl;
	//StateType predictBox = get_rect_xysr(p.at<float>(0, 0), p.at<float>(1, 0), p.at<float>(2, 0), p.at<float>(3, 0));
	StateType predictBox = get_rect_xysr(kf.X(0, 0), kf.X(1, 0), kf.X(2, 0), kf.X(3, 0));


	m_history.push_back(predictBox);
	return m_history.back();
}


// Update the state vector with observed bounding box.
void KalmanTracker::update(StateType stateMat)
{
	this->lastRect = stateMat;

	m_time_since_update = 0;
	m_history.clear();
	m_hits += 1;
	m_hit_streak += 1;

	// measurement
	//measurement.at<float>(0, 0) = stateMat.x + stateMat.width / 2;
	//measurement.at<float>(1, 0) = stateMat.y + stateMat.height / 2;
	//measurement.at<float>(2, 0) = stateMat.area();
	//measurement.at<float>(3, 0) = stateMat.width / stateMat.height;
	Z(0,0) = stateMat.x + stateMat.width / 2;
	Z(1,0) = stateMat.y + stateMat.height / 2;
	Z(2,0) = stateMat.area();
	Z(3,0) = stateMat.width / stateMat.height;

	// update
	kf.correct(Z);
}

/*
// Return the current state vector
StateType KalmanTracker::get_state()
{
	Mat s = kf.statePost;
	return get_rect_xysr(s.at<float>(0, 0), s.at<float>(1, 0), s.at<float>(2, 0), s.at<float>(3, 0));
}

*/


// Convert bounding box from [cx,cy,s,r] to [x,y,w,h] style.
StateType KalmanTracker::get_rect_xysr(float cx, float cy, float s, float r)
{
	float w = sqrt(s * r);
	float h = s / w;
	float x = (cx - w / 2);
	float y = (cy - h / 2);

	if (x < 0 && cx > 0)
		x = 0;
	if (y < 0 && cy > 0)
		y = 0;

	return StateType(x, y, w, h);
}



/*
// --------------------------------------------------------------------
// Kalman Filter Demonstrating, a 2-d ball demo
// --------------------------------------------------------------------

const int winHeight = 600;
const int winWidth = 800;

Point mousePosition = Point(winWidth >> 1, winHeight >> 1);

// mouse event callback
void mouseEvent(int event, int x, int y, int flags, void *param)
{
	if (event == CV_EVENT_MOUSEMOVE) {
		mousePosition = Point(x, y);
	}
}

void TestKF();

void main()
{
	TestKF();
}


void TestKF()
{
	int stateNum = 4;
	int measureNum = 2;
	KalmanFilter kf = KalmanFilter(stateNum, measureNum, 0);

	// initialization
	Mat processNoise(stateNum, 1, CV_32F);
	Mat measurement = Mat::zeros(measureNum, 1, CV_32F);

	kf.transitionMatrix = *(Mat_<float>(stateNum, stateNum) <<
		1, 0, 1, 0,
		0, 1, 0, 1,
		0, 0, 1, 0,
		0, 0, 0, 1);

	setIdentity(kf.measurementMatrix);
	setIdentity(kf.processNoiseCov, Scalar::all(1e-2));
	setIdentity(kf.measurementNoiseCov, Scalar::all(1e-1));
	setIdentity(kf.errorCovPost, Scalar::all(1));

	randn(kf.statePost, Scalar::all(0), Scalar::all(winHeight));

	namedWindow("Kalman");
	setMouseCallback("Kalman", mouseEvent);
	Mat img(winHeight, winWidth, CV_8UC3);

	while (1)
	{
		// predict
		Mat prediction = kf.predict();
		Point predictPt = Point(prediction.at<float>(0, 0), prediction.at<float>(1, 0));

		// generate measurement
		Point statePt = mousePosition;
		measurement.at<float>(0, 0) = statePt.x;
		measurement.at<float>(1, 0) = statePt.y;

		// update
		kf.correct(measurement);

		// visualization
		img.setTo(Scalar(255, 255, 255));
		circle(img, predictPt, 8, CV_RGB(0, 255, 0), -1); // predicted point as green
		circle(img, statePt, 8, CV_RGB(255, 0, 0), -1); // current position as red

		imshow("Kalman", img);
		char code = (char)waitKey(100);
		if (code == 27 || code == 'q' || code == 'Q')
			break;
	}
	destroyWindow("Kalman");
}
*/
