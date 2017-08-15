/**
* CameraProcessor.h
* 
* Reference:
* M. Ye, et al. "Real-time 3D Tracking of Articulated Tools for Robotic Surgery". 
* MICCAI 2016.
*
* @author  Menglong Ye, Imperial College London
* @contact magnumye@gmail.com
* @license BSD License
*/
#pragma once

#include <QObject>
#include <QThread>
#include <QTimer>
#include <QImage>

#include "QGODetector.h"
#include "ExtendedKalmanFilter.h"
#include "simulator.h"

class Simulator;


class CameraProcessor : public QObject
{
	Q_OBJECT
public:
	CameraProcessor(double fx, double fy, double px, double py, int w, int h,
		const cv::Mat& cHw_, const cv::Mat& cHw2_,
		const std::vector<cv::Scalar>& LND_partcolors,
		const std::map<std::string, int>& LND_partname2ids);
	~CameraProcessor();

	void set_simulator(Simulator* sim);

	void start();
	
	void show_kinematics(bool checked);

signals:
	void updateCameraImage(QImage);
	void dispFrameRate(QString);

	public slots:
		void process();
		
protected:
	QTimer timer_;
	QThread thread_;

	Simulator* simulator_;
private:

	// Show kinematics
	bool show_kine;

	double Fx;
	double Fy;
	double Px;
	double Py;
	int width;
	int height;


	cv::Mat cHw, cHw2;
	cv::Mat corr_T;
	cv::Mat corr_T2;
	std::vector<cv::Scalar> tool_partcolors;
	std::map<std::string, int> tool_partname2ids;

	bool T_cam_need_set;
	bool T2_cam_need_set;
	cv::Mat corr_T_cam;
	cv::Mat corr_T2_cam;

	int low_num_threshold_psm1;
	int high_num_threshold_psm1;
	int low_num_threshold_psm2;
	int high_num_threshold_psm2;

	int camera_img_id;

	// Tool detector
	QGODetector *qgo_detector;

	// EKF related
	ExtendedKalmanFilter* ekFilter_psm1;
	ExtendedKalmanFilter* ekFilter_psm2;

	bool process_camera(const std::vector<cv::Point3f>& psm1_allKeypoints,
		const std::vector<cv::Point2f>& psm1_projectedKeypoints,
		const std::vector<cv::Point3f>& psm2_allKeypoints,
		const std::vector<cv::Point2f>& psm2_projectedKeypoints,
		const std::vector<int>& ra_template_half_sizes,
		const std::vector<int>& la_template_half_sizes,
		cv::Mat& camera_image, cv::Mat& psm1_err_T,
		cv::Mat& psm2_err_T, bool& T_cam_set,
		bool& T2_cam_set, int img_id);
	void shape_context_verification(cv::Mat& img, const std::vector<Match>& matches, 
		const std::vector<cv::Point2f>& projectedKeypoints, 
		std::vector<cv::Point2f>& verifiedCamerapoints,
		std::vector<std::string>& verifiedNames,
		const std::vector<int>& template_half_sizes);
	void calc_H_z_estiamted(const std::vector<cv::Point3f>& kpts_cstar, 
		const std::vector<double>& state_estimated, 
		double f_x, double f_y, double d_x, double d_y,
		cv::Mat& H, cv::Mat& z_estimated);
	void initEKFPSM1(int state_dim, int measure_dim, int w_dim, int v_dim, cv::Mat& init_state);
	void initEKFPSM2(int state_dim, int measure_dim, int w_dim, int v_dim, cv::Mat& init_state);
	void draw_skel(cv::Mat &img,
		const cv::Mat &cHb,
		const Eigen::Matrix<double, 4, 4, Eigen::RowMajor> &psm_Hj4,
		const Eigen::Matrix<double, 4, 4, Eigen::RowMajor> &psm_Hj5,
		const Eigen::Matrix<double, 4, 4, Eigen::RowMajor> &psm_He,
		const float jaw_in, bool virt_or_cam, int psm_num);
	void draw_skel(cv::Mat &img, const cv::Mat &cHb,
		const Eigen::Matrix<double, 4, 4, Eigen::RowMajor> &psm_Hj4,
		const Eigen::Matrix<double, 4, 4, Eigen::RowMajor> &psm_Hj5,
		const Eigen::Matrix<double, 4, 4, Eigen::RowMajor> &psm_He,
		const float jaw_in, const cv::Mat& err_T, int psm_num, float slope);

	QImage cvtCvMat2QImage(const cv::Mat & image);

	Eigen::MatrixXd cvMat2Eigen(cv::Mat& m);
#ifdef __linux__    // Linux ROS specific
	// Framerate calculation
	std::deque<timeval> time_count;
	double time_to_double(timeval *t);
	double time_diff(timeval *t1, timeval *t2);
#endif

#ifdef _WIN32
	std::deque<time_t> time_count;
#endif

};

