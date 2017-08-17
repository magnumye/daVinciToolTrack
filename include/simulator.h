/**
* Simulator.h
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

#include <vtkSmartPointer.h>
#include <vtkSTLReader.h>
#include <vtkPLYReader.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkTransform.h>
#include <vtkTransformPolyDataFilter.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkAxesActor.h>
#include <vtkActor.h>
#include <vtkWindowToImageFilter.h>
#include <vtkImageData.h>
#include <vtkTriangleFilter.h>
#include <vtkStripper.h>

#include <QObject>
#include <QTimer>
#include <QMutex>

#include <time.h>
#include "Eigen/Geometry"


#include "camera.h"

#include "reader_camera.h"
#include "reader_robot.h"

#include "psm_tool.h"

#include "QGODetector.h"

#ifdef __linux__    // Linux ROS specific
#include <sys/time.h> // for framerate calculation
#endif

#include "CameraProcessor.h"


class GraphicalUserInterface;
class CameraProcessor;

class Simulator : public QObject
{
	Q_OBJECT
		friend class GraphicalUserInterface;
public:
	Simulator(std::string config_path);

	~Simulator();

	bool init();

	void setupGUI(GraphicalUserInterface *g);

	////////////////
	/// Locked functions
	void WriteVirtualRenderingOutput(const cv::Mat& render_img_flip_cv, const cv::Mat& camera_image,
		const std::vector<cv::Rect>& psm1_part_boxes, 
		const std::vector<std::string>& psm1_class_names, const std::vector<cv::Point2f>& psm1_projectedKeypoints,
		const std::vector<cv::Point3f>& psm1_allKeypoints_no_corr,
		const std::vector<int>& psm1_template_half_sizes,
		const std::vector<cv::Rect>& psm2_part_boxes, 
		const std::vector<std::string>& psm2_class_names, const std::vector<cv::Point2f>& psm2_projectedKeypoints,
		const std::vector<cv::Point3f>& psm2_allKeypoints_no_corr,
		const std::vector<int>& psm2_template_half_sizes, 
		const Eigen::Matrix<double, 4, 4, Eigen::RowMajor>& psm1_bHe,
		const Eigen::Matrix<double, 4, 4, Eigen::RowMajor>& psm1_bHj4,
		const Eigen::Matrix<double, 4, 4, Eigen::RowMajor>& psm1_bHj5,
		const Eigen::Matrix<double, 4, 4, Eigen::RowMajor>& psm2_bHe,
		const Eigen::Matrix<double, 4, 4, Eigen::RowMajor>& psm2_bHj4,
		const Eigen::Matrix<double, 4, 4, Eigen::RowMajor>& psm2_bHj5,
		float psm1_jaw, float psm2_jaw, 
		float psm1_slope, float psm2_slope
		);
	bool ReadVirtualRenderingOutput(cv::Mat& render_img_flip_cv_local, cv::Mat& camera_image_local,
		std::vector<cv::Rect>& psm1_part_boxes_local, 
		std::vector<std::string>& psm1_class_names_local, std::vector<cv::Point2f>& psm1_projectedKeypoints_local,
		std::vector<cv::Point3f>& psm1_allKeypoints_no_corr_local,
		std::vector<int>& psm1_template_half_sizes_local,
		std::vector<cv::Rect>& psm2_part_boxes_local, 
		std::vector<std::string>& psm2_class_names_local, std::vector<cv::Point2f>& psm2_projectedKeypoints_local,
		std::vector<cv::Point3f>& psm2_allKeypoints_no_corr_local,
		std::vector<int>& psm2_template_half_sizes_local, 
		Eigen::Matrix<double, 4, 4, Eigen::RowMajor>& psm1_bHe_local,
		Eigen::Matrix<double, 4, 4, Eigen::RowMajor>& psm1_bHj4_local,
		Eigen::Matrix<double, 4, 4, Eigen::RowMajor>& psm1_bHj5_local,
		Eigen::Matrix<double, 4, 4, Eigen::RowMajor>& psm2_bHe_local,
		Eigen::Matrix<double, 4, 4, Eigen::RowMajor>& psm2_bHj4_local,
		Eigen::Matrix<double, 4, 4, Eigen::RowMajor>& psm2_bHj5_local,
		float& psm1_jaw_local, float& psm2_jaw_local, 
		float& psm1_slope_local, float& psm2_slope_local
		);
	void WriteCameraOutput(bool T_cam_set, bool T2_cam_set, const cv::Mat& corr_T, const cv::Mat& corr_T2);
	void ReadCameraOupput(bool& T_cam_set_local, bool& T2_cam_set_local, cv::Mat& corr_T_local, cv::Mat& corr_T2_local);


	//////////////////////////
	// Calculate Key point of LND
	// Output a vector of points in camera coordinate
	// Index is defined as follow:
	// 0:   shaft_pivot_flat
	// 1:   shaft_pivot_deep
	// 2:   logo_pin_flat
	// 3:   logo_pin_deep
	// 4:   logo_wheel_flat
	// 5:   logo_wheel_deep
	// 6:   logo_is_flat
	// 7:   logo_is_deep
	// 8:   logo_idot_flat
	// 9:   logo_idot_deep
	// 10:  logo_pivot_flat
	// 11:  logo_pivot_deep
	// 12:  tip_flat (right)
	// 13:  tip_deep (left)
	// 14:	shaft_centre
	std::vector<cv::Point3f> calc_keypoints(const cv::Mat &bHshaft,
		const cv::Mat &bHlogo,
		const cv::Mat &bHe,
		const float jaw_ang,
		const cv::Mat &cHb,
		const int psm_num);

	// Calculate extra point of LND
	// Output a vector of points in camera coordinate
	// Index is defined as follow:
	// 0:   shaft_centre
	std::vector<cv::Point3f> calc_extrapoints(const cv::Mat &bHshaft,
		const cv::Mat &cHb,
		const int psm_num);
	PsmTool *psm1_tool;

	PsmTool *psm2_tool;

protected:
	// Virtual camera for VTK
	MonoCamera *m_cam;

	vtkSmartPointer<vtkLight> m_light;

	void show_simulation(bool checked);
	void show_kinematics(bool checked);

	// setup record and start
	void setup_record();

	// stop record and save files
	void stop_record();

private:
	//////////////////////////////////
	////// Virutal rendering variables

	// VTK renderer
	vtkSmartPointer<vtkRenderer> renderer;

	// VTK renderWindow
	vtkSmartPointer<vtkRenderWindow> renderWin;

	// VTK PBO for async RGBA and Z buffer access
	float *RawZBuffer;
	size_t RawZBufferSize;

	vtkSmartPointer<vtkWindowToImageFilter> windowToImageFilter;

	vtkSmartPointer<vtkImageData> render_img_vtk;

	Eigen::Matrix<double, 4, 4, Eigen::RowMajor> b1Hb2;

	///// End of virtual rendering variables
	//////////////////////////////////

	//////////////////////////////////
	////// Shared processing variables

	bool data_ready;

	cv::Mat render_img_cv;

	cv::Mat render_img_flip_cv_shared;
	// Index is defined as follow:
	// 0:   shaft_pivot_flat
	// 1:   shaft_pivot_deep
	// 2:   logo_pin_flat
	// 3:   logo_pin_deep
	// 4:   logo_wheel_flat
	// 5:   logo_wheel_deep
	// 6:   logo_is_flat
	// 7:   logo_is_deep
	// 8:   logo_idot_flat
	// 9:   logo_idot_deep
	// 10:  logo_pivot_flat
	// 11:  logo_pivot_deep
	// 12:  tip_flat (right)
	// 13:  tip_deep (left)
	// 14:	shaft_centre
	// Read-only
	std::vector<std::string> LND_partnames;
	std::vector<cv::Scalar> LND_partcolors;
	std::map<std::string, int> LND_partname2ids;
	std::vector<float> LND_part_tolerances;
	///


	std::vector<int> LND_template_half_sizes;
	std::vector<int> LND_la_template_half_sizes;
	std::vector<int> LND_ra_template_half_sizes;

	bool T_cam_set, T2_cam_set;
	bool T_cam_need_set;
	bool T2_cam_need_set;
	cv::Mat corr_T;
	cv::Mat corr_T2;

	int time_stamp;
	std::vector<int> LND_la_template_half_sizes_shared;
	std::vector<int> LND_ra_template_half_sizes_shared;
	bool T_cam_set_shared, T2_cam_set_shared;
	cv::Mat corr_T_shared;
	cv::Mat corr_T2_shared;
	cv::Mat camera_image_shared;

	Eigen::Matrix<double, 4, 4, Eigen::RowMajor> psm1_bHe_shared, psm1_bHj4_shared, psm1_bHj5_shared;
	Eigen::Matrix<double, 4, 4, Eigen::RowMajor> psm2_bHe_shared, psm2_bHj4_shared, psm2_bHj5_shared;
	float psm1_jaw_shared, psm2_jaw_shared;


	std::vector<cv::Rect> psm1_part_boxes_shared; 
	std::vector<std::string> psm1_class_names_shared;
	std::vector<cv::Point2f> psm1_projectedPoints_shared, psm1_projectedKeypoints_shared;
	std::vector<cv::Point3f> psm1_allKeypoints_no_corr_shared;

	std::vector<cv::Rect> psm2_part_boxes_shared;
	std::vector<std::string> psm2_class_names_shared;
	std::vector<cv::Point2f> psm2_projectedPoints_shared, psm2_projectedKeypoints_shared;
	std::vector<cv::Point3f> psm2_allKeypoints_no_corr_shared;

	float psm1_shaft_slop_shared;
	float psm2_shaft_slop_shared;

	///// End of shared processing variables
	//////////////////////////////////


	//////////////////////////////////
	////// Camera processing variables

	CameraProcessor* cam_proc;


	///// End of camera processing variables
	//////////////////////////////////

	// list of directory
	std::string mod_dir;

	std::string tool_config_dir;

	std::string data_dir;

	std::string cam_dir;

	std::string handeye_dir;

	std::string video_dir;

	std::string psm1_dir;
	std::string psm2_dir;


	ReaderRobot *psm1_file;
	ReaderRobot *psm2_file;

	ReaderCamera *cam_file;

	QTimer pose_timer;

	void start_listener();

	// Process new pose, transform model and update GUI
	void process();

	// Process new frame and update GUI
	bool capture_camera(cv::Mat& img);
	bool render_view(cv::Mat &cloud);


	void virtual_rendering_thread();

	void calc_shaft_slope(const cv::Mat &cHb,
		const Eigen::Matrix<double, 4, 4, Eigen::RowMajor> &psm_Hj4,
		const Eigen::Matrix<double, 4, 4, Eigen::RowMajor> &psm_Hj5,
		const Eigen::Matrix<double, 4, 4, Eigen::RowMajor> &psm_He,
		const float jaw_in,
		float& psm_slope);

	// Linked gui, use setupGUI() function
	GraphicalUserInterface *gui_;

	// Live/file mode, in Windows only read file
	bool is_live_;

	bool show_sim;

	// if recording current scene (without rendering)
	bool is_recording_;

	std::string active_arm_;

	// Current rendered cloud
	cv::Mat cloud_;

	QMutex mx_cld_;

	QMutex mx_virt_out;
	QMutex mx_cam_out;
	QMutex mx_data_ready;

	void init_LND_parameters();

	void calc_kpts_2D_sim(const cv::Mat& camHbase_psm, const cv::Mat& camHbase_psm_no_corr,
		Eigen::Matrix<double, 4, 4, Eigen::RowMajor> &psm_Hj4,
		Eigen::Matrix<double, 4, 4, Eigen::RowMajor> &psm_Hj5,
		Eigen::Matrix<double, 4, 4, Eigen::RowMajor> &psm_He,
		float jaw_in, std::vector<cv::Point2f>& projectedPoints, 
		std::vector<cv::Point2f>& projectedKeypoints,
		std::vector<bool> &visibilities, const cv::Mat& depth_map,
		std::vector<cv::Point3f>& cHpkey, bool& sc_visible,
		cv::Point2f& shaft_centre, std::vector<int>& template_half_sizes,
		int psm_num);
	void draw_pts(cv::Mat &img, const std::vector<cv::Point2f>& projectedPoints, 
		const std::vector<cv::Point2f>& projectedKeypoints,
		const std::vector<bool>& visibilities);
	void check_parts_visible(const cv::Mat &depth_map, const std::vector<cv::Point3f> &keypoints,
		const std::vector<cv::Point2f>& projectedKeypoints, std::vector<bool> &visibilities,
		const std::vector<int>& template_half_sizes);

	void check_parts_visible(const std::vector<cv::Point3f> &keypoints,
		const std::vector<cv::Point2f>& projectedKeypoints, std::vector<bool> &visibilities,
		const std::vector<int>& template_half_sizes);

	double get_orientation(std::vector<cv::Point> &contour, cv::Mat &img);

	bool detect_shaft(cv::Mat &camera_image,const cv::Mat &binary_mask,
		const cv::Rect &roi);

	cv::Point3f get_3D (const cv::Point &pt);


#ifdef _WIN32
	int count_processed; 
	cv::VideoWriter full_vid_wrt;
#endif


	QImage cvtCvMat2QImage(const cv::Mat & image);


	void decompose_rotation_xyz(const cv::Mat &R, double& thetaX,
		double& thetaY, double& thetaZ);
	void compose_rotation(const double &thetaX, const double &thetaY, 
		const double &thetaZ, cv::Mat &R);


#ifdef __linux__    // Linux ROS specific
	// Framerate calculation
	std::deque<timeval> time_count;
#endif

#ifdef _WIN32
	std::deque<time_t> time_count;
#endif
	/************************************************************************/
	/* Signals and slots                                                    */
	/************************************************************************/
Q_SIGNALS:
	void updateRenderImage(QImage);
	void updateCameraImage(QImage);
	void dispFrameRate(QString);
};

