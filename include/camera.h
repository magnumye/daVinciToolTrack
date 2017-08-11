#pragma once

#define _USE_MATH_DEFINES
#include <cmath>
#include <vtkCamera.h>
#include <vtkSmartPointer.h>
#include <opencv2/calib3d/calib3d.hpp>


class MonoCamera {

public:

	/**
	* Construct a camera with a calibration file. This file should be in the opencv calibration xml format.
	* @param[in] calibration_filename The path of the calibration file.
	*/
	explicit MonoCamera(const std::string &calibration_filename);

	/**
	* Construct a camera directly specifying the intrinsic and distortion parameters.
	* @param[in] intrinsic The intrinsic parameters of the camera.
	* @param[in] distortion The distortion parameters of the camera.
	* @param[in] image_width The width of the image plane in pixels.
	* @param[in] image_width The height of the image plane in pixels.
	*/
	MonoCamera(const cv::Mat &intrinsic, const cv::Mat &distortion, const int image_width, const int image_height);

	/**
	* Construct a camera setting its parameters to /f$f_{x} = 1000, f_{y} = 1000, c_{x} = 0.5\times ImageWidth, c_{y} = 0.5\times ImageHeight \f$.
	*/
	MonoCamera(){}

	/**
	* Destructor for the camera.
	*/
	virtual ~MonoCamera(){}

	/**
	* Setup the camera for vtk renderer.
	* Reference: http://stackoverflow.com/questions/25539898/how-to-apply-the-camera-pose-transformation-computed-using-epnp-to-the-vtk-camer
	*/
	vtkSmartPointer<vtkCamera> SetupCameraForRender();

	void set_window_size(int width, int height);

	void set_wHc(cv::Mat &H);
	void set_w2Hc(cv::Mat &H);
	void set_wHc_corr(cv::Mat &H);
	void set_w2Hc_corr(cv::Mat &H);

	/**
	* Camera focal length in x pixel dimensions.
	* @return The focal length.
	*/
	float Fx() const { return fx_; }

	/**
	* Camera focal length in y pixel dimensions.
	* @return The focal length.
	*/
	float Fy() const { return fy_; }

	/**
	* Camera principal point in the x dimension.
	* @return The principal point.
	*/
	float Px() const { return px_; }

	/**
	* Camera principal point in the y dimension.
	* @return The principal point.
	*/
	float Py() const { return py_; }

	/**
	* Get image width.
	* @return The image width.
	*/
	int Width() const { return image_width_; }

	/**
	* Get image height.
	* @return The image height.
	*/
	int Height() const { return image_height_; }

	/**
	* Camera principal point in the x dimension in window image coordinate.
	* @return The principal point.
	*/
	float Px_win() const { return px_win_; }

	/**
	* Camera principal point in the y dimension in window image coordinate.
	* @return The principal point.
	*/
	float Py_win() const { return py_win_; }

	/**
	* Get window width.
	* @return The window width.
	*/
	int Win_Width() const { return window_width_; }

	/**
	* Get window height.
	* @return The window height.
	*/
	int Win_Height() const { return window_height_; }

	/**
	* Get wHc, camera in world.
	* @return The wHc.
	*/
	cv::Mat wHc() const { return wHc_; }
	cv::Mat wHc_corr() const { return wHc_corr_; }
	/**
	* Get w2Hc, camera in 2nd world.
	* @return The wHc.
	*/
	cv::Mat w2Hc() const { return w2Hc_; }
	cv::Mat w2Hc_corr() const { return w2Hc_corr_; }

	inline vtkSmartPointer<vtkCamera> vtk_cam() {return m_vtkcam_;}

protected:
	float fx_; /**< The camera focal length in units of horizontal pixel length. */
	float fy_; /**< The camera focal length in units of vertical pixel length. */
	float px_; /**< The camera horizontal principal point in units of horizontal pixel length. */
	float py_; /**< The camera horizontal principal point in units of vertical pixel length. */
	float px_win_; /**< Principle point in enlarged image coordinate, in case window size is different from original image size. */
	float py_win_; /**< Principle point in enlarged image coordinate. */
	cv::Mat distortion_params_; /**< The camera distortion parameters. */

	int image_width_; /**< The width of the image plane in pixels. */
	int image_height_; /**< The height of the image plane in pixels. */

	int window_width_; /**< The width of the display window in pixels. */
	int window_height_; /**< The height of the display window in pixels. */

	cv::Point2d window_center; /**< Center of the window in viewport coordinates. The viewport coordinate range is ([-1,+1],[-1,+1]) */

	cv::Vec3f camera_center_; /**< The center of the camera with respect to the camera coordinate system (useful for offsetting a camera in stereo/trinocular rig). */
	cv::Vec3f world_up_; /**< The world up vector. */
	cv::Vec3f look_at_; /**< The camera look at vector. */

	cv::Mat wHc_; /**< The transformation of the camera eye relative to world coordinates. */
	cv::Mat w2Hc_; /**< The transformation of the camera eye relative to 2nd world coordinates. */
	cv::Mat wHc_corr_;
	cv::Mat w2Hc_corr_;
	vtkSmartPointer<vtkCamera> m_vtkcam_;

private:

	inline bool isEqual(double x, double y)
	{
		const double epsilon = 1e-5 /* some small number such as 1e-5 */;
		return std::abs(x - y) <= epsilon * std::abs(x);
		// see Knuth section 4.2.2 pages 217-218
	}

};

