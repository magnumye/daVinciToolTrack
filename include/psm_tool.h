#pragma once

// c++ std library
#include <string>

// VTK
#include <vtkSmartPointer.h>
#include <vtkSTLReader.h>
#include <vtkPLYReader.h>
#include <vtkPolyDataMapper.h>
#include <vtkActor.h>
#include <vtkTriangleFilter.h>
#include <vtkStripper.h>
#include <vtkTransform.h>
#include <vtkTransformPolyDataFilter.h>
#include <vtkAxesActor.h>


// OpenCV 2
#include <opencv2/core/core.hpp>


class PsmTool
{
public:
	enum ToolType
	{
		LND,
		MCS,
		ProGrasp,
		CF,
		MBF,
		RTS
	};
	PsmTool (ToolType type);

	void init(std::string mod_dir, std::string tool_config_dir);

	// Update Transform
	void Update_Shaft_Transform (const double *elements);

	void Update_Logo_Transform (const double *elements);

	void Update_JawL_Transform (const double *elements);

	void Update_JawR_Transform (const double *elements);

	// Get VTK Related
	vtkSmartPointer<vtkActor> Shaft_Actor () { return shaft_actor; }
	vtkSmartPointer<vtkActor> Logo_Actor () { return logo_actor; }
	vtkSmartPointer<vtkActor> JawL_Actor () { return jawL_actor; }
	vtkSmartPointer<vtkActor> JawR_Actor () { return jawR_actor; }

	// Get Keypoints
	cv::Mat &Shaft_Centre_Mat () { return shaft_centre;}

	cv::Mat &Shaft_Pivot_Flat_Mat () { return shaft_pivot_flat;}

	cv::Mat &Shaft_Pivot_Deep_Mat () { return shaft_pivot_deep;}

	cv::Mat &Logo_Pin_Flat_Mat () { return logo_pin_flat;}

	cv::Mat &Logo_Pin_Deep_Mat () { return logo_pin_deep;}

	cv::Mat &Logo_Wheel_Flat_Mat () { return logo_wheel_flat;}

	cv::Mat &Logo_Wheel_Deep_Mat () { return logo_wheel_deep;}

	cv::Mat &Logo_S_Flat_Mat () { return logo_s_flat;}

	cv::Mat &Logo_S_Deep_Mat () { return logo_s_deep;}

	cv::Mat &Logo_IDot_Flat_Mat () { return logo_idot_flat;}

	cv::Mat &Logo_IDot_Deep_Mat () { return logo_idot_deep;}

	cv::Mat &Logo_Pivot_Flat_Mat () { return logo_pivot_flat;}

	cv::Mat &Logo_Pivot_Deep_Mat () { return logo_pivot_deep;}

	cv::Mat &Tip_Flat_Mat () { return tip_flat;}

	cv::Mat &Tip_Deep_Mat () { return tip_deep;}

	cv::Mat &Tip_Mat () { return tip;}

	ToolType Type () { return mtype; }


protected:

private:
	ToolType mtype;

	// VTK stl reader
	vtkSmartPointer<vtkSTLReader> shaft_stl_reader;
	vtkSmartPointer<vtkSTLReader> logo_stl_reader;
	vtkSmartPointer<vtkSTLReader> jawL_stl_reader;
	vtkSmartPointer<vtkSTLReader> jawR_stl_reader;

	// VTK ply reader
	vtkSmartPointer<vtkPLYReader> shaft_ply_reader;
	vtkSmartPointer<vtkPLYReader> logo_ply_reader;
	vtkSmartPointer<vtkPLYReader> jawL_ply_reader;
	vtkSmartPointer<vtkPLYReader> jawR_ply_reader;

	// VTK mapper
	vtkSmartPointer<vtkPolyDataMapper> shaft_mapper;
	vtkSmartPointer<vtkPolyDataMapper> logo_mapper;
	vtkSmartPointer<vtkPolyDataMapper> jawL_mapper;
	vtkSmartPointer<vtkPolyDataMapper> jawR_mapper;

	// VTK actor
	vtkSmartPointer<vtkActor> shaft_actor;
	vtkSmartPointer<vtkActor> logo_actor;
	vtkSmartPointer<vtkActor> jawL_actor;
	vtkSmartPointer<vtkActor> jawR_actor;

	// VTK TriangleFilter (for fast rendering)
	vtkSmartPointer<vtkTriangleFilter> shaft_tris;
	vtkSmartPointer<vtkTriangleFilter> logo_tris;
	vtkSmartPointer<vtkTriangleFilter> jawL_tris;
	vtkSmartPointer<vtkTriangleFilter> jawR_tris;

	// VTK vtkStripper (for fast rendering)
	vtkSmartPointer<vtkStripper> shaft_strip;
	vtkSmartPointer<vtkStripper> logo_strip;
	vtkSmartPointer<vtkStripper> jawL_strip;
	vtkSmartPointer<vtkStripper> jawR_strip;

	vtkSmartPointer<vtkTransform> shaft_transform;
	vtkSmartPointer<vtkTransform> logo_transform;
	vtkSmartPointer<vtkTransform> jawL_transform;
	vtkSmartPointer<vtkTransform> jawR_transform;

	vtkSmartPointer<vtkTransformPolyDataFilter> shaft_transform_filter;
	vtkSmartPointer<vtkTransformPolyDataFilter> logo_transform_filter;
	vtkSmartPointer<vtkTransformPolyDataFilter> jaw_transform_filter;
	vtkSmartPointer<vtkTransformPolyDataFilter> jawL_transform_filter;
	vtkSmartPointer<vtkTransformPolyDataFilter> jawR_transform_filter;

	vtkSmartPointer<vtkAxesActor> shaft_axes;
	vtkSmartPointer<vtkAxesActor> logo_axes;
	vtkSmartPointer<vtkAxesActor> jaw_axes;

	// Pre-measured key point in local coordinate (in homo matrix)

	cv::Mat shaft_centre;
	cv::Mat shaft_pivot_flat; // In shaft coord
	cv::Mat shaft_pivot_deep;
	cv::Mat logo_pin_flat;    // In logo body coord
	cv::Mat logo_pin_deep;
	cv::Mat logo_wheel_flat;
	cv::Mat logo_wheel_deep;
	cv::Mat logo_s_flat;
	cv::Mat logo_s_deep;
	cv::Mat logo_idot_flat;
	cv::Mat logo_idot_deep;
	cv::Mat logo_pivot_flat;
	cv::Mat logo_pivot_deep;
	cv::Mat tip_flat; // According to shaft
	cv::Mat tip_deep; // According to shaft
	cv::Mat tip;      // Tip central
};





