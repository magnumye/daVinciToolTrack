#include "psm_tool.h"


PsmTool::PsmTool(ToolType type) :
	mtype (type)
{
}

void PsmTool::init(std::string mod_dir, std::string tool_config_dir)
{
	std::string prefix;
	switch (mtype)
	{
	case LND:
		prefix = "lnd";
		mod_dir = mod_dir + "LND/";
		break;
	case MCS:
		prefix = "mcs";
		mod_dir = mod_dir + "MCS/";
		break;
	case ProGrasp:
		prefix = "pgrasp";
		mod_dir = mod_dir + "PGrasp/";
		break;
	case CF:
		prefix = "cf";
		mod_dir = mod_dir + "CF/";
		break;
	case MBF:
		prefix = "mbf";
		mod_dir = mod_dir + "MBF/";
		break;
	case RTS:
		prefix = "rts";
		mod_dir = mod_dir + "RTS/";
		break;
	default:
		break;
	}

	/************************************************************************/
	/* Init Keypoint                                                        */
	/************************************************************************/
	shaft_centre = cv::Mat::eye(4, 4, CV_64FC1);
	shaft_pivot_flat = cv::Mat::eye(4, 4, CV_64FC1);
	shaft_pivot_deep = cv::Mat::eye(4, 4, CV_64FC1);
	logo_pin_flat = cv::Mat::eye(4, 4, CV_64FC1);
	logo_pin_deep = cv::Mat::eye(4, 4, CV_64FC1);
	logo_wheel_flat = cv::Mat::eye(4, 4, CV_64FC1);
	logo_wheel_deep = cv::Mat::eye(4, 4, CV_64FC1);
	logo_s_flat= cv::Mat::eye(4, 4, CV_64FC1);
	logo_s_deep = cv::Mat::eye(4, 4, CV_64FC1);
	logo_idot_flat = cv::Mat::eye(4, 4, CV_64FC1);
	logo_idot_deep = cv::Mat::eye(4, 4, CV_64FC1);
	logo_pivot_flat = cv::Mat::eye(4, 4, CV_64FC1);
	logo_pivot_deep = cv::Mat::eye(4, 4, CV_64FC1);
	tip_flat = cv::Mat::eye(4, 4, CV_64FC1);
	tip_deep = cv::Mat::eye(4, 4, CV_64FC1);
	tip = cv::Mat::eye(4, 4, CV_64FC1);

	cv::Point3d pt;
	cv::FileStorage fs_config(tool_config_dir, cv::FileStorage::READ);
	fs_config[prefix + "_shaft_centre"] >> pt;
	shaft_centre.at<double>(0, 3) = pt.x;
	shaft_centre.at<double>(1, 3) = pt.y;
	shaft_centre.at<double>(2, 3) = pt.z;
	fs_config[prefix + "_shaft_pivot_flat"] >> pt;
	shaft_pivot_flat.at<double>(0, 3) = pt.x;
	shaft_pivot_flat.at<double>(1, 3) = pt.y;
	shaft_pivot_flat.at<double>(2, 3) = pt.z;
	fs_config[prefix + "_shaft_pivot_deep"] >> pt;
	shaft_pivot_deep.at<double>(0, 3) = pt.x;
	shaft_pivot_deep.at<double>(1, 3) = pt.y;
	shaft_pivot_deep.at<double>(2, 3) = pt.z;
	fs_config[prefix + "_logo_pin_flat"] >> pt;
	logo_pin_flat.at<double>(0, 3) = pt.x;
	logo_pin_flat.at<double>(1, 3) = pt.y;
	logo_pin_flat.at<double>(2, 3) = pt.z;
	fs_config[prefix + "_logo_pin_deep"] >> pt;
	logo_pin_deep.at<double>(0, 3) = pt.x;
	logo_pin_deep.at<double>(1, 3) = pt.y;
	logo_pin_deep.at<double>(2, 3) = pt.z;
	fs_config[prefix + "_logo_wheel_flat"] >> pt;
	logo_wheel_flat.at<double>(0, 3) = pt.x;
	logo_wheel_flat.at<double>(1, 3) = pt.y;
	logo_wheel_flat.at<double>(2, 3) = pt.z;
	fs_config[prefix + "_logo_wheel_deep"] >> pt;
	logo_wheel_deep.at<double>(0, 3) = pt.x;
	logo_wheel_deep.at<double>(1, 3) = pt.y;
	logo_wheel_deep.at<double>(2, 3) = pt.z;
	fs_config[prefix + "_logo_s_flat"] >> pt;
	logo_s_flat.at<double>(0, 3) = pt.x;
	logo_s_flat.at<double>(1, 3) = pt.y;
	logo_s_flat.at<double>(2, 3) = pt.z;
	fs_config[prefix + "_logo_s_deep"] >> pt;
	logo_s_deep.at<double>(0, 3) = pt.x;
	logo_s_deep.at<double>(1, 3) = pt.y;
	logo_s_deep.at<double>(2, 3) = pt.z;
	fs_config[prefix + "_logo_idot_flat"] >> pt;
	logo_idot_flat.at<double>(0, 3) = pt.x;
	logo_idot_flat.at<double>(1, 3) = pt.y;
	logo_idot_flat.at<double>(2, 3) = pt.z;
	fs_config[prefix + "_logo_idot_deep"] >> pt;
	logo_idot_deep.at<double>(0, 3) = pt.x;
	logo_idot_deep.at<double>(1, 3) = pt.y;
	logo_idot_deep.at<double>(2, 3) = pt.z;
	fs_config[prefix + "_logo_pivot_flat"] >> pt;
	logo_pivot_flat.at<double>(0, 3) = pt.x;
	logo_pivot_flat.at<double>(1, 3) = pt.y;
	logo_pivot_flat.at<double>(2, 3) = pt.z;
	fs_config[prefix + "_logo_pivot_deep"] >> pt;
	logo_pivot_deep.at<double>(0, 3) = pt.x;
	logo_pivot_deep.at<double>(1, 3) = pt.y;
	logo_pivot_deep.at<double>(2, 3) = pt.z;
	fs_config[prefix + "_tip_flat"] >> pt;
	tip_flat.at<double>(0, 3) = pt.x;
	tip_flat.at<double>(1, 3) = pt.y;
	tip_flat.at<double>(2, 3) = pt.z;
	fs_config[prefix + "_tip_deep"] >> pt;
	tip_deep.at<double>(0, 3) = pt.x;
	tip_deep.at<double>(1, 3) = pt.y;
	tip_deep.at<double>(2, 3) = pt.z;
	fs_config[prefix + "_tip"] >> pt;
	tip.at<double>(0, 3) = pt.x;
	tip.at<double>(1, 3) = pt.y;
	tip.at<double>(2, 3) = pt.z;



	/************************************************************************/
	/* Load model and setup VTK pipeline                                    */
	/************************************************************************/
	std::string shaft_mod_name = "shaft.STL";
	std::string logo_mod_name = "logobody.STL";
	std::string jaw_mod_name = "gripper_coord.STL";
	std::string jawL_mod_name = "jawLeft.STL";
	std::string jawR_mod_name = "jawRight.STL";

	std::string shaft_ply_name = "shaft_color.ply";
	std::string logo_ply_name = "logobody_color.ply";
	std::string jawL_ply_name = "jawLeft_color.ply";
	std::string jawR_ply_name = "jawRight_color.ply";

	shaft_ply_reader = vtkSmartPointer<vtkPLYReader>::New();
	shaft_ply_reader->SetFileName((mod_dir+shaft_ply_name).c_str());
	shaft_ply_reader->Update();
	shaft_ply_reader->ReleaseDataFlagOn();

	shaft_tris = vtkSmartPointer<vtkTriangleFilter>::New();
	shaft_tris->SetInputConnection(shaft_ply_reader->GetOutputPort());
	shaft_strip = vtkSmartPointer<vtkStripper>::New();
	shaft_strip->SetInputConnection(shaft_tris->GetOutputPort());

	// Logo body
	logo_ply_reader = vtkSmartPointer<vtkPLYReader>::New();
	logo_ply_reader->SetFileName((mod_dir+logo_ply_name).c_str());
	logo_ply_reader->Update();
	logo_ply_reader->ReleaseDataFlagOn();

	logo_tris = vtkSmartPointer<vtkTriangleFilter>::New();
	logo_tris->SetInputConnection(logo_ply_reader->GetOutputPort());
	logo_strip = vtkSmartPointer<vtkStripper>::New();
	logo_strip->SetInputConnection(logo_tris->GetOutputPort());

	// Jaw Left
	jawL_ply_reader = vtkSmartPointer<vtkPLYReader>::New();
	jawL_ply_reader->SetFileName((mod_dir+jawL_ply_name).c_str());
	jawL_ply_reader->Update();
	jawL_ply_reader->ReleaseDataFlagOn();

	jawL_tris = vtkSmartPointer<vtkTriangleFilter>::New();
	jawL_tris->SetInputConnection(jawL_ply_reader->GetOutputPort());
	jawL_strip = vtkSmartPointer<vtkStripper>::New();
	jawL_strip->SetInputConnection(jawL_tris->GetOutputPort());

	// Jaw Right
	jawR_ply_reader = vtkSmartPointer<vtkPLYReader>::New();
	jawR_ply_reader->SetFileName((mod_dir+jawR_ply_name).c_str());
	jawR_ply_reader->Update();
	jawR_ply_reader->ReleaseDataFlagOn();

	jawR_tris = vtkSmartPointer<vtkTriangleFilter>::New();
	jawR_tris->SetInputConnection(jawR_ply_reader->GetOutputPort());
	jawR_strip = vtkSmartPointer<vtkStripper>::New();
	jawR_strip->SetInputConnection(jawR_tris->GetOutputPort());

	// Transform Filter
	shaft_transform = vtkSmartPointer<vtkTransform>::New();
	logo_transform = vtkSmartPointer<vtkTransform>::New();
	jawL_transform = vtkSmartPointer<vtkTransform>::New();
	jawR_transform = vtkSmartPointer<vtkTransform>::New();
	

	// M = A * M (A is the applied transform)
	shaft_transform->PreMultiply();
	logo_transform->PreMultiply();
	jawL_transform->PreMultiply();
	jawR_transform->PreMultiply();

	shaft_transform_filter = vtkSmartPointer<vtkTransformPolyDataFilter>::New();
	logo_transform_filter = vtkSmartPointer<vtkTransformPolyDataFilter>::New();
	jawL_transform_filter = vtkSmartPointer<vtkTransformPolyDataFilter>::New();
	jawR_transform_filter = vtkSmartPointer<vtkTransformPolyDataFilter>::New();

	shaft_transform_filter->SetInputConnection(shaft_strip->GetOutputPort());
	logo_transform_filter->SetInputConnection(logo_strip->GetOutputPort());
	jawL_transform_filter->SetInputConnection(jawL_strip->GetOutputPort());
	jawR_transform_filter->SetInputConnection(jawR_strip->GetOutputPort());

	shaft_transform_filter->SetTransform(shaft_transform);
	logo_transform_filter->SetTransform(logo_transform);
	jawL_transform_filter->SetTransform(jawL_transform);
	jawR_transform_filter->SetTransform(jawR_transform);

	shaft_transform_filter->Update();
	logo_transform_filter->Update();
	jawL_transform_filter->Update();
	jawR_transform_filter->Update();

	// Mapper
	shaft_mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
	logo_mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
	jawL_mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
	jawR_mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
	
	// May speed up for large number of polygons
	shaft_mapper->ImmediateModeRenderingOn();
	logo_mapper->ImmediateModeRenderingOn();
	jawL_mapper->ImmediateModeRenderingOn();
	jawR_mapper->ImmediateModeRenderingOn();

	shaft_mapper->SetInputConnection(shaft_transform_filter->GetOutputPort());
	logo_mapper->SetInputConnection(logo_transform_filter->GetOutputPort());
	jawL_mapper->SetInputConnection(jawL_transform_filter->GetOutputPort());
	jawR_mapper->SetInputConnection(jawR_transform_filter->GetOutputPort());

	// Actor
	shaft_actor = vtkSmartPointer<vtkActor>::New();
	logo_actor = vtkSmartPointer<vtkActor>::New();
	jawL_actor = vtkSmartPointer<vtkActor>::New();
	jawR_actor = vtkSmartPointer<vtkActor>::New();

	shaft_actor->SetMapper(shaft_mapper);
	logo_actor->SetMapper(logo_mapper);
	jawL_actor->SetMapper(jawL_mapper);
	jawR_actor->SetMapper(jawR_mapper);

	// Axis
	shaft_axes = vtkSmartPointer<vtkAxesActor>::New();
	logo_axes = vtkSmartPointer<vtkAxesActor>::New();
	jaw_axes = vtkSmartPointer<vtkAxesActor>::New();
	shaft_axes->SetTotalLength(15, 15, 15);
	logo_axes->SetTotalLength(10, 10, 10);
	jaw_axes->SetTotalLength(20, 20, 20);


	fs_config.release();
}

void PsmTool::Update_Shaft_Transform (const double *elements)
{
	shaft_transform->SetMatrix(elements);
	shaft_transform->Update();
}

void PsmTool::Update_Logo_Transform (const double *elements)
{
	logo_transform->SetMatrix(elements);
	logo_transform->Update();
}

void PsmTool::Update_JawL_Transform (const double *elements)
{
	jawL_transform->SetMatrix(elements);
	jawL_transform->Update();
}

void PsmTool::Update_JawR_Transform (const double *elements)
{
	jawR_transform->SetMatrix(elements);
	jawR_transform->Update();
}