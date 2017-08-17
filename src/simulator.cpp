#include "simulator.h"
#include "GraphicalUserInterface.h"

#include <vtkLight.h>
#include <vtkLightCollection.h>

#include <vtkTimerLog.h>

#include "ShapeContextPro.h"


#ifdef __linux__
// Framerate calculation
double time_to_double(timeval *t);
double time_diff(timeval *t1, timeval *t2);
#endif

#define DETECTSHAFT 
#define MULTITHREADING


Simulator::Simulator(std::string config_path) :
	psm1_file (NULL)
	, cam_file (NULL)
	, m_cam (NULL)
	, psm1_tool (NULL)
	, psm2_tool (NULL)
	, gui_(NULL)
	, is_live_(false)
	, is_recording_ (false)
	, show_sim (false)
{
	std::string psm1_type_str, psm2_type_str;
	cv::FileStorage fs_config;
	fs_config.open(config_path, cv::FileStorage::READ);
	fs_config["data_dir"] >> data_dir;
	fs_config["tool_config_dir"] >> tool_config_dir;
	fs_config["model_dir"] >> mod_dir;
	fs_config["cam_config_name"] >> cam_dir;
	fs_config["handeye_config_name"] >> handeye_dir;

	fs_config["video_name"] >> video_dir;
	fs_config["psm1_kin_name"] >> psm1_dir;
	fs_config["psm2_kin_name"] >> psm2_dir;
	fs_config["active_arm"] >> active_arm_;
	fs_config["psm1_type"] >> psm1_type_str;
	fs_config["psm2_type"] >> psm2_type_str;

	fs_config.release();

	cam_dir		= data_dir + cam_dir;
	handeye_dir = data_dir + handeye_dir;
	video_dir	= data_dir + video_dir;
	psm1_dir	= data_dir + psm1_dir;
	psm2_dir	= data_dir + psm2_dir;
	tool_config_dir = tool_config_dir;

	PsmTool::ToolType tool_type;
	if (!psm1_type_str.empty())
	{
		std::string prefix;
		if (psm1_type_str.compare("LND") == 0)
			tool_type = PsmTool::LND;

		else if (psm1_type_str.compare("MCS") == 0)
			tool_type = PsmTool::MCS;
		else if (psm1_type_str.compare("ProGrasp") == 0)
			tool_type = PsmTool::ProGrasp;
		else if (psm1_type_str.compare("CF") == 0)
			tool_type = PsmTool::CF;
		else if (psm1_type_str.compare("MBF") == 0)
			tool_type = PsmTool::MBF;
		else if (psm1_type_str.compare("RTS") == 0)
			tool_type = PsmTool::RTS;
		psm1_tool = new PsmTool(tool_type);
		psm1_tool->init(mod_dir, tool_config_dir);
	}

	if (!psm2_type_str.empty())
	{
		std::string prefix;
		if (psm2_type_str.compare("LND") == 0)
			tool_type = PsmTool::LND;

		else if (psm2_type_str.compare("MCS") == 0)
			tool_type = PsmTool::MCS;
		else if (psm2_type_str.compare("ProGrasp") == 0)
			tool_type = PsmTool::ProGrasp;
		else if (psm2_type_str.compare("CF") == 0)
			tool_type = PsmTool::CF;
		else if (psm2_type_str.compare("MBF") == 0)
			tool_type = PsmTool::MBF;
		else if (psm2_type_str.compare("RTS") == 0)
			tool_type = PsmTool::RTS;
		psm2_tool = new PsmTool(tool_type);
		psm2_tool->init(mod_dir, tool_config_dir);
	}


	// Init
	m_cam = new MonoCamera(cam_dir);

	init_LND_parameters();


	corr_T = cv::Mat::eye(4, 4, CV_64F);
	corr_T2 = cv::Mat::eye(4, 4, CV_64F);
	T_cam_set = false;
	T2_cam_set = false;
	T_cam_need_set = true;
	T2_cam_need_set = true;

	corr_T2_shared = cv::Mat::eye(4, 4, CV_64F);
	corr_T2_shared = cv::Mat::eye(4, 4, CV_64F);
	T_cam_set_shared = false;
	T2_cam_set_shared = false;

	psm1_shaft_slop_shared = 0;
	psm2_shaft_slop_shared = 0;

	// Load camera to robot base transform
	cv::FileStorage fs;
	cv::Mat bHc_psm1, bHc_psm2;
	fs.open(handeye_dir, cv::FileStorage::READ);
	fs["bHc_psm1"] >> bHc_psm1;
	fs["bHc_psm2"] >> bHc_psm2;
	if(!bHc_psm1.empty())
	{
		std::cout << "PSM1 Handeye" << std::endl;
		std::cout <<  bHc_psm1 << std::endl;


		m_cam->set_wHc(bHc_psm1); m_cam->set_wHc_corr(bHc_psm1);


	}
	if(!bHc_psm2.empty())
	{
		std::cout << "PSM2 Handeye" << std::endl;
		std::cout <<  bHc_psm2 << std::endl;


		m_cam->set_w2Hc(bHc_psm2); m_cam->set_w2Hc_corr(bHc_psm2);


		// IMPORTANT: transform psm2 to psm1 coordinate which is reference frame in VTK rendering
		cv::Mat b1Hb2_cv = bHc_psm1 * bHc_psm2.inv();
		::memcpy(b1Hb2.data(), b1Hb2_cv.data, 16 * sizeof(double));
	}
	fs.release();

	time_stamp = 0;

	data_ready = false;

	cam_proc = new CameraProcessor(m_cam->Fx(), m_cam->Fy(), m_cam->Px(), m_cam->Py(),  m_cam->Width(), m_cam->Height(),
		m_cam->wHc().inv(), m_cam->w2Hc().inv(), LND_partcolors, LND_partname2ids);
	cam_proc->set_simulator(this);
	// Connect timer to callback
	//QObject::connect(&pose_timer, &QTimer::timeout, this, &Simulator::process_dual_pose);
#ifdef MULTITHREADING
	QObject::connect(&pose_timer, &QTimer::timeout, this, &Simulator::virtual_rendering_thread);
#else
	QObject::connect(&pose_timer, &QTimer::timeout, this, &Simulator::process);
#endif

}

void Simulator::init_LND_parameters()
{
	LND_partnames.push_back("shaft_pivot_flat");LND_partnames.push_back("shaft_pivot_deep");
	LND_partnames.push_back("logo_pin_flat");LND_partnames.push_back("logo_pin_deep");
	LND_partnames.push_back("logo_wheel_flat");LND_partnames.push_back("logo_wheel_deep");
	LND_partnames.push_back("logo_is_flat");LND_partnames.push_back("logo_is_deep");
	LND_partnames.push_back("logo_idot_flat");LND_partnames.push_back("logo_idot_deep");
	LND_partnames.push_back("logo_pivot_flat");LND_partnames.push_back("logo_pivot_deep");
	LND_partnames.push_back("tip_flat");LND_partnames.push_back("tip_deep");
	LND_partcolors.push_back(cv::Scalar(0,255,255)); LND_partcolors.push_back(cv::Scalar(0,255,255)); // yellow shaft_pivot flat & deep
	LND_partcolors.push_back(cv::Scalar(0,153,255)); LND_partcolors.push_back(cv::Scalar(0,153,255)); // orange logo_pin flat & deep
	LND_partcolors.push_back(cv::Scalar(0,0,255)); LND_partcolors.push_back(cv::Scalar(0,0,255)); // red logo_wheel flat & deep
	LND_partcolors.push_back(cv::Scalar(255,255,0)); LND_partcolors.push_back(cv::Scalar(255,255,0)); // cyan logo_is flat & deep
	LND_partcolors.push_back(cv::Scalar(190,190,190)); LND_partcolors.push_back(cv::Scalar(190,190,190)); // gray logo_idot flat & deep
	LND_partcolors.push_back(cv::Scalar(255,0,0)); LND_partcolors.push_back(cv::Scalar(255,0,0)); // blue logo_pivot flat & deep
	LND_partcolors.push_back(cv::Scalar(255,0,255)); // magenta tip_flat
	LND_partcolors.push_back(cv::Scalar(0,255,0)); //green tip_deep
	LND_partcolors.push_back(cv::Scalar(255,255,150)); //light blue
	LND_partname2ids.insert(std::make_pair("shaft_pivot_flat", 0));
	LND_partname2ids.insert(std::make_pair("shaft_pivot_deep", 1));
	LND_partname2ids.insert(std::make_pair("logo_pin_flat", 2));
	LND_partname2ids.insert(std::make_pair("logo_pin_deep", 3));
	LND_partname2ids.insert(std::make_pair("logo_wheel_flat", 4));
	LND_partname2ids.insert(std::make_pair("logo_wheel_deep", 5));
	LND_partname2ids.insert(std::make_pair("logo_is_flat", 6));
	LND_partname2ids.insert(std::make_pair("logo_is_deep", 7));
	LND_partname2ids.insert(std::make_pair("logo_idot_flat", 8));
	LND_partname2ids.insert(std::make_pair("logo_idot_deep", 9));
	LND_partname2ids.insert(std::make_pair("logo_pivot_flat", 10));
	LND_partname2ids.insert(std::make_pair("logo_pivot_deep", 11));
	LND_partname2ids.insert(std::make_pair("tip_flat", 12));
	LND_partname2ids.insert(std::make_pair("tip_deep", 13));
	LND_partname2ids.insert(std::make_pair("shaft_centre", 14));
	LND_part_tolerances.resize(LND_partnames.size(), 1.0f);
	LND_part_tolerances[12] = 3.0f; LND_part_tolerances[13] = 3.0f;

}

Simulator::~Simulator()
{

#ifdef _WIN32
	//----------------------------
	//full_vid_wrt.release();
	//----------------------------
#endif
	if (psm1_file != NULL)
		delete psm1_file;

	if (cam_file != NULL)
		delete cam_file;
	if (m_cam != NULL)
		delete m_cam;
	if (cam_proc != NULL)
		delete cam_proc;

	if (this->RawZBuffer != NULL)
		delete[] this->RawZBuffer;
}

bool Simulator::init()
{

	for (char & c : active_arm_)
	{
		if (c == '1')
			psm1_file = new ReaderRobot(psm1_dir);
		if (c == '2')
			psm2_file = new ReaderRobot(psm2_dir);
	}
	cam_file = new ReaderCamera(video_dir, cam_dir);
#ifdef _WIN32
	//---------------------------------
	count_processed = 0;
	//full_vid_wrt.open("full_view.avi", CV_FOURCC('X','V','I','D'), 25, cv::Size(m_cam->Width()*2, m_cam->Height()));
	//---------------------------------
#endif

	renderer = vtkSmartPointer<vtkRenderer>::New();
	renderWin = vtkSmartPointer<vtkRenderWindow>::New();

	renderer->AddActor(psm1_tool->Shaft_Actor());
	renderer->AddActor(psm1_tool->Logo_Actor());
	renderer->AddActor(psm1_tool->JawL_Actor());
	renderer->AddActor(psm1_tool->JawR_Actor());
	renderer->AddActor(psm2_tool->Shaft_Actor());
	renderer->AddActor(psm2_tool->Logo_Actor());
	renderer->AddActor(psm2_tool->JawL_Actor());
	renderer->AddActor(psm2_tool->JawR_Actor());

	//    renderer->AddActor(origin_axes);
	//    renderer->AddActor(shaft_axes);
	//    renderer->AddActor(logo_axes);
	//    renderer->AddActor(jaw_axes);

	renderer->LightFollowCameraOn();
	//renderer->UseShadowsOn();
	m_cam->set_window_size(1000, 800);
	//m_cam->set_window_size(m_cam->Width(), m_cam->Height());
	renderer->SetActiveCamera(m_cam->SetupCameraForRender());

	//renderer->ResetCameraClippingRange();


	m_light = vtkSmartPointer<vtkLight>::New();
	m_light->SetLightTypeToCameraLight();
	//    m_ligh->SetPosition(lightPosition[0], lightPosition[1], lightPosition[2]);
	//    m_ligh->SetPositional(true); // required for vtkLightActor below
	m_light->SetConeAngle(10);
	m_light->SetIntensity(0.9);
	//    m_ligh->SetFocalPoint(lightFocalPoint[0], lightFocalPoint[1], lightFocalPoint[2]);
	m_light->SetDiffuseColor(1,1,1);
	m_light->SetAmbientColor(1,1,1);
	//    m_light->SetSpecularColor(0,0,1);

	renderer->AddLight(m_light);
	renderer->SetBackground(1,1,1); // Background color


	renderWin->PolygonSmoothingOff();
	renderWin->LineSmoothingOff();
	renderWin->PointSmoothingOff();
	renderWin->SetOffScreenRendering(1);
	renderWin->SetSize(m_cam->Win_Width(), m_cam->Win_Height());
	renderWin->AddRenderer(renderer);

	windowToImageFilter = vtkSmartPointer<vtkWindowToImageFilter>::New();
	windowToImageFilter->SetInput(renderWin);

	this->RawZBuffer = new float[m_cam->Win_Width() * m_cam->Win_Height()];

	pose_timer.start(38);

#ifdef MULTITHREADING
	cam_proc->start();
#endif

	return true;
}

void Simulator::setupGUI(GraphicalUserInterface *g)
{
	gui_ = g;

	// Connect to GUI
	QObject::connect(this,  SIGNAL(updateRenderImage(QImage)), gui_, SLOT(updateRenderImage(QImage)));
	QObject::connect(this,  SIGNAL(updateCameraImage(QImage)), gui_, SLOT(updateCameraImage(QImage)));
	QObject::connect(this, SIGNAL(dispFrameRate(QString)), gui_, SLOT(setInfo1(QString)));
	QObject::connect(cam_proc,  SIGNAL(updateCameraImage(QImage)), gui_, SLOT(updateCameraImage(QImage)));
	QObject::connect(cam_proc, SIGNAL(dispFrameRate(QString)), gui_, SLOT(setInfo2(QString)));
}

void Simulator::setup_record()
{

}

void Simulator::stop_record()
{

}

void Simulator::show_simulation(bool checked)
{
	show_sim = checked;
}
void Simulator::show_kinematics(bool checked)
{
	if (cam_proc != NULL)
		cam_proc->show_kinematics(checked);
}


void Simulator::process()
{
	virtual_rendering_thread();

	cam_proc->process();
	//camera_processing_thread();
}

bool Simulator::capture_camera(cv::Mat& img)
{

	cv::Mat stereo_img;

	cam_file->read_curr_image(img);

	if (img.empty())
		return false;


	return true;
}


void Simulator::virtual_rendering_thread()
{
	// Reset virtual camera given the corrected hand-eye
	////////////////////////
	// From global to local
	bool T_cam_set_local; //= T_cam_set_shared;
	bool T2_cam_set_local;// = T2_cam_set_shared;
	cv::Mat corr_T_local, corr_T2_local;
	//corr_T_shared.copyTo(corr_T_local);
	//corr_T2_shared.copyTo(corr_T2_local);
	ReadCameraOupput(T_cam_set_local, T2_cam_set_local, corr_T_local, corr_T2_local);
	////////////////////////

	if (T_cam_set_local && T_cam_need_set)
	{
		cv::Mat now_cHb = m_cam->wHc().inv();
		cv::Mat corr_cHb = corr_T_local * now_cHb;
		cv::Mat corr_wHc = corr_cHb.inv();

		cv::Mat now_cHb2 = m_cam->w2Hc_corr().inv();

		m_cam->set_wHc_corr(corr_wHc);

		cv::Mat b1Hb2_cv = corr_wHc * now_cHb2;

		//std::cout << "corr_T2: " << corr_T2 (cv::Rect(3, 0, 1, 3)) << std::endl;

		::memcpy(b1Hb2.data(), b1Hb2_cv.data, 16 * sizeof(double));

		T_cam_need_set = false;
	}
	if (T2_cam_set_local && T2_cam_need_set )
	{
		cv::Mat now_cHb = m_cam->wHc_corr().inv();
		cv::Mat corr_wHc = now_cHb.inv();

		cv::Mat now_cHb2 = m_cam->w2Hc().inv();
		cv::Mat corr_cHb2 = corr_T2_local * now_cHb2;
		cv::Mat corr_w2Hc = corr_cHb2.inv();

		m_cam->set_w2Hc_corr(corr_w2Hc);
		cv::Mat b1Hb2_cv = corr_wHc * corr_cHb2;

		//std::cout << "corr_T2: " << corr_T2 (cv::Rect(3, 0, 1, 3)) << std::endl;

		::memcpy(b1Hb2.data(), b1Hb2_cv.data, 16 * sizeof(double));
		T2_cam_need_set = false;
	}
	//////////////////////

	Eigen::Matrix<double, 4, 4, Eigen::RowMajor> psm1_bHe, psm1_bHj4, psm1_bHj5;
	Eigen::Matrix<double, 4, 4, Eigen::RowMajor> psm2_bHe, psm2_bHj4, psm2_bHj5;
	float psm1_jaw = 0.0, psm2_jaw = 0.0;

	// Process camera here to sync with robot pose
	// If no image is processed, do not process pose
	cv::Mat camera_image;
	if (!capture_camera(camera_image))
		return;

	for (char & c : active_arm_)
	{
		if (c == '1')
		{

			psm1_file->read_pose(psm1_bHj4, psm1_bHj5, psm1_bHe, psm1_jaw);

			psm1_tool->Update_Shaft_Transform(psm1_bHj4.data());
			psm1_tool->Update_Logo_Transform(psm1_bHj5.data());



			psm1_jaw = psm1_jaw < 0.0 ? 0.0 : psm1_jaw;

			Eigen::Matrix<double, 4, 4, Eigen::RowMajor> eHeL, eHeR, psm1_bHeL, psm1_bHeR;
			eHeL = Eigen::MatrixXd::Identity(4, 4);
			eHeR = Eigen::MatrixXd::Identity(4, 4);
			eHeL(1, 1) = cos(psm1_jaw/2); eHeL(1, 2) = -sin(psm1_jaw/2);
			eHeL(2, 1) = sin(psm1_jaw/2); eHeL(2, 2) = cos(psm1_jaw/2);

			eHeR(1, 1) = cos(psm1_jaw/2); eHeR(1, 2) = sin(psm1_jaw/2);
			eHeR(2, 1) = -sin(psm1_jaw/2); eHeR(2, 2) = cos(psm1_jaw/2);

			psm1_bHeL = psm1_bHe * eHeL;
			psm1_bHeR = psm1_bHe * eHeR;

			psm1_tool->Update_JawL_Transform(psm1_bHeL.data());
			psm1_tool->Update_JawR_Transform(psm1_bHeR.data());
		}
		if (c == '2')
		{

			psm2_file->read_pose(psm2_bHj4, psm2_bHj5, psm2_bHe, psm2_jaw);

			// IMPORTANT: transform psm2 to psm1 coordinate which is reference frame in VTK rendering
			Eigen::Matrix<double, 4, 4, Eigen::RowMajor> psm2_b1He, psm2_b1Hj4, psm2_b1Hj5;
			psm2_b1Hj4 = b1Hb2 * psm2_bHj4;
			psm2_b1Hj5 = b1Hb2 * psm2_bHj5;
			psm2_b1He = b1Hb2 * psm2_bHe;

			psm2_tool->Update_Shaft_Transform(psm2_b1Hj4.data());
			psm2_tool->Update_Logo_Transform(psm2_b1Hj5.data());

			psm2_jaw = psm2_jaw < 0.0 ? 0.0 : psm2_jaw;

			Eigen::Matrix<double, 4, 4, Eigen::RowMajor> eHeL, eHeR, psm2_bHeL, psm2_bHeR;
			eHeL = Eigen::MatrixXd::Identity(4, 4);
			eHeR = Eigen::MatrixXd::Identity(4, 4);
			eHeL(1, 1) = cos(psm2_jaw/2); eHeL(1, 2) = -sin(psm2_jaw/2);
			eHeL(2, 1) = sin(psm2_jaw/2); eHeL(2, 2) = cos(psm2_jaw/2);

			eHeR(1, 1) = cos(psm2_jaw/2); eHeR(1, 2) = sin(psm2_jaw/2);
			eHeR(2, 1) = -sin(psm2_jaw/2); eHeR(2, 2) = cos(psm2_jaw/2);

			psm2_bHeL = psm2_b1He * eHeL;
			psm2_bHeR = psm2_b1He * eHeR;

			psm2_tool->Update_JawL_Transform(psm2_bHeL.data());
			psm2_tool->Update_JawR_Transform(psm2_bHeR.data());
		}
	}

	if (!render_view(cloud_))
	{
		return;
	}


	//    vtkTimerLog *timer=vtkTimerLog::New();
	//    timer->StartTimer();

	//    timer->StopTimer();
	//        double time_0=timer->GetElapsedTime();
	//        std::cout<<"Render: "<<time_0<<" seconds."<<std::endl;
	//        timer->Delete();

	windowToImageFilter->Modified();    // Must have this to get updated rendered image
	windowToImageFilter->Update();


	render_img_vtk = windowToImageFilter->GetOutput();
	int dims[3];
	render_img_vtk->GetDimensions(dims);



	render_img_cv = cv::Mat(dims[1], dims[0], CV_8UC3, render_img_vtk->GetScalarPointer());

	cv::Mat render_img_flip_cv;
	cv::flip(render_img_cv, render_img_flip_cv, 0); // VTK's origin is at bottom-left
	cv::cvtColor(render_img_flip_cv, render_img_flip_cv, CV_RGB2BGR); // OpenCV uses BGR, VTK uses RGB

	std::vector<cv::Rect> psm1_part_boxes;
	std::vector<std::string> psm1_class_names;
	std::vector<cv::Point2f> psm1_projectedPoints, psm1_projectedKeypoints;
	std::vector<cv::Point3f> psm1_allKeypoints_no_corr;

	std::vector<bool> psm1_visible;
	cv::Point2f psm1_shaft_centre;
	bool psm1_shaft_centre_visible;
	calc_kpts_2D_sim(m_cam->wHc_corr().inv(), m_cam->wHc().inv(),psm1_bHj4, psm1_bHj5, psm1_bHe, psm1_jaw, psm1_projectedPoints,
		psm1_projectedKeypoints, psm1_visible, cloud_, psm1_allKeypoints_no_corr,
		psm1_shaft_centre_visible, psm1_shaft_centre, LND_ra_template_half_sizes, 1);

	std::vector<cv::Rect> psm2_part_boxes;
	std::vector<std::string> psm2_class_names;
	std::vector<cv::Point2f> psm2_projectedPoints, psm2_projectedKeypoints;
	std::vector<cv::Point3f> psm2_allKeypoints_no_corr;

	std::vector<bool> psm2_visible;
	cv::Point2f psm2_shaft_centre;
	bool psm2_shaft_centre_visible;
	calc_kpts_2D_sim(m_cam->w2Hc_corr().inv(), m_cam->w2Hc().inv(), psm2_bHj4, psm2_bHj5, psm2_bHe, psm2_jaw, psm2_projectedPoints,
		psm2_projectedKeypoints, psm2_visible, cloud_, psm2_allKeypoints_no_corr,
		psm2_shaft_centre_visible, psm2_shaft_centre, LND_la_template_half_sizes, 2);

	//////////// Perform QGO //////////////
	// 1. Collect part boxes and names, fix the box size to 60px (w/h) for now but
	// will be changed after QGO template extraction (saved in bounding_boxes).
	// PSM1
	psm1_part_boxes.reserve(psm1_projectedKeypoints.size());
	psm1_class_names.reserve(psm1_projectedKeypoints.size());

	for (int pb_id = 0; pb_id < psm1_projectedKeypoints.size(); pb_id++)
	{
		int half_size = LND_ra_template_half_sizes[pb_id], full_size = LND_ra_template_half_sizes[pb_id]*2;

		if (psm1_visible[pb_id])
		{
			cv::Rect bb, sbb;
			bb.x = psm1_projectedKeypoints[pb_id].x - half_size;
			bb.y = psm1_projectedKeypoints[pb_id].y - half_size;
			bb.width = full_size;
			bb.height = full_size;

			psm1_part_boxes.push_back(bb);
			psm1_class_names.push_back(LND_partnames[pb_id]);
		}
	}
#ifdef DETECTSHAFT
	if (psm1_shaft_centre_visible)
	{
		cv::Rect bb;
		bb.x = psm1_shaft_centre.x - LND_ra_template_half_sizes[14];
		bb.y = psm1_shaft_centre.y - LND_ra_template_half_sizes[14];
		bb.width = LND_ra_template_half_sizes[14]*2;
		bb.height = LND_ra_template_half_sizes[14]*2;
		psm1_part_boxes.push_back(bb);
		psm1_class_names.push_back("shaft_centre");
	}
	psm1_projectedKeypoints.push_back(psm1_shaft_centre);
#endif
	// PSM2
	psm2_part_boxes.reserve(psm2_projectedKeypoints.size());
	psm2_class_names.reserve(psm2_projectedKeypoints.size());

	for (int pb_id = 0; pb_id < psm2_projectedKeypoints.size(); pb_id++)
	{
		int half_size = LND_la_template_half_sizes[pb_id], full_size = LND_la_template_half_sizes[pb_id]*2;

		if (psm2_visible[pb_id])
		{
			cv::Rect bb, sbb;
			bb.x = psm2_projectedKeypoints[pb_id].x - half_size;
			bb.y = psm2_projectedKeypoints[pb_id].y - half_size;
			bb.width = full_size;
			bb.height = full_size;

			psm2_part_boxes.push_back(bb);
			psm2_class_names.push_back(LND_partnames[pb_id]);
		}
	}
#ifdef DETECTSHAFT
	if (psm2_shaft_centre_visible)
	{
		cv::Rect bb;
		bb.x = psm2_shaft_centre.x - LND_la_template_half_sizes[14];
		bb.y = psm2_shaft_centre.y - LND_la_template_half_sizes[14];
		bb.width = LND_la_template_half_sizes[14]*2;
		bb.height = LND_la_template_half_sizes[14]*2;
		psm2_part_boxes.push_back(bb);
		psm2_class_names.push_back("shaft_centre");
	}
	psm2_projectedKeypoints.push_back(psm2_shaft_centre);
#endif

	float psm1_slope = 0;
	calc_shaft_slope(m_cam->wHc_corr().inv(), psm1_bHj4, psm1_bHj5, psm1_bHe, psm1_jaw, psm1_slope);
	float psm2_slope = 0;
	calc_shaft_slope(m_cam->w2Hc_corr().inv(), psm2_bHj4, psm2_bHj5, psm2_bHe, psm2_jaw, psm2_slope);

	WriteVirtualRenderingOutput(render_img_flip_cv, camera_image,
		psm1_part_boxes,
		psm1_class_names, psm1_projectedKeypoints,
		psm1_allKeypoints_no_corr,
		LND_ra_template_half_sizes,
		psm2_part_boxes,
		psm2_class_names, psm2_projectedKeypoints,
		psm2_allKeypoints_no_corr,
		LND_la_template_half_sizes,
		psm1_bHe,
		psm1_bHj4,
		psm1_bHj5,
		psm2_bHe,
		psm2_bHj4,
		psm2_bHj5,
		psm1_jaw, psm2_jaw,
		psm1_slope, psm2_slope);



#ifdef DETECTSHAFT
	psm1_visible.push_back(psm1_shaft_centre_visible);
	psm2_visible.push_back(psm2_shaft_centre_visible);
#endif

	if (show_sim)
	{
		draw_pts(render_img_flip_cv, psm1_projectedPoints, psm1_projectedKeypoints, psm1_visible);
		draw_pts(render_img_flip_cv, psm2_projectedPoints, psm2_projectedKeypoints, psm2_visible);

		QImage qtemp = cvtCvMat2QImage(render_img_flip_cv);
		Q_EMIT updateRenderImage(qtemp);
	}


#ifdef __linux__    // Linux ROS specific
	// Framerate calculation
	timeval t_curr;
	gettimeofday(&t_curr, NULL);
	time_count.push_back(t_curr);
	if (time_count.size() > 50)
	{
		time_count.pop_front();
		double td = time_diff(&time_count.front(), &time_count.back());
		Q_EMIT dispFrameRate("Rendering FPS: " + QString::number(50.0/td));
	}
#endif
#ifdef _WIN32
	time_t t_curr;
	time(&t_curr);
	time_count.push_back(t_curr);
	if (time_count.size() > 100)
	{
		time_count.pop_front();
		double td = difftime(time_count.back(), time_count.front());
		Q_EMIT dispFrameRate("Rendering FPS: " + QString::number(100.0/td));
	}
#endif


}

void Simulator::WriteVirtualRenderingOutput(const cv::Mat& render_img_flip_cv, const cv::Mat& camera_image,
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
											float psm1_jaw, float psm2_jaw, float psm1_slope, float psm2_slope)
{

	//////////////////////////////
	// From local to global
	mx_virt_out.lock();

	render_img_flip_cv.copyTo(render_img_flip_cv_shared);
	camera_image.copyTo(camera_image_shared);

	psm1_part_boxes_shared.resize(psm1_part_boxes.size());
	psm1_class_names_shared.resize(psm1_class_names.size());
	std::copy(psm1_part_boxes.begin(), psm1_part_boxes.end(), psm1_part_boxes_shared.begin());
	std::copy(psm1_class_names.begin(), psm1_class_names.end(), psm1_class_names_shared.begin());

	psm2_part_boxes_shared.resize(psm2_part_boxes.size());
	psm2_class_names_shared.resize(psm2_class_names.size());
	std::copy(psm2_part_boxes.begin(), psm2_part_boxes.end(), psm2_part_boxes_shared.begin());
	std::copy(psm2_class_names.begin(), psm2_class_names.end(), psm2_class_names_shared.begin());

	psm1_allKeypoints_no_corr_shared.resize(psm1_allKeypoints_no_corr.size());
	std::copy(psm1_allKeypoints_no_corr.begin(), psm1_allKeypoints_no_corr.end(),
		psm1_allKeypoints_no_corr_shared.begin());

	psm2_allKeypoints_no_corr_shared.resize(psm2_allKeypoints_no_corr.size());
	std::copy(psm2_allKeypoints_no_corr.begin(), psm2_allKeypoints_no_corr.end(),
		psm2_allKeypoints_no_corr_shared.begin());

	psm1_projectedKeypoints_shared.resize(psm1_projectedKeypoints.size());
	std::copy(psm1_projectedKeypoints.begin(), psm1_projectedKeypoints.end(),
		psm1_projectedKeypoints_shared.begin());

	psm2_projectedKeypoints_shared.resize(psm2_projectedKeypoints.size());
	std::copy(psm2_projectedKeypoints.begin(), psm2_projectedKeypoints.end(),
		psm2_projectedKeypoints_shared.begin());

	psm1_bHj4_shared = psm1_bHj4;
	psm1_bHj5_shared = psm1_bHj5;
	psm1_bHe_shared = psm1_bHe;
	psm1_jaw_shared = psm1_jaw;

	psm2_bHj4_shared = psm2_bHj4;
	psm2_bHj5_shared = psm2_bHj5;
	psm2_bHe_shared = psm2_bHe;
	psm2_jaw_shared = psm2_jaw;

	LND_ra_template_half_sizes_shared.resize(psm1_template_half_sizes.size());
	std::copy(psm1_template_half_sizes.begin(), psm1_template_half_sizes.end(),
		LND_ra_template_half_sizes_shared.begin());

	LND_la_template_half_sizes_shared.resize(psm2_template_half_sizes.size());
	std::copy(psm2_template_half_sizes.begin(), psm2_template_half_sizes.end(),
		LND_la_template_half_sizes_shared.begin());

	psm1_shaft_slop_shared = psm1_slope;
	psm2_shaft_slop_shared = psm2_slope;

	data_ready = true;


	mx_virt_out.unlock();
	///////////////////////////////////
}


bool Simulator::ReadVirtualRenderingOutput(cv::Mat& render_img_flip_cv_local, cv::Mat& camera_image_local,
										   std::vector<cv::Rect>& psm1_part_boxes_local,
										   std::vector<std::string>& psm1_class_names_local, std::vector<cv::Point2f>& psm1_projectedKeypoints_local,
										   std::vector<cv::Point3f>& psm1_allKeypoints_no_corr_local,
										   std::vector<int>& psm1_template_half_sizes_local,
										   std::vector<cv::Rect>& psm2_part_boxes_local,
										   std::vector<std::string>& psm2_class_names_local, std::vector<cv::Point2f> & psm2_projectedKeypoints_local,
										   std::vector<cv::Point3f>& psm2_allKeypoints_no_corr_local,
										   std::vector<int>& psm2_template_half_sizes_local,
										   Eigen::Matrix<double, 4, 4, Eigen::RowMajor>& psm1_bHe_local,
										   Eigen::Matrix<double, 4, 4, Eigen::RowMajor>& psm1_bHj4_local,
										   Eigen::Matrix<double, 4, 4, Eigen::RowMajor>& psm1_bHj5_local,
										   Eigen::Matrix<double, 4, 4, Eigen::RowMajor>& psm2_bHe_local,
										   Eigen::Matrix<double, 4, 4, Eigen::RowMajor>& psm2_bHj4_local,
										   Eigen::Matrix<double, 4, 4, Eigen::RowMajor>& psm2_bHj5_local,
										   float& psm1_jaw_local, float& psm2_jaw_local,
										   float& psm1_slope_local, float& psm2_slope_local)
{
	///////////////////////////////////
	// From global to local
	mx_virt_out.lock();

	//std::cout<< data_ready << std::endl;
	if (!data_ready)
	{
		mx_virt_out.unlock();
		return false;
	}
	render_img_flip_cv_shared.copyTo(render_img_flip_cv_local);
	camera_image_shared.copyTo(camera_image_local);
	psm1_bHe_local = psm1_bHe_shared; psm1_bHj4_local = psm1_bHj4_shared; psm1_bHj5_local = psm1_bHj5_shared;
	psm2_bHe_local = psm2_bHe_shared; psm2_bHj4_local = psm2_bHj4_shared; psm2_bHj5_local = psm2_bHj5_shared;
	psm1_jaw_local = psm1_jaw_shared; psm2_jaw_local = psm2_jaw_shared;

	psm1_part_boxes_local.resize(psm1_part_boxes_shared.size());
	psm1_class_names_local.resize(psm1_class_names_shared.size());
	std::copy(psm1_part_boxes_shared.begin(), psm1_part_boxes_shared.end(), psm1_part_boxes_local.begin());
	std::copy(psm1_class_names_shared.begin(), psm1_class_names_shared.end(), psm1_class_names_local.begin());

	psm2_part_boxes_local.resize(psm2_part_boxes_shared.size());
	psm2_class_names_local.resize(psm2_class_names_shared.size());
	std::copy(psm2_part_boxes_shared.begin(), psm2_part_boxes_shared.end(), psm2_part_boxes_local.begin());
	std::copy(psm2_class_names_shared.begin(), psm2_class_names_shared.end(), psm2_class_names_local.begin());

	psm1_allKeypoints_no_corr_local.resize(psm1_allKeypoints_no_corr_shared.size());
	std::copy(psm1_allKeypoints_no_corr_shared.begin(), psm1_allKeypoints_no_corr_shared.end(),
		psm1_allKeypoints_no_corr_local.begin());

	psm2_allKeypoints_no_corr_local.resize(psm2_allKeypoints_no_corr_shared.size());
	std::copy(psm2_allKeypoints_no_corr_shared.begin(), psm2_allKeypoints_no_corr_shared.end(),
		psm2_allKeypoints_no_corr_local.begin());

	psm1_projectedKeypoints_local.resize(psm1_projectedKeypoints_shared.size());
	std::copy(psm1_projectedKeypoints_shared.begin(), psm1_projectedKeypoints_shared.end(),
		psm1_projectedKeypoints_local.begin());

	psm2_projectedKeypoints_local.resize(psm2_projectedKeypoints_shared.size());
	std::copy(psm2_projectedKeypoints_shared.begin(), psm2_projectedKeypoints_shared.end(),
		psm2_projectedKeypoints_local.begin());

	psm1_template_half_sizes_local.resize(LND_ra_template_half_sizes_shared.size());
	std::copy(LND_ra_template_half_sizes_shared.begin(), LND_ra_template_half_sizes_shared.end(),
		psm1_template_half_sizes_local.begin());

	psm2_template_half_sizes_local.resize(LND_la_template_half_sizes_shared.size());
	std::copy(LND_la_template_half_sizes_shared.begin(), LND_la_template_half_sizes_shared.end(),
		psm2_template_half_sizes_local.begin());

	psm1_slope_local = psm1_shaft_slop_shared;
	psm2_slope_local = psm2_shaft_slop_shared;

	mx_virt_out.unlock();
	///////////////////////////////////
	return true;
}

void Simulator::WriteCameraOutput(bool T_cam_set, bool T2_cam_set, const cv::Mat& corr_T, const cv::Mat& corr_T2)
{
	///////////////////////////////////
	// From local to global
	mx_cam_out.lock();
	T_cam_set_shared = T_cam_set;
	T2_cam_set_shared = T2_cam_set;
	corr_T.copyTo(corr_T_shared);
	corr_T2.copyTo(corr_T2_shared);
	mx_cam_out.unlock();
	///////////////////////////////////
}

void Simulator::ReadCameraOupput(bool& T_cam_set_local, bool& T2_cam_set_local, cv::Mat& corr_T_local, cv::Mat& corr_T2_local)
{
	///////////////////////////////////
	// From global to local
	mx_cam_out.lock();

	T_cam_set_local = T_cam_set_shared;
	T2_cam_set_local = T2_cam_set_shared;
	corr_T_shared.copyTo(corr_T_local);
	corr_T2_shared.copyTo(corr_T2_local);

	mx_cam_out.unlock();
	////////////////////////
}


bool Simulator::detect_shaft(cv::Mat &camera_image, const cv::Mat &binary_mask, const cv::Rect &roi)
{

	std::vector<std::vector<cv::Point>> contours;
	//std::vector<std::vector<cv::Vec4i>> hierarchy;
	//cv::Mat roi_mat = binary_mask(roi);
	cv::findContours(binary_mask(roi), contours, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
	int largest_area=0;
	int largest_contour_index=0;
	for ( int i = 0; i < contours.size(); i++ ) // iterate through each contour.
	{

		double a=cv::contourArea(contours[i], false);  //  Find the area of contour
		if (a > largest_area)
		{
			largest_area=a;
			largest_contour_index=i;                //Store the index of largest contour
			// bounding_rect=boundingRect(contours[i]); // Find the bounding rectangle for biggest contour
		}
	}

	//cv::Mat binary_mask_show;
	//cv::cvtColor(binary_mask, binary_mask_show, CV_GRAY2BGR);
	//binary_mask_show(roi).copyTo(camera_image(roi));
	if (contours.size() > 0 && largest_area > 1600)
	{
		cv::Mat temp = camera_image(roi);
		get_orientation(contours[largest_contour_index], temp);
		return true;
	}
	else
	{
		return false;
	}
}

cv::Point3f Simulator::get_3D(const cv::Point &pt)
{
	// Transform cloud to give camera coordinates instead of world coordinates!
	vtkCamera *camera = renderer->GetActiveCamera ();
	vtkSmartPointer<vtkMatrix4x4> composite_projection_transform = camera->GetCompositeProjectionTransformMatrix (renderer->GetTiledAspectRatio (), 0, 1);
	vtkSmartPointer<vtkMatrix4x4> view_transform = camera->GetViewTransformMatrix ();

	cv::Mat comp_proj_transform_cv(4, 4, CV_32FC1), view_transform_cv(4, 4, CV_32FC1);

	for (int i = 0; i < 4; ++i)
	{
		for (int j = 0; j < 4; ++j)
		{
			comp_proj_transform_cv.at<float>(i, j) = static_cast<float> (composite_projection_transform->Element[i][j]);
			view_transform_cv.at<float>(i, j) = static_cast<float> (view_transform->Element[i][j]);
		}
	}

	comp_proj_transform_cv = comp_proj_transform_cv.inv();
	cv::Mat camera2world_transform = view_transform_cv * comp_proj_transform_cv;
	cv::Mat camera_coords = cv::Mat::ones(4, 1, CV_32FC1);

	int xres = m_cam->Win_Width(), yres = m_cam->Win_Height();
	float dwidth = 2.0f / float (xres),
		dheight = 2.0f / float (yres);

	// NB: Convert to OpenCV coordinate, VTK's origin is at bottom-left
	cv::Point pt_vtk (pt.x, yres - (pt.y + 1));
	int index = pt_vtk.y * xres + pt_vtk.x;

	cv::Point3f pt_3d;
	if (this->RawZBuffer[index] == 1.0)
	{
		pt_3d.x = pt_3d.y = pt_3d.z = std::numeric_limits<float>::quiet_NaN ();
	}
	else
	{
		camera_coords.at<float>(0, 0) = dwidth  * float (pt_vtk.x) - 1.0f;
		camera_coords.at<float>(1, 0) = dheight * float (pt_vtk.y) - 1.0f;
		camera_coords.at<float>(2, 0) = this->RawZBuffer[index];

		cv::Mat world_coords = camera2world_transform * camera_coords;

		float w3 = 1.0f / world_coords.at<float>(3);

		// vtk view coordinate system is different than the standard camera coordinates (z forward, y down, x right), thus, the fliping in y and z
		pt_3d.x = world_coords.at<float>(0, 0) * w3;
		pt_3d.y = - world_coords.at<float>(1, 0) * w3;
		pt_3d.z = - world_coords.at<float>(2, 0) * w3;
	}


	return pt_3d;
}


bool Simulator::render_view(cv::Mat &cloud)
{
	// Reference from:
	// https://github.com/PointCloudLibrary/pcl/blob/master/visualization/src/pcl_visualizer.cpp#L3852

	renderWin->Render ();

	int xres = m_cam->Win_Width(), yres = m_cam->Win_Height();

	float dwidth = 2.0f / float (xres),
		dheight = 2.0f / float (yres);

	cloud.create(yres, xres, CV_32FC3);

	//    float *depth = new float[xres * yres];
	renderWin->GetZbufferData (0, 0, xres - 1, yres - 1, &(this->RawZBuffer[0]));

	return true;
}

void Simulator::check_parts_visible(const cv::Mat &depth_map, const std::vector<cv::Point3f> &keypoints,
									const std::vector<cv::Point2f>& projectedKeypoints, std::vector<bool> &visibilities,
									const std::vector<int>& template_half_sizes)
{



	visibilities.resize(keypoints.size(), false);
	for (int i = 0; i < visibilities.size(); i++)
	{
		int border = template_half_sizes[i];
		int width_limit = depth_map.cols - border;
		int height_limit = depth_map.rows - border;

		int x = (int)projectedKeypoints[i].x;
		int y = (int)projectedKeypoints[i].y;
		if (x < border || x > width_limit || y < border || y > height_limit)
		{
			continue;
		}

		const cv::Vec3f &pt = depth_map.at<cv::Vec3f>(y, x);

		// Check this keypoint is visible or not. ( ~1mm tolerance)
		if ((pt[2] + LND_part_tolerances[i]) < keypoints[i].z )
		{
			continue;
		}
		visibilities[i] = true;
	}
}

void Simulator::check_parts_visible(const std::vector<cv::Point3f> &keypoints,
									const std::vector<cv::Point2f>& projectedKeypoints, std::vector<bool> &visibilities,
									const std::vector<int>& template_half_sizes)
{



	visibilities.resize(keypoints.size(), false);
	for (int i = 0; i < visibilities.size(); i++)
	{
		int border = template_half_sizes[i];
		int width_limit = m_cam->Win_Width() - border;
		int height_limit = m_cam->Win_Height() - border;

		int x = (int)projectedKeypoints[i].x;
		int y = (int)projectedKeypoints[i].y;
		if (x < border || x > width_limit || y < border || y > height_limit)
		{
			continue;
		}

		const cv::Point3f &pt = get_3D(cv::Point(x,y));//depth_map.at<cv::Vec3f>(y, x);

		// Check this keypoint is visible or not. ( ~1mm tolerance)
		if ((pt.z + LND_part_tolerances[i]) < keypoints[i].z )
		{
			continue;
		}
		visibilities[i] = true;
	}
}

void Simulator::calc_kpts_2D_sim(const cv::Mat& camHbase_psm, const cv::Mat& camHbase_psm_no_corr,
								 Eigen::Matrix<double, 4, 4, Eigen::RowMajor> &psm_Hj4,
								 Eigen::Matrix<double, 4, 4, Eigen::RowMajor> &psm_Hj5,
								 Eigen::Matrix<double, 4, 4, Eigen::RowMajor> &psm_He,
								 float jaw_in, std::vector<cv::Point2f>& projectedPoints,
								 std::vector<cv::Point2f>& projectedKeypoints,
								 std::vector<bool> &visibilities, const cv::Mat& depth_map,
								 std::vector<cv::Point3f>& cHpkey, bool& sc_visible,
								 cv::Point2f& shaft_centre, std::vector<int>& template_half_sizes, int psm_num)
{
	cv::Mat subP = cv::Mat::eye(3, 3, CV_32FC1);
	subP.at<float>(0,0) = m_cam->Fx(); subP.at<float>(1,1) = m_cam->Fy();
	subP.at<float>(0,2) = m_cam->Px_win(); subP.at<float>(1,2) = m_cam->Py_win();


	std::vector<cv::Mat> bHj_vec;
	std::vector<cv::Point3f> cPj(7); // 1 shaft_far, 3 joint, 3 tips (cen/l/r)

	for (int i = 0; i < 3; i++)
		bHj_vec.push_back(cv::Mat(4, 4, CV_64F));


	::memcpy(bHj_vec[0].data, psm_Hj4.data(), sizeof(double)*16);
	::memcpy(bHj_vec[1].data, psm_Hj5.data(), sizeof(double)*16);
	::memcpy(bHj_vec[2].data, psm_He.data(), sizeof(double)*16);

	for (int i = 1; i <= bHj_vec.size(); i++)
	{
		// NB: bHj_vec unit in mm
		cv::Mat crHj_curr = camHbase_psm * bHj_vec[i-1];
		cPj[i].x = crHj_curr.at<double>(0, 3);
		cPj[i].y = crHj_curr.at<double>(1, 3);
		cPj[i].z = abs(crHj_curr.at<double>(2, 3));    // abs to ensure projection make sense
	}

	// NB: Shaft line
	cv::Mat j2H_offset = cv::Mat::eye(4,4,CV_64FC1);
	j2H_offset.at<double>(0, 3) = 50;  // 200mm
	cv::Mat temp = camHbase_psm * bHj_vec[0] * j2H_offset;
	cPj[0].x = temp.at<double>(0, 3);
	cPj[0].y = temp.at<double>(1, 3);
	cPj[0].z = abs(temp.at<double>(2, 3));    // abs to ensure projection make sense

	// Apply tip offset
	cv::Mat crHt;
	if (psm_num == 1)
		crHt = camHbase_psm * bHj_vec.back() * psm1_tool->Tip_Mat();
	else if (psm_num == 2)
		crHt = camHbase_psm * bHj_vec.back() * psm2_tool->Tip_Mat();

	cPj[4].x = crHt.at<double>(0, 3);
	cPj[4].y = crHt.at<double>(1, 3);
	cPj[4].z = crHt.at<double>(2, 3);

	// Calc keypoints
	cHpkey = calc_keypoints(bHj_vec[0], bHj_vec[1], bHj_vec[2], jaw_in, camHbase_psm, psm_num);


	cPj[5] = cHpkey[12];
	cPj[6] = cHpkey[13];

	// cPj[0]: shaft_far end
	// cPj[1]: shaft
	// cPj[2]: logo
	// cPj[3]: end-effector
	// cPj[4]: tip central
	// cPj[5]: tip left
	// cPj[6]: tip right

	cv::Mat rVec, tVec, distCoeffs;
	rVec = cv::Mat::zeros(3,1,CV_32FC1); tVec = cv::Mat::zeros(3,1,CV_32FC1);
	cv::projectPoints(cPj, rVec, tVec, subP, distCoeffs, projectedPoints);
	cv::projectPoints(cHpkey, rVec, tVec, subP, distCoeffs, projectedKeypoints);

	template_half_sizes.resize(cHpkey.size()+1, 30);
	//std::cout << cHpkey[13].z << ' ' << cHpkey[1].z << std::endl;
	for (int i = 0; i < cHpkey.size(); i++)
	{
		template_half_sizes[i] = std::max(30, cvRound(30 + 2*(80-cHpkey[i].z)));
	}
	//check_parts_visible(depth_map, cHpkey, projectedKeypoints, visibilities, template_half_sizes);
	check_parts_visible(cHpkey, projectedKeypoints, visibilities, template_half_sizes);

	// Calc shaft centre
	std::vector<cv::Point3f> sc = calc_extrapoints(bHj_vec[0], camHbase_psm, psm_num);
	std::vector<cv::Point2f> sc_2d;
	cv::projectPoints(sc, rVec, tVec, subP, distCoeffs, sc_2d);
	shaft_centre = sc_2d[0];
	template_half_sizes[cHpkey.size()] = std::max(60, cvRound(60 + 2*(70-sc[0].z)));
	int border = template_half_sizes[cHpkey.size()]; //Assuming the final one is the shaft centre.
	int width_limit = m_cam->Win_Width() - border;
	int height_limit = m_cam->Win_Height() - border;
	sc_visible = true;
	if (shaft_centre.x < border || shaft_centre.x > width_limit
		|| shaft_centre.y < border || shaft_centre.y > height_limit)
	{
		sc_visible = false;
	}

	// Calc keypoints in the orignal, without error correction. Important!
	cHpkey = calc_keypoints(bHj_vec[0], bHj_vec[1], bHj_vec[2], jaw_in, camHbase_psm_no_corr, psm_num);
#ifdef DETECTSHAFT
	sc = calc_extrapoints(bHj_vec[0], camHbase_psm_no_corr, psm_num);
	cHpkey.push_back(sc[0]);
#endif
}

void Simulator::draw_pts(cv::Mat &img, const std::vector<cv::Point2f>& projectedPoints, 
						 const std::vector<cv::Point2f>& projectedKeypoints,
						 const std::vector<bool>& visibilities)
{
	for (int i = 0; i < visibilities.size(); i++)
	{
		if (visibilities[i])
		{
			cv::circle(img, projectedKeypoints[i], 5, LND_partcolors[i], -1, CV_AA);
		}
	}
}


std::vector<cv::Point3f> Simulator::calc_keypoints(const cv::Mat &bHshaft,
												   const cv::Mat &bHlogo,
												   const cv::Mat &bHe,
												   const float jaw_ang,
												   const cv::Mat &cHb,
												   const int psm_num)
{
	/* Draw for extra points */
	cv::Mat cH_shaft_pivot_flat, cH_shaft_pivot_deep,
		cH_logo_pin_flat, cH_logo_pin_deep,
		cH_logo_wheel_flat, cH_logo_wheel_deep,
		cH_logo_s_flat, cH_logo_s_deep,
		cH_logo_idot_flat, cH_logo_idot_deep,
		cH_logo_pivot_flat, cH_logo_pivot_deep,
		cH_tip_left, cH_tip_right;
	if (psm_num == 1)
	{

		cH_shaft_pivot_flat = cHb * bHshaft * psm1_tool->Shaft_Pivot_Flat_Mat();
		cH_shaft_pivot_deep = cHb * bHshaft * psm1_tool->Shaft_Pivot_Deep_Mat();
		cH_logo_pin_flat = cHb * bHlogo * psm1_tool->Logo_Pin_Flat_Mat();
		cH_logo_pin_deep = cHb * bHlogo * psm1_tool->Logo_Pin_Deep_Mat();
		cH_logo_wheel_flat = cHb * bHlogo * psm1_tool->Logo_Wheel_Flat_Mat();
		cH_logo_wheel_deep = cHb * bHlogo * psm1_tool->Logo_Wheel_Deep_Mat();
		cH_logo_s_flat = cHb * bHlogo * psm1_tool->Logo_S_Flat_Mat();
		cH_logo_s_deep = cHb * bHlogo * psm1_tool->Logo_S_Deep_Mat();
		cH_logo_idot_flat = cHb * bHlogo * psm1_tool->Logo_IDot_Flat_Mat();
		cH_logo_idot_deep = cHb * bHlogo * psm1_tool->Logo_IDot_Deep_Mat();
		cH_logo_pivot_flat = cHb * bHlogo * psm1_tool->Logo_Pivot_Flat_Mat();
		cH_logo_pivot_deep = cHb * bHlogo * psm1_tool->Logo_Pivot_Deep_Mat();
	}
	else if (psm_num == 2)
	{
		cH_shaft_pivot_flat = cHb * bHshaft * psm2_tool->Shaft_Pivot_Flat_Mat();
		cH_shaft_pivot_deep = cHb * bHshaft * psm2_tool->Shaft_Pivot_Deep_Mat();
		cH_logo_pin_flat = cHb * bHlogo * psm2_tool->Logo_Pin_Flat_Mat();
		cH_logo_pin_deep = cHb * bHlogo * psm2_tool->Logo_Pin_Deep_Mat();
		cH_logo_wheel_flat = cHb * bHlogo * psm2_tool->Logo_Wheel_Flat_Mat();
		cH_logo_wheel_deep = cHb * bHlogo * psm2_tool->Logo_Wheel_Deep_Mat();
		cH_logo_s_flat = cHb * bHlogo * psm2_tool->Logo_S_Flat_Mat();
		cH_logo_s_deep = cHb * bHlogo * psm2_tool->Logo_S_Deep_Mat();
		cH_logo_idot_flat = cHb * bHlogo * psm2_tool->Logo_IDot_Flat_Mat();
		cH_logo_idot_deep = cHb * bHlogo * psm2_tool->Logo_IDot_Deep_Mat();
		cH_logo_pivot_flat = cHb * bHlogo * psm2_tool->Logo_Pivot_Flat_Mat();
		cH_logo_pivot_deep = cHb * bHlogo * psm2_tool->Logo_Pivot_Deep_Mat();
	}


	std::vector<cv::Point3f> cPp_vec;
	cPp_vec.push_back(cv::Point3f(cH_shaft_pivot_flat.at<double>(0, 3),
		cH_shaft_pivot_flat.at<double>(1, 3),
		abs(cH_shaft_pivot_flat.at<double>(2, 3))));
	cPp_vec.push_back(cv::Point3f(cH_shaft_pivot_deep.at<double>(0, 3),
		cH_shaft_pivot_deep.at<double>(1, 3),
		abs(cH_shaft_pivot_deep.at<double>(2, 3))));
	cPp_vec.push_back(cv::Point3f(cH_logo_pin_flat.at<double>(0, 3),
		cH_logo_pin_flat.at<double>(1, 3),
		abs(cH_logo_pin_flat.at<double>(2, 3))));
	cPp_vec.push_back(cv::Point3f(cH_logo_pin_deep.at<double>(0, 3),
		cH_logo_pin_deep.at<double>(1, 3),
		abs(cH_logo_pin_deep.at<double>(2, 3))));
	cPp_vec.push_back(cv::Point3f(cH_logo_wheel_flat.at<double>(0, 3),
		cH_logo_wheel_flat.at<double>(1, 3),
		abs(cH_logo_wheel_flat.at<double>(2, 3))));
	cPp_vec.push_back(cv::Point3f(cH_logo_wheel_deep.at<double>(0, 3),
		cH_logo_wheel_deep.at<double>(1, 3),
		abs(cH_logo_wheel_deep.at<double>(2, 3))));
	cPp_vec.push_back(cv::Point3f(cH_logo_s_flat.at<double>(0, 3),
		cH_logo_s_flat.at<double>(1, 3),
		abs(cH_logo_s_flat.at<double>(2, 3))));
	cPp_vec.push_back(cv::Point3f(cH_logo_s_deep.at<double>(0, 3),
		cH_logo_s_deep.at<double>(1, 3),
		abs(cH_logo_s_deep.at<double>(2, 3))));
	cPp_vec.push_back(cv::Point3f(cH_logo_idot_flat.at<double>(0, 3),
		cH_logo_idot_flat.at<double>(1, 3),
		abs(cH_logo_idot_flat.at<double>(2, 3))));
	cPp_vec.push_back(cv::Point3f(cH_logo_idot_deep.at<double>(0, 3),
		cH_logo_idot_deep.at<double>(1, 3),
		abs(cH_logo_idot_deep.at<double>(2, 3))));
	cPp_vec.push_back(cv::Point3f(cH_logo_pivot_flat.at<double>(0, 3),
		cH_logo_pivot_flat.at<double>(1, 3),
		abs(cH_logo_pivot_flat.at<double>(2, 3))));
	cPp_vec.push_back(cv::Point3f(cH_logo_pivot_deep.at<double>(0, 3),
		cH_logo_pivot_deep.at<double>(1, 3),
		abs(cH_logo_pivot_deep.at<double>(2, 3))));

	// Two tips
	float psm_jaw = jaw_ang < 0.0 ? 0.0 : jaw_ang;

	cv::Mat eHeL, eHeR;
	eHeL = cv::Mat::eye(4, 4, CV_64FC1);
	eHeR = cv::Mat::eye(4, 4, CV_64FC1);
	eHeL.at<double>(1, 1) = cos(psm_jaw/2); eHeL.at<double>(1, 2) = -sin(psm_jaw/2);
	eHeL.at<double>(2, 1) = sin(psm_jaw/2); eHeL.at<double>(2, 2) = cos(psm_jaw/2);
	eHeR.at<double>(1, 1) = cos(psm_jaw/2); eHeR.at<double>(1, 2) = sin(psm_jaw/2);
	eHeR.at<double>(2, 1) = -sin(psm_jaw/2); eHeR.at<double>(2, 2) = cos(psm_jaw/2);

	if (psm_num == 1)
	{
		cH_tip_left = cHb * bHe * eHeL * psm1_tool->Tip_Deep_Mat();
		cH_tip_right = cHb * bHe * eHeR * psm1_tool->Tip_Flat_Mat();
	}
	else if (psm_num == 2)
	{
		cH_tip_left = cHb * bHe * eHeL * psm2_tool->Tip_Deep_Mat();
		cH_tip_right = cHb * bHe * eHeR * psm2_tool->Tip_Flat_Mat();
	}

	// right is flat, left is deep
	cPp_vec.push_back(cv::Point3f(cH_tip_right.at<double>(0, 3),
		cH_tip_right.at<double>(1, 3),
		abs(cH_tip_right.at<double>(2, 3))));
	cPp_vec.push_back(cv::Point3f(cH_tip_left.at<double>(0, 3),
		cH_tip_left.at<double>(1, 3),
		abs(cH_tip_left.at<double>(2, 3))));
	return cPp_vec;
}

std::vector<cv::Point3f> Simulator::calc_extrapoints(const cv::Mat &bHshaft,
													 const cv::Mat &cHb,
													 const int psm_num)
{
	cv::Mat cH_shaft_centre;
	if (psm_num == 1)
	{
		cH_shaft_centre = cHb * bHshaft * psm1_tool->Shaft_Centre_Mat();
	}
	else if (psm_num == 2)
	{
		cH_shaft_centre = cHb * bHshaft * psm2_tool->Shaft_Centre_Mat();
	}

	// Shaft centre point
	std::vector<cv::Point3f> cPp_vec;
	cPp_vec.push_back(cv::Point3f(cH_shaft_centre.at<double>(0, 3),
		cH_shaft_centre.at<double>(1, 3),
		abs(cH_shaft_centre.at<double>(2, 3))));

	return cPp_vec;
}

QImage Simulator::cvtCvMat2QImage(const cv::Mat & image)
{
	QImage qtemp;
	if(!image.empty() && image.depth() == CV_8U)
	{
		const unsigned char * data = image.data;
		qtemp = QImage(image.cols, image.rows, QImage::Format_RGB32);
		for(int y = 0; y < image.rows; ++y, data += image.cols*image.elemSize())
		{
			for(int x = 0; x < image.cols; ++x)
			{
				QRgb * p = ((QRgb*)qtemp.scanLine (y)) + x;
				*p = qRgb(data[x * image.channels()+2], data[x * image.channels()+1], data[x * image.channels()]);
			}
		}
	}
	else if(!image.empty() && image.depth() != CV_8U)
	{
		printf("Wrong image format, must be 8_bits\n");
	}
	return qtemp;
}



void Simulator::decompose_rotation_xyz(const cv::Mat &R, double& thetaX, 
									   double& thetaY, double& thetaZ)
{

	// R = Rx * Ry * Rz order. R is CV_64F.
	thetaX = atan2(-R.at<double>(1, 2), R.at<double>(2, 2));
	thetaY = atan2(R.at<double>(0, 2), sqrt(R.at<double>(1, 2) * R.at<double>(1, 2)
		+ R.at<double>(2, 2) * R.at<double>(2, 2)));
	thetaZ = atan2(-R.at<double>(0, 1), R.at<double>(0, 0));
	// MATLAB:
	//x = atan2(-R(2,3), R(3,3));
	//y = atan2(R(1,3), sqrt(R(2,3)*R(2,3) + R(3,3)*R(3,3)));
	//z = atan2(-R(1,2), R(1,1));

}

void Simulator::compose_rotation(const double &thetaX, const double &thetaY,
								 const double &thetaZ, cv::Mat &R)
{
	cv::Mat X = cv::Mat::eye(3, 3, CV_64F);
	cv::Mat Y = cv::Mat::eye(3, 3, CV_64F);
	cv::Mat Z = cv::Mat::eye(3, 3, CV_64F);

	X.at<double>(1, 1) = cos(thetaX);
	X.at<double>(1, 2) = -sin(thetaX);
	X.at<double>(2, 1) = sin(thetaX);
	X.at<double>(2, 2) = cos(thetaX);

	Y.at<double>(0, 0) = cos(thetaY);
	Y.at<double>(0, 2) = sin(thetaY);
	Y.at<double>(2, 0) = -sin(thetaY);
	Y.at<double>(2, 2) = cos(thetaY);

	Z.at<double>(0, 0) = cos(thetaZ);
	Z.at<double>(0, 1) = -sin(thetaZ);
	Z.at<double>(1, 0) = sin(thetaZ);
	Z.at<double>(1, 1) = cos(thetaZ);

	R = X * Y * Z; // R is CV_64F.
}

double Simulator::get_orientation(std::vector<cv::Point> &contour, cv::Mat &img)
{
	cv::Moments ms = cv::moments(contour);
	double theta = 0.5 * atan(2 * ms.mu11 / (ms.mu20 - ms.mu02 ));
	double slope = tan(theta);
	//theta = (theta / CV_PI) * 180;
	double x_mass = ms.m10/ms.m00;
	double y_mass = ms.m01/ms.m00;
	cv::circle(img, cv::Point(x_mass, y_mass), 3, CV_RGB(0, 0, 255), 2);
	cv::Point p(0,0), q(img.cols,img.rows);
	p.y = -(x_mass - p.x) * slope + y_mass;
	q.y = -(x_mass - q.x) * slope + y_mass;
	cv::line(img,p,q, CV_RGB(255, 0, 255), 3, 8, 0);
	//std::cout<< (theta / CV_PI) * 180 << std::endl;
	return theta;
}

void Simulator::calc_shaft_slope(const cv::Mat &cHb,
								 const Eigen::Matrix<double, 4, 4, Eigen::RowMajor> &psm_Hj4,
								 const Eigen::Matrix<double, 4, 4, Eigen::RowMajor> &psm_Hj5,
								 const Eigen::Matrix<double, 4, 4, Eigen::RowMajor> &psm_He,
								 const float jaw_in,
								 float& psm_slope)
{
	cv::Mat subP = cv::Mat::eye(3, 3, CV_32FC1);
	subP.at<float>(0,0) = m_cam->Fx(); subP.at<float>(1,1) = m_cam->Fy();
	subP.at<float>(0,2) = m_cam->Px_win(); subP.at<float>(1,2) = m_cam->Py_win();

	std::vector<cv::Mat> bHj_vec;
	std::vector<cv::Point3f> cPj(4);

	for (int i = 0; i < 3; i++)
		bHj_vec.push_back(cv::Mat(4, 4, CV_64F));


	::memcpy(bHj_vec[0].data, psm_Hj4.data(), sizeof(double)*16);
	::memcpy(bHj_vec[1].data, psm_Hj5.data(), sizeof(double)*16);
	::memcpy(bHj_vec[2].data, psm_He.data(), sizeof(double)*16);


	for (int i = 1; i <= bHj_vec.size(); i++)
	{
		// NB: bHj_vec unit in mm
		cv::Mat crHj_curr = cHb * bHj_vec[i-1];
		cPj[i].x = crHj_curr.at<double>(0, 3);
		cPj[i].y = crHj_curr.at<double>(1, 3);
		cPj[i].z = abs(crHj_curr.at<double>(2, 3));    // abs to ensure projection make sense
	}

	// NB: Shaft line
	cv::Mat j2H_offset = cv::Mat::eye(4,4,CV_64FC1);
	j2H_offset.at<double>(0, 3) = 50;  // 200mm
	cv::Mat temp = cHb * bHj_vec[0] * j2H_offset;
	cPj[0].x = temp.at<double>(0, 3);
	cPj[0].y = temp.at<double>(1, 3);
	cPj[0].z = abs(temp.at<double>(2, 3));    // abs to ensure projection make sense

	std::vector<cv::Point2f> projectedPoints;
	cv::Mat rVec, tVec, distCoeffs;
	rVec = cv::Mat::zeros(3,1,CV_32FC1); tVec = cv::Mat::zeros(3,1,CV_32FC1);
	cv::projectPoints(cPj, rVec, tVec, subP, distCoeffs, projectedPoints);

	psm_slope = (projectedPoints[1].y - projectedPoints[0].y) / (projectedPoints[1].x - projectedPoints[0].x);
}

#ifdef __linux__
// Framerate calculation
double time_to_double(timeval *t)
{
	return (t->tv_sec + (t->tv_usec/1000000.0));
}

double time_diff(timeval *t1, timeval *t2)
{
	return time_to_double(t2) - time_to_double(t1);
}
#endif



