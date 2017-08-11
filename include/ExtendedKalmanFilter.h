/**
 * ExtendedKalmanFilter.h
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

#include "Eigen\Core"
#include "Eigen\Geometry"
#include "Eigen\StdVector"


class ExtendedKalmanFilter
{
public:
	ExtendedKalmanFilter();
	~ExtendedKalmanFilter();
private:
	////Dimensions
	//Dimension of State Vector
	int kalman_x_dimension;
	//Dimension of Measurement Vector
	int kalman_z_dimension;
	//Dimension of Control Vector
	int kalman_u_dimension;
	//Dimension of Process Noise Vector
	int kalman_w_dimension;
	//Dimension of Measurement Noise Vector
	int kalman_v_dimension;

	////Vectors
	//State Vector
	Eigen::MatrixXd kalman_x_last;
	Eigen::MatrixXd kalman_x_estimated;
	Eigen::MatrixXd kalman_x_corrected;
	//Control Vector
	Eigen::MatrixXd kalman_u;
	//Measurement Vector
	Eigen::MatrixXd kalman_z;
	Eigen::MatrixXd kalman_z_estimated;
	////Matrices
	Eigen::MatrixXd kalman_P_last;
	Eigen::MatrixXd kalman_P_estimated;
	Eigen::MatrixXd kalman_P_corrected;
	Eigen::MatrixXd kalman_A;						
	Eigen::MatrixXd kalman_W;
	Eigen::MatrixXd kalman_Q;
	Eigen::MatrixXd kalman_H;
	Eigen::MatrixXd kalman_V;
	Eigen::MatrixXd kalman_R;
	Eigen::MatrixXd kalman_S;
	Eigen::MatrixXd kalman_K;

	int iteration_number;

public:
	
	////Auxiliary Functions
 	int CheckDimension(Eigen::MatrixXd M1,Eigen::MatrixXd M2);
	
	int Get_x_estimated(std::vector<double>& x_estimated);
	int Get_x_corrected(std::vector<double>& x_corrected);

	////Initialize Functions
	int SetDimensions(int x_dimension, 
		int u_dimension, 
		int z_dimension, 
		int w_dimension, 
		int v_dimension);
	int InitMatrices();
	int ClearMatrices();

	////EKF Working Functions
	
	//Set process noise Q & measurement noise R
	int Set_NoiseCovariance(Eigen::MatrixXd Q,Eigen::MatrixXd R);
	
	// Initialise x
	int Init_x(Eigen::MatrixXd x);
	// Initialise P to identity
	int Init_P();

	//// Jacobians. These need to be set beforehand
	//A=df/dx 
	int Set_A();
	//W=df/dw
	int Set_W();
	//H=dz/dx
	int Set_H();
	//V=dh/dv
	int Set_V();
	//// End of Jacobians

	//// Step 1. Estimate the current state (Prediction)
	//f(x,u,0) 
	int Calc_x_estimated();
	//// End of Step 1.

	//// Step 2. Estimate the error covariance (Prediction)
	//P-=A*P*At+W*Q*Wt
	int Calc_P_estimated();
	//// End of Step 2.

	//// Step 3. Compute Kalman gain (Correction)
	//Set H on-the-fly
	int Set_H(Eigen::MatrixXd H);
	//S=H*P-*Ht+V*R*Vt	V=I
	int Calc_S();
	//K=P*Ht*S_inv
	int Calc_K();
	//// End of Step 3.

	//// Step 4. Update estimate with measurement (Correction)
	//Set Measurement
	int Set_z(Eigen::MatrixXd z);
	//zk=h(xk-,0)
	int Set_z_estimated(Eigen::MatrixXd z_estimated);
	//int Calc_z_estimated();
	//x_corrected=x_estimated+K(z-z_estimated);
	int Calc_x_corrected();
	int Calc_x_corrected(const std::vector<bool>& z_valid);
	//// End of Step 4.

	//// Step 5. Update error covariance (Correction)
	//P=(I-K*H)*P_estimated
	int Calc_P_corrected();
	//// End of Step 5.

	//// Step 6. Finish for next loop
	int Finish_for_next();
	//// End of Step 6.
};
