#include "ExtendedKalmanFilter.h"
#include <iostream>
#include <fstream>

ExtendedKalmanFilter::ExtendedKalmanFilter()
{
	////Dimensions
	//Dimension of State Vector
	kalman_x_dimension=0;
	//Dimension of Measurement Vector
	kalman_z_dimension=0;
	//Dimension of Control Vector
	kalman_u_dimension=0;
	//Dimension of Process Noise Vector
	kalman_w_dimension=0;
	//Dimension of Measurement Noise Vector
	kalman_v_dimension=0;


	iteration_number = 0;
}

ExtendedKalmanFilter::~ExtendedKalmanFilter()
{

}

int ExtendedKalmanFilter::ClearMatrices()
{

	return 0;
}

int ExtendedKalmanFilter::SetDimensions(int x_dimension, int u_dimension, int z_dimension, int w_dimension, int v_dimension)
{
	kalman_x_dimension=x_dimension;
	kalman_u_dimension=u_dimension;
	kalman_z_dimension=z_dimension;
	kalman_w_dimension=w_dimension;
	kalman_v_dimension=v_dimension;

	return 0;
}

int ExtendedKalmanFilter::CheckDimension(Eigen::MatrixXd M1,Eigen::MatrixXd M2)
{
	if (M1.cols()!=M2.cols() ||M1.rows()!=M2.rows())
		return -1;

	return 0;
}

int ExtendedKalmanFilter::InitMatrices()
{
	ClearMatrices();

	////Vectors
	//State Vector
	kalman_x_last = Eigen::MatrixXd::Zero(kalman_x_dimension,1);
	kalman_x_estimated = Eigen::MatrixXd::Zero(kalman_x_dimension,1);
	kalman_x_corrected = Eigen::MatrixXd::Zero(kalman_x_dimension,1);

	//Control Vector
	kalman_u = Eigen::MatrixXd::Zero(kalman_u_dimension,1);

	//Measurement Vector
	kalman_z = Eigen::MatrixXd::Zero(kalman_z_dimension,1);
	kalman_z_estimated = Eigen::MatrixXd::Zero(kalman_z_dimension,1);

	////Matrices
	kalman_P_last = Eigen::MatrixXd::Zero(kalman_x_dimension,kalman_x_dimension);
	kalman_P_estimated = Eigen::MatrixXd::Zero(kalman_x_dimension,kalman_x_dimension);
	kalman_P_corrected = Eigen::MatrixXd::Zero(kalman_x_dimension,kalman_x_dimension);
	kalman_A = Eigen::MatrixXd::Zero(kalman_x_dimension,kalman_x_dimension);
	kalman_W = Eigen::MatrixXd::Zero(kalman_x_dimension,kalman_w_dimension);
	kalman_Q = Eigen::MatrixXd::Zero(kalman_w_dimension,kalman_w_dimension);
	kalman_H = Eigen::MatrixXd::Zero(kalman_z_dimension,kalman_x_dimension);
	kalman_V = Eigen::MatrixXd::Zero(kalman_z_dimension,kalman_v_dimension);
	kalman_R = Eigen::MatrixXd::Zero(kalman_v_dimension,kalman_v_dimension);
	kalman_S = Eigen::MatrixXd::Zero(kalman_z_dimension,kalman_z_dimension);
	kalman_K = Eigen::MatrixXd::Zero(kalman_x_dimension,kalman_z_dimension);



	return 0;
}

int ExtendedKalmanFilter::Set_NoiseCovariance(Eigen::MatrixXd Q,Eigen::MatrixXd R)
{
	
	if (CheckDimension(Q,kalman_Q)!=0 || CheckDimension(R,kalman_R)!=0)
		return -1;
	
	std::copy(Q.data(), Q.data()+Q.size(), kalman_Q.data());
	std::copy(R.data(), R.data()+R.size(), kalman_R.data());

	return 0;
}

int ExtendedKalmanFilter::Init_x(Eigen::MatrixXd x)
{
	std::copy(x.data(), x.data()+x.size(), kalman_x_last.data());

	return 0;
}

int ExtendedKalmanFilter::Init_P()
{
	kalman_P_last.setIdentity();
	//cvCopy(P,kalman_P_last);
	return 0;
}

int ExtendedKalmanFilter::Set_A()
{
	kalman_A.setIdentity();
	return 0;
}

int ExtendedKalmanFilter::Set_W()
{
	kalman_W.setIdentity();
	return 0;
}


//int EKFilter::Set_H()
//{
//	cvSetZero(kalman_H);
//	return 0;
//}

int ExtendedKalmanFilter::Set_H(Eigen::MatrixXd H)
{
	if (CheckDimension(H,kalman_H)!=0)
	{
		std::cout << "Dimensions of H and kalman_h are not the same!" << std::endl;
		return -1;
	}
	std::copy(H.data(), H.data()+H.size(), kalman_H.data());


	return 0;
}

int ExtendedKalmanFilter::Set_V()
{
	kalman_V.setIdentity();
	return 0;
}

int ExtendedKalmanFilter::Calc_x_estimated()
{
	std::copy(kalman_x_last.data(), kalman_x_last.data()+kalman_x_last.size(), kalman_x_estimated.data());

	//cout << "kalman_x_estimated" << endl;
	//cv::Mat t1(kalman_x_estimated);
	//cout << t1 << endl;
	return 0;
}


int ExtendedKalmanFilter::Calc_P_estimated()
{
	Eigen::MatrixXd kalman_At = kalman_A.transpose();
	Eigen::MatrixXd kalman_Wt = kalman_W.transpose();
	Eigen::MatrixXd kalman_A_x_P = kalman_A*kalman_P_last;
	Eigen::MatrixXd kalman_A_x_P_x_At = kalman_A_x_P*kalman_At;
	Eigen::MatrixXd kalman_W_x_Q = kalman_W*kalman_Q;
	Eigen::MatrixXd kalman_W_x_Q_x_Wt = kalman_W_x_Q*kalman_Wt;

	kalman_P_estimated = kalman_A_x_P_x_At+kalman_W_x_Q_x_Wt;


	return 0;
}


int ExtendedKalmanFilter::Calc_S()
{


	Eigen::MatrixXd kalman_Ht = kalman_H.transpose();
	Eigen::MatrixXd kalman_Vt = kalman_V.transpose();
	Eigen::MatrixXd kalman_H_x_P = kalman_H*kalman_P_estimated;
	Eigen::MatrixXd kalman_H_x_P_x_Ht = kalman_H_x_P*kalman_Ht;
	Eigen::MatrixXd kalman_V_x_R = kalman_V*kalman_R;
	Eigen::MatrixXd kalman_V_x_R_x_Vt = kalman_V_x_R*kalman_Vt;
	
	kalman_S = kalman_H_x_P_x_Ht+kalman_V_x_R_x_Vt;



	return 0;
}

int ExtendedKalmanFilter::Calc_K()
{

	Eigen::MatrixXd kalman_Ht = kalman_H.transpose();
	Eigen::MatrixXd kalman_Sinv = kalman_S.inverse();
	Eigen::MatrixXd kalman_P_x_Ht = kalman_P_estimated*kalman_Ht;
	kalman_K = kalman_P_x_Ht*kalman_Sinv;


	return 0;
}

int ExtendedKalmanFilter::Set_z(Eigen::MatrixXd z)
{
	if (CheckDimension(z,kalman_z)!=0)
		return -1;
	std::copy(z.data(), z.data()+z.size(), kalman_z.data());

	//cvReleaseMat(&kalman_z);
	//kalman_z=cvCloneMat(z);
	return 0;
}

int ExtendedKalmanFilter::Set_z_estimated(Eigen::MatrixXd z_estimated)
{
	if (CheckDimension(z_estimated,kalman_z_estimated)!=0)
		return -1;
	std::copy(z_estimated.data(), z_estimated.data()+z_estimated.size(), kalman_z_estimated.data());


	return 0;
}

//int EKFilter::Calc_z_estimated()
//{
//	return 0;
//}

int ExtendedKalmanFilter::Calc_x_corrected()
{
	Eigen::MatrixXd kalman_z_minus_z_estimated = kalman_z-kalman_z_estimated;
	Eigen::MatrixXd kalman_K_x_z_minus_z_estimated = kalman_K*kalman_z_minus_z_estimated;
	kalman_x_corrected = kalman_x_estimated+kalman_K_x_z_minus_z_estimated;

	return 0;
}

int ExtendedKalmanFilter::Calc_x_corrected(const std::vector<bool>& z_valid)
{
	if (kalman_z_dimension != (int)z_valid.size())
		return -1;
	Eigen::MatrixXd kalman_z_minus_z_estimated = kalman_z-kalman_z_estimated;

	// Set validity of measurements
	for (int i = 0; i < kalman_z_dimension; i++)
	{
		if (!z_valid[i])
		{
			kalman_z_minus_z_estimated.data()[i] = 0;
		}
		else // if measurement difference is large, need to skip this iteration.
		{
            if (abs(kalman_z_minus_z_estimated.data()[i]) > 80)
			{
				std::cout << "kalman_z_minus_z_estimated too big!" << std::endl;
				
				return -1;
			}
		}
	}
	Eigen::MatrixXd kalman_K_x_z_minus_z_estimated = kalman_K*kalman_z_minus_z_estimated;
	kalman_x_corrected = kalman_x_estimated+kalman_K_x_z_minus_z_estimated;



	return 0;
}

int ExtendedKalmanFilter::Calc_P_corrected()
{
	Eigen::MatrixXd kalman_I;
	kalman_I.setIdentity(kalman_P_estimated.rows(),kalman_P_estimated.rows());
	Eigen::MatrixXd kalman_K_x_H = kalman_K*kalman_H;
	Eigen::MatrixXd kalman_I_minus_K_x_H = kalman_I-kalman_K_x_H;
	kalman_P_corrected = kalman_I_minus_K_x_H*kalman_P_estimated;
	
	//Make sure kalman_P_corrected is symmetric
	for (int row=0;row<kalman_P_corrected.rows();row++)
	{
		for (int col=row;col<kalman_P_corrected.cols();col++)
		{
			double tmp;
			tmp = kalman_P_corrected(row,col) + kalman_P_corrected(col,row);
			tmp=tmp/2;
			kalman_P_corrected(row,col) = tmp;
			kalman_P_corrected(col,row) = tmp;
		}
	}


	return 0;
}

int ExtendedKalmanFilter::Finish_for_next()
{
	std::copy(kalman_x_corrected.data(), kalman_x_corrected.data()+kalman_x_corrected.size(),
		kalman_x_last.data());
	std::copy(kalman_P_corrected.data(), kalman_P_corrected.data()+kalman_P_corrected.size(),
		kalman_P_last.data());

	iteration_number++;

	return 0;
}


int ExtendedKalmanFilter::Get_x_estimated(std::vector<double>& x_estimated)
{
	
	x_estimated.resize(kalman_x_estimated.rows());
	for (int i = 0; i < kalman_x_estimated.rows(); i++)
	{
		x_estimated[i] = (double)kalman_x_estimated(i,0);
	}
	return 0;
}

int ExtendedKalmanFilter::Get_x_corrected(std::vector<double>& x_corrected)
{

	x_corrected.resize(kalman_x_corrected.rows());
	for (int i = 0; i < kalman_x_corrected.rows(); i++)
	{
		x_corrected[i] = (double)kalman_x_corrected(i,0);
	}
	return 0;
}













