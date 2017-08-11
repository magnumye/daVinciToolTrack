/**
 * ShapeContextPro.h
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

#include <vector>
#include <opencv2/opencv.hpp>

struct SCConfig
{
	int sampleSize;
	int maxNumSamples;
};

struct ProsacPoint2f
{
	float x;
	float y;
};

struct PROSACSample
{
	double score;
	int numInliers;
	//cv::Mat solution;
	// point IDs
	std::vector<int> sample;
	std::vector<bool> inliers;

	// inline bool operator>(const PROSACSample& sp);
};

class ShapeContextPro
{
public:
	ShapeContextPro(SCConfig config);
	~ShapeContextPro();
	void SetData(const std::vector<ProsacPoint2f>* pointsA, const std::vector<ProsacPoint2f>* pointsB, const std::vector<int>* detectionIds);
    void SetDetectionIds(const std::vector<int>* landmarkIds);
    void SetScores(const std::vector<double>* scores);

	void DrawSamples(std::vector<int> &randList);
	bool DrawASample(std::vector<int> &sample) const;
	bool SampleValidation(const std::vector<int> &sample) const;

	void ComputeModel();

	bool Refine();
	bool RunKernel(const std::vector<int>& sample, cv::Mat& solution);
	inline bool IsInlier(float radiusA, float angleA, float radiusB, float angleB, float scaleFactor) const
	{
		if (fabs(angleA - angleB) < 10 && fabs(radiusA - scaleFactor * radiusB) < 8)
		{
			return true;
		}
		else
		{
			return false;
		}
	}

	inline int GetBestNumInliers() const {return sBestSample.numInliers;}
	inline const std::vector<bool>& GetBestInliers() const {return sBestSample.inliers;}
	inline const PROSACSample& GetBestSample() const {return sBestSample;}
	inline const PROSACSample& Get2ndBestSample() const {return s2ndBestSample;}
//	inline const cv::Mat& GetBestSolution() const {return sBestSample.solution;}
private:
	float sInlierThreshold;
	int sIterations;
	int sNumSamples;
	int sDataSize;
    const std::vector<ProsacPoint2f>* sPointsA;
    const std::vector<ProsacPoint2f>* sPointsB;
    const std::vector<int>* sLandmarkIds;
    const std::vector<int>* sDetectionIds;
    const std::vector<double>* sScores;

    PROSACSample sBestSample;
    PROSACSample s2ndBestSample;

	int sSampleSize;
	std::vector<int> sSample;
	int sSetSize;
};

