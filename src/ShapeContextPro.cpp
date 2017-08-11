#include "ShapeContextPro.h"
#include <cmath>

ShapeContextPro::ShapeContextPro(SCConfig config) : 
	sIterations(config.maxNumSamples), sSampleSize(config.sampleSize)
{
	sPointsA = 0;
	sPointsB = 0;
	sScores = 0;

	sDataSize = 0;

	sLandmarkIds = 0;
	sDetectionIds = 0;
}

ShapeContextPro::~ShapeContextPro()
{

}

void ShapeContextPro::SetData(const std::vector<ProsacPoint2f>* pointsA,
		const std::vector<ProsacPoint2f>* pointsB,
		const std::vector<int>* landmarkIds)
{
	sPointsA = pointsA;
	sPointsB = pointsB;
	sLandmarkIds = landmarkIds;
	sDataSize = (int) pointsA->size();
}

void ShapeContextPro::SetDetectionIds(const std::vector<int>* detectionIds)
{
	sDetectionIds = detectionIds;
}

void ShapeContextPro::SetScores(const std::vector<double>* scores)
{
	sScores = scores;
}

void ShapeContextPro::ComputeModel()
{
	sBestSample.score = 0;
	s2ndBestSample.score = 0;
	sBestSample.numInliers = 0;
	s2ndBestSample.numInliers = 0;
	
	std::vector<int> randList;
	DrawSamples(randList);
	int iterations = randList.size() / sSampleSize;

	for(int i = 0; i < iterations; i++)
	{

		// get centre position
		float centreXA = 0.0;
		float centreYA = 0.0;
		float centreXB = 0.0;
		float centreYB = 0.0;
		int idx = sSampleSize * i;

		centreXA = (*sPointsA)[randList[idx]].x;
		centreYA = (*sPointsA)[randList[idx]].y;
		centreXB = (*sPointsB)[randList[idx]].x;
		centreYB = (*sPointsB)[randList[idx]].y;

		float diffXA = ((*sPointsA)[randList[idx+1]].x - centreXA);
		float diffYA = ((*sPointsA)[randList[idx+1]].y - centreYA);

		float rA = std::sqrt(diffXA * diffXA + diffYA * diffYA); 


		float diffXB = ((*sPointsB)[randList[idx+1]].x - centreXB);
		float diffYB = ((*sPointsB)[randList[idx+1]].y - centreYB);

		float rB = std::sqrt(diffXB * diffXB + diffYB * diffYB);
		
		float scaleFactor = 1.0;//rA / rB;

		double score = 0.0;
		int numInliers = 0;
		std::map<int, bool> landmarksUsed;
		std::map<int, bool> detectionsUsed;
		std::vector<bool> inliers(sDataSize, false);

		for (int j = 0; j < sDataSize; j++)
		{
			float vecXA = (*sPointsA)[j].x - centreXA;
			float vecYA = (*sPointsA)[j].y - centreYA;

			float vecRA = std::sqrt(vecXA * vecXA + vecYA * vecYA);

			float vecXB = (*sPointsB)[j].x - centreXB;
			float vecYB = (*sPointsB)[j].y - centreYB;

			float vecRB = std::sqrt(vecXB * vecXB + vecYB * vecYB);

			float angleA = std::acos((vecXA * diffXA + vecYA * diffYA) / (rA * vecRA)) * 180 / CV_PI;
			
			float angleB = std::acos((vecXB * diffXB + vecYB * diffYB) / (rB * vecRB)) * 180 / CV_PI;
			
			// check the direction using cross product
			if (diffXA * vecYA - diffYA * vecXA < 0)
				angleA = 360 - angleA;
			if (diffXB * vecYB - diffYB * vecXB < 0)
				angleB = 360 - angleB;

			// shape context verification and score calculation
			if (IsInlier(vecRA, angleA, vecRB, angleB, scaleFactor))
			{
				if ( (sLandmarkIds && landmarksUsed.count((*sLandmarkIds)[j])) || (sDetectionIds && detectionsUsed.count((*sDetectionIds)[j])) )
				{
					inliers[j] = 0;
					continue;
				}
				inliers[j] = 1;
				numInliers++;
				if (sScores)
				{
					score += (*sScores)[j];
				}

				if (sLandmarkIds)
				{
					landmarksUsed[(*sLandmarkIds)[j]] = true;
				}
				if (sDetectionIds)
				{
					detectionsUsed[(*sDetectionIds)[j]] = true;
				}
			}
			//else
			//{
			//	inliers[j] = 0;
			//}
		}

		if (sBestSample.numInliers < numInliers)
		{
			s2ndBestSample.numInliers = sBestSample.numInliers;
			s2ndBestSample.score = sBestSample.score;
			s2ndBestSample.inliers = sBestSample.inliers;

			sBestSample.numInliers = numInliers;
			sBestSample.score = score;
			sBestSample.inliers = inliers;
		}

	}


	if (sBestSample.numInliers > 3)
		if (!Refine())
			sBestSample.numInliers = 0;


}

bool ShapeContextPro::Refine()
{
	float centreXA = 0.0;
	float centreYA = 0.0;
	float centreXB = 0.0;
	float centreYB = 0.0;

	for (int j = 0; j < sDataSize; j++)
	{
		if (sBestSample.inliers[j])
		{
			centreXA += (*sPointsA)[j].x;
			centreYA += (*sPointsA)[j].y;

			centreXB += (*sPointsB)[j].x;
			centreYB += (*sPointsB)[j].y;
		}
	}
	centreXA /= sBestSample.numInliers;
	centreYA /= sBestSample.numInliers;
	centreXB /= sBestSample.numInliers;
	centreYB /= sBestSample.numInliers;

	// get axis line equation ax + b - y = 0
	float diffXA = ((*sPointsA)[0].x - centreXA);
	float diffYA = ((*sPointsA)[0].y - centreYA);

	float rA = std::sqrt(diffXA * diffXA + diffYA * diffYA); 


	float diffXB = ((*sPointsB)[0].x - centreXB);
	float diffYB = ((*sPointsB)[0].y - centreYB);

	float rB = std::sqrt(diffXB * diffXB + diffYB * diffYB);

	float scaleFactor = rA / rB;

	sBestSample.score = 0.0;
	sBestSample.numInliers = 0;
	std::map<int, bool> landmarksUsed;
	std::map<int, bool> detectionsUsed;

	for (int j = 0; j < sDataSize; j++)
	{
		float vecXA = (*sPointsA)[j].x - centreXA;
		float vecYA = (*sPointsA)[j].y - centreYA;

		float vecRA = std::sqrt(vecXA * vecXA + vecYA * vecYA);

		float vecXB = (*sPointsB)[j].x - centreXB;
		float vecYB = (*sPointsB)[j].y - centreYB;

		float vecRB = std::sqrt(vecXB * vecXB + vecYB * vecYB);

		float angleA = std::acos((vecXA * diffXA + vecYA * diffYA) / (rA * vecRA)) * 180 / CV_PI;

		float angleB = std::acos((vecXB * diffXB + vecYB * diffYB) / (rB * vecRB)) * 180 / CV_PI;

		// check the direction using cross product
		if (diffXA * vecYA - diffYA * vecXA < 0)
			angleA = 360 - angleA;
		if (diffXB * vecYB - diffYB * vecXB < 0)
			angleB = 360 - angleB;

		// shape context verification and score calculation
		if (IsInlier(vecRA, angleA, vecRB, angleB, scaleFactor))
		{
			if ( (sLandmarkIds && landmarksUsed.count((*sLandmarkIds)[j])) || (sDetectionIds && detectionsUsed.count((*sDetectionIds)[j])) )
			{
				sBestSample.inliers[j] = 0;
				continue;
			}
			sBestSample.inliers[j] = 1;
			sBestSample.numInliers++;
			if (sScores)
			{
				sBestSample.score += (*sScores)[j];
			}

			if (sLandmarkIds)
			{
				landmarksUsed[(*sLandmarkIds)[j]] = true;
			}
			if (sDetectionIds)
			{
				detectionsUsed[(*sDetectionIds)[j]] = true;
			}
		}
		else
		{
			sBestSample.inliers[j] = 0;
		}
	}

	//std::vector<int> sample;
	//sample.reserve(sBestSample.numInliers);
	//for (int i = 0; i < sDataSize; i++)
	//{
	//	if (sBestSample.inliers[i])
	//	{
	//		sample.push_back(i);
	//	}
	//}

	//cv::Mat solution;
	//bool success = RunKernel(sample, solution);

	//if (!success)
	//{
	//	std::cout << "Refine failed!" << std::endl;
	//	return false;
	//}

	//std::vector<bool> inliers;
	//inliers.resize(sDataSize);

	//sBestSample.solution = solution;


	return true;
}

bool ShapeContextPro::RunKernel(const std::vector<int>& sample,
		cv::Mat& solution)
{
	// reference: opencv - CvHomographyEstimator::runkernel

	if (solution.empty())
		solution.create(3, 3, CV_64F);
	CvMat H = solution;

	int i, count = (int) sample.size();

	double LtL[9][9], W[9][1], V[9][9];
	CvMat _LtL = cvMat(9, 9, CV_64F, LtL);
	CvMat matW = cvMat(9, 1, CV_64F, W);
	CvMat matV = cvMat(9, 9, CV_64F, V);
	CvMat _H0 = cvMat(3, 3, CV_64F, V[8]);
	CvMat _Htemp = cvMat(3, 3, CV_64F, V[7]);
	CvPoint2D64f cM =
	{ 0, 0 }, cm =
	{ 0, 0 }, sM =
	{ 0, 0 }, sm =
	{ 0, 0 };

	for (i = 0; i < count; i++)
	{
		ProsacPoint2f M = (*sPointsA)[sample[i]];
		ProsacPoint2f m = (*sPointsB)[sample[i]];
		cm.x += m.x;
		cm.y += m.y;
		cM.x += M.x;
		cM.y += M.y;
	}

	cm.x /= count;
	cm.y /= count;
	cM.x /= count;
	cM.y /= count;

	for (i = 0; i < count; i++)
	{
		ProsacPoint2f M = (*sPointsA)[sample[i]];
		ProsacPoint2f m = (*sPointsB)[sample[i]];
		sm.x += fabs(m.x - cm.x);
		sm.y += fabs(m.y - cm.y);
		sM.x += fabs(M.x - cM.x);
		sM.y += fabs(M.y - cM.y);
	}

	if (fabs(sm.x) < DBL_EPSILON || fabs(sm.y) < DBL_EPSILON
			|| fabs(sM.x) < DBL_EPSILON || fabs(sM.y) < DBL_EPSILON)
		return false;

	sm.x = count / sm.x;
	sm.y = count / sm.y;
	sM.x = count / sM.x;
	sM.y = count / sM.y;

	double invHnorm[9] =
	{ 1. / sm.x, 0, cm.x, 0, 1. / sm.y, cm.y, 0, 0, 1 };
	double Hnorm2[9] =
	{ sM.x, 0, -cM.x * sM.x, 0, sM.y, -cM.y * sM.y, 0, 0, 1 };
	CvMat _invHnorm = cvMat(3, 3, CV_64FC1, invHnorm);
	CvMat _Hnorm2 = cvMat(3, 3, CV_64FC1, Hnorm2);

	cvZero(&_LtL);
	for (i = 0; i < count; i++)
	{
		ProsacPoint2f M = (*sPointsA)[sample[i]];
		ProsacPoint2f m = (*sPointsB)[sample[i]];
		double x = (m.x - cm.x) * sm.x, y = (m.y - cm.y) * sm.y;
		double X = (M.x - cM.x) * sM.x, Y = (M.y - cM.y) * sM.y;
		double Lx[] =
		{ X, Y, 1, 0, 0, 0, -x * X, -x * Y, -x };
		double Ly[] =
		{ 0, 0, 0, X, Y, 1, -y * X, -y * Y, -y };
		int j, k;
		for (j = 0; j < 9; j++)
			for (k = j; k < 9; k++)
				LtL[j][k] += Lx[j] * Lx[k] + Ly[j] * Ly[k];
	}
	cvCompleteSymm(&_LtL);

	//cvSVD( &_LtL, &matW, 0, &matV, CV_SVD_MODIFY_A + CV_SVD_V_T );
	cvEigenVV(&_LtL, &matV, &matW);
	cvMatMul(&_invHnorm, &_H0, &_Htemp);
	cvMatMul(&_Htemp, &_Hnorm2, &_H0);
	cvConvertScale(&_H0, &H, 1. / _H0.data.db[8]);

	return true;
}

void ShapeContextPro::DrawSamples(std::vector<int>& randList)
{
	randList.reserve(sSampleSize * sIterations);
	sSample.resize(sSampleSize);
	sSetSize = sSampleSize;
	double Tn = 1.0;
	int Tn_dash = 1;
	sNumSamples = 0;
	int failures = 0;
	for (sNumSamples = 1; sNumSamples <= sIterations; sNumSamples++)
	{
		// 1. Choice of the hypothesis generation set
        if ((sNumSamples == Tn_dash || failures == 5) && sSetSize < sDataSize)
        {
            sSetSize++;
            failures = 0;
            double Tn_minusOne = Tn;
            Tn *= (((double)sSetSize)/(sSetSize - sSampleSize));
            Tn_dash += (int)std::ceil(Tn - Tn_minusOne);
        }
        
        // 2. Semi-random sampling
        bool success = DrawASample(sSample);
        if (!success)
        {
            failures++;
            continue;
        }

		// 3. Fill the list
		for (int i = 0; i < sSampleSize; i++)
		{
			randList.push_back(sSample[i]);
		}


		failures = 0;
	}
}

bool ShapeContextPro::DrawASample(std::vector<int> &sample) const
{
	if (sSampleSize > sSetSize)
	{
		return false;
	}
    
    int count = 0;
    const int maxCount = 20;
    bool isValid = false;
    
    while (!isValid && count++ < maxCount)
    {
        for (int i = 0; i < (sSampleSize - 1); i++)
        {
            sample[i] = rand() % (sSetSize - 1);
            for (int j = 0; j < i; j++)
            {
                if (sample[i] == sample[j])
                {
                    // re-draw
                    i--;
                    break;
                }
            }
        }
        sample[sSampleSize - 1] = sSetSize - 1;
        
        isValid = SampleValidation(sample);
    }
    
    return isValid;
}

bool ShapeContextPro::SampleValidation(const std::vector<int> &sample) const
{
    for (int i = 0; i < (sSampleSize - 1); i++)
    {
        const ProsacPoint2f& PAi = (*sPointsA)[sample[i]];
        const ProsacPoint2f& PBi = (*sPointsB)[sample[i]];
        
        for (int j = i + 1; j < sSampleSize; j++)
        {
            const ProsacPoint2f& PAj = (*sPointsA)[sample[j]];
            const ProsacPoint2f& PBj = (*sPointsB)[sample[j]];
            
            if ((PAi.x == PAj.x && PAi.y == PAj.y) || (PBi.x == PBj.x && PBi.y == PBj.y))
            {
                return false;
            }
        }
    }
    
    return true;
}
