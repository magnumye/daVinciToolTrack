#include "QuantisedGradOrientation.h"
#include <opencv2/core/internal.hpp>

#define  SQR(a) ((a) * (a))



QuantisedGradOrientation::QuantisedGradOrientation()
{

}
QuantisedGradOrientation::~QuantisedGradOrientation()
{

}

void QuantisedGradOrientation::hysteresisGradient(cv::Mat& magnitude, cv::Mat& quantized_angle,
												  cv::Mat& angle, float threshold)
{
	// Quantize 360 degree range of orientations into 16 buckets
	// Note that [0, 11.25), [348.75, 360) both get mapped in the end to label 0,
	// for stability of horizontal and vertical features.
	//cv::Mat_<unsigned char> quantized_unfiltered;
	//angle.convertTo(quantized_unfiltered, CV_8U, 16.0 / 360.0);
	const float alpha = 16.0/360.0;
	cv::Mat_<unsigned char> quantized_unfiltered(angle.size());
	for (int r = 0; r < angle.rows; r++)
	{
		float* angle_r = angle.ptr<float>(r);
		uchar* quantized_r = quantized_unfiltered.ptr<uchar>(r);
		for (int c = 0; c < angle.cols; c++)
		{
			quantized_r[c] = (uchar)floor(angle_r[c]*alpha + 0.5f);
		}
	}

	// Zero out top and bottom rows
	/// @todo is this necessary, or even correct?
	memset(quantized_unfiltered.ptr(), 0, quantized_unfiltered.cols);
	memset(quantized_unfiltered.ptr(quantized_unfiltered.rows - 1), 0, quantized_unfiltered.cols);
	// Zero out first and last columns
	for (int r = 0; r < quantized_unfiltered.rows; ++r)
	{
		quantized_unfiltered(r, 0) = 0;
		quantized_unfiltered(r, quantized_unfiltered.cols - 1) = 0;
	}

	// Mask 16 buckets into 8 quantized orientations
	for (int r = 1; r < angle.rows - 1; ++r)
	{
		uchar* quant_r = quantized_unfiltered.ptr<uchar>(r);
		for (int c = 1; c < angle.cols - 1; ++c)
		{
			quant_r[c] &= 7;
		}
	}

	// Filter the raw quantized image. Only accept pixels where the magnitude is above some
	// threshold, and there is local agreement on the quantization.
	quantized_angle = cv::Mat::zeros(angle.size(), CV_8U);
	for (int r = 1; r < angle.rows - 1; ++r)
	{
		float* mag_r = magnitude.ptr<float>(r);

		for (int c = 1; c < angle.cols - 1; ++c)
		{
			if (mag_r[c] > threshold)
			{
				// Compute histogram of quantized bins in 3x3 patch around pixel
				int histogram[8] = {0, 0, 0, 0, 0, 0, 0, 0};

				uchar* patch3x3_row = &quantized_unfiltered(r-1, c-1);
				histogram[patch3x3_row[0]]++;
				histogram[patch3x3_row[1]]++;
				histogram[patch3x3_row[2]]++;

				patch3x3_row += quantized_unfiltered.step1();
				histogram[patch3x3_row[0]]++;
				histogram[patch3x3_row[1]]++;
				histogram[patch3x3_row[2]]++;

				patch3x3_row += quantized_unfiltered.step1();
				histogram[patch3x3_row[0]]++;
				histogram[patch3x3_row[1]]++;
				histogram[patch3x3_row[2]]++;

				// Find bin with the most votes from the patch
				int max_votes = 0;
				int index = -1;
				for (int i = 0; i < 8; ++i)
				{
					if (max_votes < histogram[i])
					{
						index = i;
						max_votes = histogram[i];
					}
				}

				// Only accept the quantization if majority of pixels in the patch agree
				int NEIGHBOR_THRESHOLD = 5;
				if (max_votes >= NEIGHBOR_THRESHOLD)
					quantized_angle.at<uchar>(r, c) = uchar(1 << index);
			}
		}
	}
}

void QuantisedGradOrientation::quantizedOrientations(const cv::Mat& src, cv::Mat& magnitude,
													 cv::Mat& angle, float threshold)
{
	magnitude.create(src.size(), CV_32F);

	// Allocate temporary buffers
	cv::Size size = src.size();
	cv::Mat sobel_3dx; // per-channel horizontal derivative
	cv::Mat sobel_3dy; // per-channel vertical derivative
	cv::Mat sobel_dx(size, CV_32F);      // maximum horizontal derivative
	cv::Mat sobel_dy(size, CV_32F);      // maximum vertical derivative
	cv::Mat sobel_ag;  // final gradient orientation (unquantized)
	cv::Mat smoothed;

	// Compute horizontal and vertical image derivatives on all color channels separately
	static const int KERNEL_SIZE = 7;
	// For some reason cvSmooth/cv::GaussianBlur, cvSobel/cv::Sobel have different defaults for border handling...
	GaussianBlur(src, smoothed, cv::Size(KERNEL_SIZE, KERNEL_SIZE), 0, 0, cv::BORDER_REPLICATE);
	Sobel(smoothed, sobel_3dx, CV_16S, 1, 0, 3, 1.0, 0.0, cv::BORDER_REPLICATE);
	Sobel(smoothed, sobel_3dy, CV_16S, 0, 1, 3, 1.0, 0.0, cv::BORDER_REPLICATE);

	short * ptrx  = (short *)sobel_3dx.data;
	short * ptry  = (short *)sobel_3dy.data;
	float * ptr0x = (float *)sobel_dx.data;
	float * ptr0y = (float *)sobel_dy.data;
	float * ptrmg = (float *)magnitude.data;

	const int length1 = static_cast<const int>(sobel_3dx.step1());
	const int length2 = static_cast<const int>(sobel_3dy.step1());
	const int length3 = static_cast<const int>(sobel_dx.step1());
	const int length4 = static_cast<const int>(sobel_dy.step1());
	const int length5 = static_cast<const int>(magnitude.step1());
	const int length0 = sobel_3dy.cols * 3;

	for (int r = 0; r < sobel_3dy.rows; ++r)
	{
		int ind = 0;

		for (int i = 0; i < length0; i += 3)
		{
			// Use the gradient orientation of the channel whose magnitude is largest
			int mag1 = SQR(ptrx[i]) + SQR(ptry[i]);
			int mag2 = SQR(ptrx[i + 1]) + SQR(ptry[i + 1]);
			int mag3 = SQR(ptrx[i + 2]) + SQR(ptry[i + 2]);

			if (mag1 >= mag2 && mag1 >= mag3)
			{
				ptr0x[ind] = ptrx[i];
				ptr0y[ind] = ptry[i];
				ptrmg[ind] = (float)mag1;
			}
			else if (mag2 >= mag1 && mag2 >= mag3)
			{
				ptr0x[ind] = ptrx[i + 1];
				ptr0y[ind] = ptry[i + 1];
				ptrmg[ind] = (float)mag2;
			}
			else
			{
				ptr0x[ind] = ptrx[i + 2];
				ptr0y[ind] = ptry[i + 2];
				ptrmg[ind] = (float)mag3;
			}
			++ind;
		}
		ptrx += length1;
		ptry += length2;
		ptr0x += length3;
		ptr0y += length4;
		ptrmg += length5;
	}

	// Calculate the final gradient orientations
	phase(sobel_dx, sobel_dy, sobel_ag, true);
	hysteresisGradient(magnitude, angle, sobel_ag, SQR(threshold));
}

// Adapted from cv_show_angles
cv::Mat QuantisedGradOrientation::displayQuantized(const cv::Mat& quantized)
{
	cv::Mat color(quantized.size(), CV_8UC3);
	for (int r = 0; r < quantized.rows; ++r)
	{
		const uchar* quant_r = quantized.ptr(r);
		cv::Vec3b* color_r = color.ptr<cv::Vec3b>(r);

		for (int c = 0; c < quantized.cols; ++c)
		{
			cv::Vec3b& bgr = color_r[c];
			switch (quant_r[c])
			{
			case 0:   bgr[0]=  0; bgr[1]=  0; bgr[2]=  0;    break;
			case 1:   bgr[0]= 55; bgr[1]= 55; bgr[2]= 55;    break;
			case 2:   bgr[0]= 80; bgr[1]= 80; bgr[2]= 80;    break;
			case 4:   bgr[0]=105; bgr[1]=105; bgr[2]=105;    break;
			case 8:   bgr[0]=130; bgr[1]=130; bgr[2]=130;    break;
			case 16:  bgr[0]=155; bgr[1]=155; bgr[2]=155;    break;
			case 32:  bgr[0]=180; bgr[1]=180; bgr[2]=180;    break;
			case 64:  bgr[0]=205; bgr[1]=205; bgr[2]=205;    break;
			case 128: bgr[0]=230; bgr[1]=230; bgr[2]=230;    break;
			case 255: bgr[0]=  0; bgr[1]=  0; bgr[2]=255;    break;
			default:  bgr[0]=  0; bgr[1]=255; bgr[2]=  0;    break;
			}
		}
	}

	return color;
}


void QuantisedGradOrientation::colormap(const cv::Mat& quantized, cv::Mat& dst)
{
	std::vector<cv::Vec3b> lut(8);
	lut[0] = cv::Vec3b(  0,   0, 255);
	lut[1] = cv::Vec3b(  0, 170, 255);
	lut[2] = cv::Vec3b(  0, 255, 170);
	lut[3] = cv::Vec3b(  0, 255,   0);
	lut[4] = cv::Vec3b(170, 255,   0);
	lut[5] = cv::Vec3b(255, 170,   0);
	lut[6] = cv::Vec3b(255,   0,   0);
	lut[7] = cv::Vec3b(255,   0, 170);

	dst = cv::Mat::zeros(quantized.size(), CV_8UC3);
	//dst = cv::Scalar(80,80,80);
	for (int r = 0; r < dst.rows; ++r)
	{
		const uchar* quant_r = quantized.ptr(r);
		cv::Vec3b* dst_r = dst.ptr<cv::Vec3b>(r);
		for (int c = 0; c < dst.cols; ++c)
		{
			uchar q = quant_r[c];
			if (q)
				dst_r[c] = lut[getLabel(q)];
		}
	}
}


void QuantisedGradOrientation::selectScatteredFeatures(const std::vector<Candidate>& candidates,
													   std::vector<Feature>& features,
													   size_t num_features, float distance)
{
	features.clear();
	float distance_sq = SQR(distance);
	int i = 0;
	while (features.size() < num_features)
	{
		Candidate c = candidates[i];

		// Add if sufficient distance away from any previously chosen feature
		bool keep = true;
		for (int j = 0; (j < (int)features.size()) && keep; ++j)
		{
			const Feature& f = features[j];
			keep = SQR(c.f.x - f.x) + SQR(c.f.y - f.y) >= distance_sq;
		}
		if (keep)
			features.push_back(c.f);

		if (++i == (int)candidates.size())
		{
			// Start back at beginning, and relax required distance
			i = 0;
			distance -= 1.0f;
			distance_sq = SQR(distance);
		}
	}
}

bool QuantisedGradOrientation::extractTemplate(Template& templ, float strong_threshold, 
											   size_t num_features, const cv::Mat& magnitude, 
											   const cv::Mat& angle, const cv::Mat& mask, cv::Rect& box)
{
	// Want features on the border to distinguish from background
	cv::Mat local_mask;
	if (!mask.empty())
	{
		erode(mask, local_mask, cv::Mat(), cv::Point(-1,-1), 1, cv::BORDER_REPLICATE);
		subtract(mask, local_mask, local_mask);
	}

	// Create sorted list of all pixels with magnitude greater than a threshold
	std::vector<Candidate> candidates;
	bool no_mask = local_mask.empty();
	float threshold_sq = SQR(strong_threshold);
	for (int r = 0; r < magnitude.rows; ++r)
	{
		const uchar* angle_r = angle.ptr<uchar>(r);
		const float* magnitude_r = magnitude.ptr<float>(r);
		const uchar* mask_r = no_mask ? NULL : local_mask.ptr<uchar>(r);

		for (int c = 0; c < magnitude.cols; ++c)
		{
			if (no_mask || mask_r[c])
			{
				uchar quantized = angle_r[c];
				if (quantized > 0)
				{
					float score = magnitude_r[c];
					if (score > threshold_sq)
					{
						candidates.push_back(Candidate(c, r, getLabel(quantized), score));
					}
				}
			}
		}
	}
	// We require a certain number of features
	if (candidates.size() < num_features)
		return false;
	// NOTE: Stable sort to agree with old code, which used std::list::sort()
	std::stable_sort(candidates.begin(), candidates.end());

	// Use heuristic based on surplus of candidates in narrow outline for initial distance threshold
	float distance = static_cast<float>(candidates.size() / num_features + 1);
	selectScatteredFeatures(candidates, templ.features, num_features, distance);

	// Size determined externally (needs to match templates for other modalities)
	templ.width = -1;
	templ.height = -1;
	templ.pyramid_level = 0; // harded-coded for now

	// Determine the size of the template
	int min_x = std::numeric_limits<int>::max();
	int min_y = std::numeric_limits<int>::max();
	int max_x = std::numeric_limits<int>::min();
	int max_y = std::numeric_limits<int>::min();

	// First pass: find min/max feature x,y 
	for (int j = 0; j < (int)templ.features.size(); ++j)
	{
		int x = templ.features[j].x;
		int y = templ.features[j].y;
		min_x = std::min(min_x, x);
		min_y = std::min(min_y, y);
		max_x = std::max(max_x, x);
		max_y = std::max(max_y, y);
	}

	/// @todo Why require even min_x, min_y?
	if (min_x % 2 == 1) --min_x;
	if (min_y % 2 == 1) --min_y;


	// Second pass: set width/height and shift all feature positions
	templ.width = (max_x - min_x); // >> templ.pyramid_level;
	templ.height = (max_y - min_y); // >> templ.pyramid_level;
	int offset_x = min_x; // >> templ.pyramid_level;
	int offset_y = min_y; // >> templ.pyramid_level;

	for (int j = 0; j < (int)templ.features.size(); ++j)
	{
		templ.features[j].x -= offset_x;
		templ.features[j].y -= offset_y;
	}

	box = cv::Rect(min_x, min_y, max_x - min_x, max_y - min_y);

	return true;
}

bool QuantisedGradOrientation::extractTemplateNoResize(Template& templ, float strong_threshold, 
													   size_t num_features, const cv::Mat& magnitude, 
													   const cv::Mat& angle, const cv::Mat& mask, cv::Rect& box)
{
	// Want features on the border to distinguish from background
	cv::Mat local_mask;
	if (!mask.empty())
	{
		erode(mask, local_mask, cv::Mat(), cv::Point(-1,-1), 1, cv::BORDER_REPLICATE);
		subtract(mask, local_mask, local_mask);
	}

	// Create sorted list of all pixels with magnitude greater than a threshold
	std::vector<Candidate> candidates;
	bool no_mask = local_mask.empty();
	float threshold_sq = SQR(strong_threshold);
	for (int r = 0; r < magnitude.rows; ++r)
	{
		const uchar* angle_r = angle.ptr<uchar>(r);
		const float* magnitude_r = magnitude.ptr<float>(r);
		const uchar* mask_r = no_mask ? NULL : local_mask.ptr<uchar>(r);

		for (int c = 0; c < magnitude.cols; ++c)
		{
			if (no_mask || mask_r[c])
			{
				uchar quantized = angle_r[c];
				if (quantized > 0)
				{
					float score = magnitude_r[c];
					if (score > threshold_sq)
					{
						candidates.push_back(Candidate(c, r, getLabel(quantized), score));
					}
				}
			}
		}
	}
	// We require a certain number of features
	if (candidates.size() < num_features)
		return false;
	// NOTE: Stable sort to agree with old code, which used std::list::sort()
	std::stable_sort(candidates.begin(), candidates.end());

	// Use heuristic based on surplus of candidates in narrow outline for initial distance threshold
	float distance = static_cast<float>(candidates.size() / num_features + 1);
	selectScatteredFeatures(candidates, templ.features, num_features, distance);

	templ.width = angle.cols;
	templ.height = angle.rows;
	templ.pyramid_level = 0; // harded-coded for now


	box = cv::Rect(0, 0, templ.width, templ.height);

	return true;
}

bool QuantisedGradOrientation::extractTemplateWithResize(Template& templ, float strong_threshold, 
														 size_t num_features, const cv::Mat& magnitude, 
														 const cv::Mat& angle, const cv::Mat& mask, cv::Rect& box, int ratio)
{
	// Want features on the border to distinguish from background
	cv::Mat local_mask;
	if (!mask.empty())
	{
		erode(mask, local_mask, cv::Mat(), cv::Point(-1,-1), 1, cv::BORDER_REPLICATE);
		subtract(mask, local_mask, local_mask);
	}

	// Create sorted list of all pixels with magnitude greater than a threshold
	std::vector<Candidate> candidates;
	bool no_mask = local_mask.empty();
	float threshold_sq = SQR(strong_threshold);
	for (int r = 0; r < magnitude.rows; ++r)
	{
		const uchar* angle_r = angle.ptr<uchar>(r);
		const float* magnitude_r = magnitude.ptr<float>(r);
		const uchar* mask_r = no_mask ? NULL : local_mask.ptr<uchar>(r);

		for (int c = 0; c < magnitude.cols; ++c)
		{
			if (no_mask || mask_r[c])
			{
				uchar quantized = angle_r[c];
				if (quantized > 0)
				{
					float score = magnitude_r[c];
					if (score > threshold_sq)
					{
						candidates.push_back(Candidate(c, r, getLabel(quantized), score));
					}
				}
			}
		}
	}
	// We require a certain number of features
	if (candidates.size() < num_features)
		return false;
	// NOTE: Stable sort to agree with old code, which used std::list::sort()
	std::stable_sort(candidates.begin(), candidates.end());

	// Use heuristic based on surplus of candidates in narrow outline for initial distance threshold
	float distance = static_cast<float>(candidates.size() / num_features + 1);
	selectScatteredFeatures(candidates, templ.features, num_features, distance);

	for (int i = 0; i < templ.features.size(); i++)
	{
		templ.features[i].x *= ratio;
		templ.features[i].y *= ratio;
	}
	templ.width = angle.cols * ratio;
	templ.height = angle.rows * ratio;
	templ.pyramid_level = 0; // harded-coded for now


	box = cv::Rect(0, 0, templ.width, templ.height);

	return true;
}


/****************************************************************************************\
*                               Response maps                                  *
\****************************************************************************************/
void QuantisedGradOrientation::orUnaligned8u(const uchar * src, const int src_stride,
											 uchar * dst, const int dst_stride,
											 const int width, const int height)
{
#if CV_SSE2
	volatile bool haveSSE2 = cv::checkHardwareSupport(CV_CPU_SSE2);
#if CV_SSE3
	volatile bool haveSSE3 = cv::checkHardwareSupport(CV_CPU_SSE3);
#endif
	bool src_aligned = reinterpret_cast<unsigned long long>(src) % 16 == 0;
#endif

	for (int r = 0; r < height; ++r)
	{
		int c = 0;

#if CV_SSE2
		// Use aligned loads if possible
		if (haveSSE2 && src_aligned)
		{
			for ( ; c < width - 15; c += 16)
			{
				const __m128i* src_ptr = reinterpret_cast<const __m128i*>(src + c);
				__m128i* dst_ptr = reinterpret_cast<__m128i*>(dst + c);
				*dst_ptr = _mm_or_si128(*dst_ptr, *src_ptr);
			}
		}
#if CV_SSE3
		// Use LDDQU for fast unaligned load
		else if (haveSSE3)
		{
			for ( ; c < width - 15; c += 16)
			{
				__m128i val = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(src + c));
				__m128i* dst_ptr = reinterpret_cast<__m128i*>(dst + c);
				*dst_ptr = _mm_or_si128(*dst_ptr, val);
			}
		}
#endif
		// Fall back to MOVDQU
		else if (haveSSE2)
		{
			for ( ; c < width - 15; c += 16)
			{
				__m128i val = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + c));
				__m128i* dst_ptr = reinterpret_cast<__m128i*>(dst + c);
				*dst_ptr = _mm_or_si128(*dst_ptr, val);
			}
		}
#endif
		for ( ; c < width; ++c)
			dst[c] |= src[c];

		// Advance to next row
		src += src_stride;
		dst += dst_stride;
	}
}


void QuantisedGradOrientation::spread(const cv::Mat& src, cv::Mat& dst, int T)
{
	// Allocate and zero-initialize spread (OR'ed) image
	dst = cv::Mat::zeros(src.size(), CV_8U);

	// Fill in spread gradient image (section 2.3)
	for (int r = 0; r < T; ++r)
	{
		int height = src.rows - r;
		for (int c = 0; c < T; ++c)
		{
			orUnaligned8u(&src.at<unsigned char>(r, c), static_cast<const int>(src.step1()), dst.ptr(),
				static_cast<const int>(dst.step1()), src.cols - c, height);
		}
	}
}

CV_DECL_ALIGNED(16) static const unsigned char SIMILARITY_LUT[256] = {0, 4, 3, 4, 2, 4, 3, 4, 1, 4, 3, 4, 2, 4, 3, 4, 0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 0, 3, 4, 4, 3, 3, 4, 4, 2, 3, 4, 4, 3, 3, 4, 4, 0, 1, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 3, 3, 4, 4, 4, 4, 3, 3, 3, 3, 4, 4, 4, 4, 0, 2, 1, 2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 0, 3, 2, 3, 1, 3, 2, 3, 0, 3, 2, 3, 1, 3, 2, 3, 0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 0, 4, 3, 4, 2, 4, 3, 4, 1, 4, 3, 4, 2, 4, 3, 4, 0, 1, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 0, 3, 4, 4, 3, 3, 4, 4, 2, 3, 4, 4, 3, 3, 4, 4, 0, 2, 1, 2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 2, 3, 3, 4, 4, 4, 4, 3, 3, 3, 3, 4, 4, 4, 4, 0, 3, 2, 3, 1, 3, 2, 3, 0, 3, 2, 3, 1, 3, 2, 3, 0, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4};

void QuantisedGradOrientation::computeResponseMaps(const cv::Mat& src, std::vector<cv::Mat>& response_maps)
{
	CV_Assert((src.rows * src.cols) % 16 == 0);

	// Allocate response maps
	response_maps.resize(8);
	for (int i = 0; i < 8; ++i)
		response_maps[i].create(src.size(), CV_8U);

	cv::Mat lsb4(src.size(), CV_8U);
	cv::Mat msb4(src.size(), CV_8U);

	for (int r = 0; r < src.rows; ++r)
	{
		const uchar* src_r = src.ptr(r);
		uchar* lsb4_r = lsb4.ptr(r);
		uchar* msb4_r = msb4.ptr(r);

		for (int c = 0; c < src.cols; ++c)
		{
			// Least significant 4 bits of spread image pixel
			lsb4_r[c] = src_r[c] & 15;
			// Most significant 4 bits, right-shifted to be in [0, 16)
			msb4_r[c] = (src_r[c] & 240) >> 4;
		}
	}

#if CV_SSSE3
	volatile bool haveSSSE3 = cv::checkHardwareSupport(CV_CPU_SSSE3);
	if (haveSSSE3)
	{
		const __m128i* lut = reinterpret_cast<const __m128i*>(SIMILARITY_LUT);
		for (int ori = 0; ori < 8; ++ori)
		{
			__m128i* map_data = response_maps[ori].ptr<__m128i>();
			__m128i* lsb4_data = lsb4.ptr<__m128i>();
			__m128i* msb4_data = msb4.ptr<__m128i>();

			// Precompute the 2D response map S_i (section 2.4)
			for (int i = 0; i < (src.rows * src.cols) / 16; ++i)
			{
				// Using SSE shuffle for table lookup on 4 orientations at a time
				// The most/least significant 4 bits are used as the LUT index
				__m128i res1 = _mm_shuffle_epi8(lut[2*ori + 0], lsb4_data[i]);
				__m128i res2 = _mm_shuffle_epi8(lut[2*ori + 1], msb4_data[i]);

				// Combine the results into a single similarity score
				map_data[i] = _mm_max_epu8(res1, res2);
			}
		}
	}
	else
#endif
	{
		// For each of the 8 quantized orientations...
		for (int ori = 0; ori < 8; ++ori)
		{
			uchar* map_data = response_maps[ori].ptr<uchar>();
			uchar* lsb4_data = lsb4.ptr<uchar>();
			uchar* msb4_data = msb4.ptr<uchar>();
			const uchar* lut_low = SIMILARITY_LUT + 32*ori;
			const uchar* lut_hi = lut_low + 16;

			for (int i = 0; i < src.rows * src.cols; ++i)
			{
				map_data[i] = std::max(lut_low[ lsb4_data[i] ], lut_hi[ msb4_data[i] ]);
			}
		}
	}
}

/****************************************************************************************\
*                               Linearized similarities                                  *
\****************************************************************************************/

void QuantisedGradOrientation::linearize(const cv::Mat& response_map, cv::Mat& linearized, int T)
{
	CV_Assert(response_map.rows % T == 0);
	CV_Assert(response_map.cols % T == 0);

	// linearized has T^2 rows, where each row is a linear memory
	int mem_width = response_map.cols / T;
	int mem_height = response_map.rows / T;
	linearized.create(T*T, mem_width * mem_height, CV_8U);

	// Outer two for loops iterate over top-left T^2 starting pixels
	int index = 0;
	for (int r_start = 0; r_start < T; ++r_start)
	{
		for (int c_start = 0; c_start < T; ++c_start)
		{
			uchar* memory = linearized.ptr(index);
			++index;

			// Inner two loops copy every T-th pixel into the linear memory
			for (int r = r_start; r < response_map.rows; r += T)
			{
				const uchar* response_data = response_map.ptr(r);
				for (int c = c_start; c < response_map.cols; c += T)
					*memory++ = response_data[c];
			}
		}
	}
}

const unsigned char* QuantisedGradOrientation::accessLinearMemory(const std::vector<cv::Mat>& linear_memories,
																  const Feature& f, int T, int W)
{
	// Retrieve the TxT grid of linear memories associated with the feature label
	const cv::Mat& memory_grid = linear_memories[f.label];
	CV_DbgAssert(memory_grid.rows == T*T);
	CV_DbgAssert(f.x >= 0);
	CV_DbgAssert(f.y >= 0);
	// The LM we want is at (x%T, y%T) in the TxT grid (stored as the rows of memory_grid)
	int grid_x = f.x % T;
	int grid_y = f.y % T;
	int grid_index = grid_y * T + grid_x;
	CV_DbgAssert(grid_index >= 0);
	CV_DbgAssert(grid_index < memory_grid.rows);
	const unsigned char* memory = memory_grid.ptr(grid_index);
	// Within the LM, the feature is at (x/T, y/T). W is the "width" of the LM, the
	// input image width decimated by T.
	int lm_x = f.x / T;
	int lm_y = f.y / T;
	int lm_index = lm_y * W + lm_x;
	CV_DbgAssert(lm_index >= 0);
	CV_DbgAssert(lm_index < memory_grid.cols);
	return memory + lm_index;
}

void QuantisedGradOrientation::similarity(const std::vector<cv::Mat>& linear_memories, const Template& templ,
										  cv::Mat& dst, cv::Size size, int T)
{
	// 63 features or less is a special case because the max similarity per-feature is 4.
	// 255/4 = 63, so up to that many we can add up similarities in 8 bits without worrying
	// about overflow. Therefore here we use _mm_add_epi8 as the workhorse, whereas a more
	// general function would use _mm_add_epi16.
	CV_Assert(templ.features.size() <= 63);
	/// @todo Handle more than 255/MAX_RESPONSE features!!

	// Decimate input image size by factor of T
	int W = size.width / T;
	int H = size.height / T;

	// Feature dimensions, decimated by factor T and rounded up
	int wf = (templ.width - 1) / T + 1;
	int hf = (templ.height - 1) / T + 1;

	// Span is the range over which we can shift the template around the input image
	int span_x = W - wf;
	int span_y = H - hf;

	// Compute number of contiguous (in memory) pixels to check when sliding feature over
	// image. This allows template to wrap around left/right border incorrectly, so any
	// wrapped template matches must be filtered out!
	int template_positions = span_y * W + span_x + 1; // why add 1?
	//int template_positions = (span_y - 1) * W + span_x; // More correct?

	/// @todo In old code, dst is buffer of size m_U. Could make it something like
	/// (span_x)x(span_y) instead?
	dst = cv::Mat::zeros(H, W, CV_8U);
	uchar* dst_ptr = dst.ptr<uchar>();

#if CV_SSE2
	volatile bool haveSSE2 = cv::checkHardwareSupport(CV_CPU_SSE2);
#if CV_SSE3
	volatile bool haveSSE3 = cv::checkHardwareSupport(CV_CPU_SSE3);
#endif
#endif

	// Compute the similarity measure for this template by accumulating the contribution of
	// each feature
	for (int i = 0; i < (int)templ.features.size(); ++i)
	{
		// Add the linear memory at the appropriate offset computed from the location of
		// the feature in the template
		Feature f = templ.features[i];
		// Discard feature if out of bounds
		/// @todo Shouldn't actually see x or y < 0 here?
		if (f.x < 0 || f.x >= size.width || f.y < 0 || f.y >= size.height)
			continue;
		const uchar* lm_ptr = accessLinearMemory(linear_memories, f, T, W);

		// Now we do an aligned/unaligned add of dst_ptr and lm_ptr with template_positions elements
		int j = 0;
		// Process responses 16 at a time if vectorization possible
#if CV_SSE2
#if CV_SSE3
		if (haveSSE3)
		{
			// LDDQU may be more efficient than MOVDQU for unaligned load of next 16 responses
			for ( ; j < template_positions - 15; j += 16)
			{
				__m128i responses = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(lm_ptr + j));
				__m128i* dst_ptr_sse = reinterpret_cast<__m128i*>(dst_ptr + j);
				*dst_ptr_sse = _mm_add_epi8(*dst_ptr_sse, responses);
			}
		}
		else
#endif
			if (haveSSE2)
			{
				// Fall back to MOVDQU
				for ( ; j < template_positions - 15; j += 16)
				{
					__m128i responses = _mm_loadu_si128(reinterpret_cast<const __m128i*>(lm_ptr + j));
					__m128i* dst_ptr_sse = reinterpret_cast<__m128i*>(dst_ptr + j);
					*dst_ptr_sse = _mm_add_epi8(*dst_ptr_sse, responses);
				}
			}
#endif
			for ( ; j < template_positions; ++j)
				dst_ptr[j] = uchar(dst_ptr[j] + lm_ptr[j]);
	}
}

void QuantisedGradOrientation::similarityLocal(const std::vector<cv::Mat>& linear_memories,
											   const Template& templ,
											   cv::Mat& dst, cv::Size size, int T, cv::Point center)
{
	// Similar to whole-image similarity() above. This version takes a position 'center'
	// and computes the energy in the 16x16 patch centered on it.
	CV_Assert(templ.features.size() <= 63);

	// Compute the similarity map in a 16x16 patch around center
	int W = size.width / T;
	dst = cv::Mat::zeros(16, 16, CV_8U);

	// Offset each feature point by the requested center. Further adjust to (-8,-8) from the
	// center to get the top-left corner of the 16x16 patch.
	// NOTE: We make the offsets multiples of T to agree with results of the original code.
	int offset_x = (center.x / T - 8) * T;
	int offset_y = (center.y / T - 8) * T;

#if CV_SSE2
	volatile bool haveSSE2 = cv::checkHardwareSupport(CV_CPU_SSE2);
#if CV_SSE3
	volatile bool haveSSE3 = cv::checkHardwareSupport(CV_CPU_SSE3);
#endif
	__m128i* dst_ptr_sse = dst.ptr<__m128i>();
#endif

	for (int i = 0; i < (int)templ.features.size(); ++i)
	{
		Feature f = templ.features[i];
		f.x += offset_x;
		f.y += offset_y;
		// Discard feature if out of bounds, possibly due to applying the offset
		if (f.x < 0 || f.y < 0 || f.x >= size.width || f.y >= size.height)
			continue;

		const uchar* lm_ptr = accessLinearMemory(linear_memories, f, T, W);

		// Process whole row at a time if vectorization possible
#if CV_SSE2
#if CV_SSE3
		if (haveSSE3)
		{
			// LDDQU may be more efficient than MOVDQU for unaligned load of 16 responses from current row
			for (int row = 0; row < 16; ++row)
			{
				__m128i aligned = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(lm_ptr));
				dst_ptr_sse[row] = _mm_add_epi8(dst_ptr_sse[row], aligned);
				lm_ptr += W; // Step to next row
			}
		}
		else
#endif
			if (haveSSE2)
			{
				// Fall back to MOVDQU
				for (int row = 0; row < 16; ++row)
				{
					__m128i aligned = _mm_loadu_si128(reinterpret_cast<const __m128i*>(lm_ptr));
					dst_ptr_sse[row] = _mm_add_epi8(dst_ptr_sse[row], aligned);
					lm_ptr += W; // Step to next row
				}
			}
			else
#endif
			{
				uchar* dst_ptr = dst.ptr<uchar>();
				for (int row = 0; row < 16; ++row)
				{
					for (int col = 0; col < 16; ++col)
						dst_ptr[col] = uchar(dst_ptr[col] + lm_ptr[col]);
					dst_ptr += 16;
					lm_ptr += W;
				}
			}
	}
}

void QuantisedGradOrientation::addUnaligned8u16u(const uchar * src1, const uchar * src2,
												 ushort * res, int length)
{
	const uchar * end = src1 + length;

	while (src1 != end)
	{
		*res = *src1 + *src2;

		++src1;
		++src2;
		++res;
	}
}
void QuantisedGradOrientation::addSimilarities(const std::vector<cv::Mat>& similarities, cv::Mat& dst)
{
	if (similarities.size() == 1)
	{
		similarities[0].convertTo(dst, CV_16U);
	}
	else
	{
		// NOTE: add() seems to be rather slow in the 8U + 8U -> 16U case
		dst.create(similarities[0].size(), CV_16U);
		addUnaligned8u16u(similarities[0].ptr(), similarities[1].ptr(), dst.ptr<ushort>(), static_cast<int>(dst.total()));

		/// @todo Optimize 16u + 8u -> 16u when more than 2 modalities
		for (size_t i = 2; i < similarities.size(); ++i)
			cv::add(dst, similarities[i], dst, cv::noArray(), CV_16U);
	}
}
