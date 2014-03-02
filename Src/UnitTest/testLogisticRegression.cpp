
#include "gtest/gtest.h"
#include "sml/Classification/LogisticRegression/BinaryLogisticRegression.h"

#include <algorithm>

namespace SML {
	using std::vector;
	using std::cout;
	using std::endl;

	static const double data1[] = {2.5, 1.0, 3.0, 1.5, 3.1, 2.1, 4.1, 1.9, 
								   3.7, 2.0, 5.0, 1.0, 5.0, 1.7, 2.7, 1.9,
								   1.1, 2.9, 1.4, 3.0, 2.0, 4.2, 2.9, 4.1,
								   2.8, 5.0, 2.1, 3.7, 1.7, 3.9, 1.8, 3.1};
	static const int label1[] = {0, 0, 0, 0, 0, 0, 0, 0,
								 1, 1, 1, 1, 1, 1, 1, 1};

	static const double data2[64] = {110, 2.62, 110, 2.875, 93, 2.32, 110, 3.215, 175, 3.44,
									105, 3.46, 245, 3.57, 62, 3.19, 95, 3.15, 123, 3.44,
									123, 3.44, 180, 4.07, 180, 3.73, 180, 3.78, 205, 5.25,
									215, 5.424, 230, 5.345, 66, 2.2, 52, 1.615, 65, 1.835,
									97,	2.465, 150,	3.52, 150, 3.435, 245, 3.84, 175, 3.845,
									66,	1.935, 91, 2.14, 113, 1.513, 264, 3.17, 175, 2.77,
									335, 3.57, 109,	2.78};
	static const int label2[32] = {1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
									1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1};


	TEST(BiLogisticRegression, stochasticGradientTraining)
	{
		vector<double> vecTrainingData;
		vecTrainingData.reserve(48);
		for (size_t i=0, j=0; i!=48; ++i)
		{
			if (i % 3 == 0)
			{
				vecTrainingData.push_back(1);
			}
			else
			{
				vecTrainingData.push_back(data1[j]);
				++j;
			}
		}

		BiLogisticRegression biLogisticRegression;
		//biLogisticRegression.stochasticGradientTraining(16, 3, &(vecTrainingData[0]), label1, 4.0, 5000);

		/**/
		const vector<double>& weights = biLogisticRegression.getWeights();
		for (size_t i=0; i!=weights.size(); ++i)
		{
			cout << "weights[" << i << "]=\t" << weights[i] << endl;
		}	
	}

	TEST(BiLogisticRegression, vlSGD1)
	{
		vector<double> vecTrainingData;
		vecTrainingData.reserve(48);
		for (size_t i=0, j=0; i!=48; ++i)
		{
			if (i % 3 == 0)
			{
				vecTrainingData.push_back(1);
			}
			else
			{
				vecTrainingData.push_back(data1[j]);
				++j;
			}
		}

		BiLogisticRegression biLogisticRegression;
		//biLogisticRegression.vlSGD(16, 3, &(vecTrainingData[0]), label1, 500);

		const vector<double>& weights = biLogisticRegression.getWeights();
		for (size_t i=0; i!=weights.size(); ++i)
		{
			cout << "weights[" << i << "]=\t" << weights[i] << endl;
		}	
	}

	TEST(BiLogisticRegression, vlSGD2)
	{
		const size_t rows = 32, cols = 2;
		const size_t len = rows * (cols + 1);
		vector<double> vecTrainingData;
		vecTrainingData.reserve(len);
		for (size_t i=0; i!=rows; ++i)
		{
			vecTrainingData.push_back(1);
			vecTrainingData.push_back(data2[2*i]);
			vecTrainingData.push_back(data2[2*i+1]);
		}

		BiLogisticRegression biLogisticRegression;
		biLogisticRegression.vlSGD(rows, (cols+1), &(vecTrainingData[0]), label2, 50000);

		const vector<double>& weights = biLogisticRegression.getWeights();
		for (size_t i=0; i!=weights.size(); ++i)
		{
			cout << "weights[" << i << "]=\t" << weights[i] << endl;
		}	
	}


	TEST(BiLogisticRegression, vgSGD1)
	{
		const size_t rows = 32, cols = 2;
		const size_t len = rows * (cols + 1);
		vector<double> vecTrainingData;
		vecTrainingData.reserve(len);
		for (size_t i=0; i!=rows; ++i)
		{
			vecTrainingData.push_back(1);
			vecTrainingData.push_back(data2[2*i]);
			vecTrainingData.push_back(data2[2*i+1]);
		}

		BiLogisticRegression biLogisticRegression;
		biLogisticRegression.vgSGD(rows, (cols+1), &(vecTrainingData[0]), label2, 50000);

		const vector<double>& weights = biLogisticRegression.getWeights();
		for (size_t i=0; i!=weights.size(); ++i)
		{
			cout << "weights[" << i << "]=\t" << weights[i] << endl;
		}	
	}


}	//end of SML