
#include "gtest/gtest.h"

#include "sml/Classification/LogisticRegression/BinaryLogisticRegression.h"

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

		/*double rate = 0.0;
		size_t max = 100;
		cout << "please input learning rate:\t";
		std::cin >> rate;
		cout << "please input max iteration:\t";
		std::cin >> max;
		cout << "OK, learning rate is: " << rate << " and max iteration is: " << max << endl;
		*/
		BiLogisticRegression biLogisticRegression;
		biLogisticRegression.stochasticGradientTraining(16, 3, &(vecTrainingData[0]), label1, 4.0, 5000);

		/*const vector<double>& weights = biLogisticRegression.getWeights();
		for (size_t i=0; i!=weights.size(); ++i)
		{
			cout << "weights[" << i << "]=\t" << weights[i] << endl;
		}
		*/
	}


}	//end of SML