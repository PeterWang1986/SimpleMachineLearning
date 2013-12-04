
#include "gtest\gtest.h"
#include "sml\Regression\LinearRegressionBuilder.h"
#include "sml\Common\SmlSequenceSelection.h"

namespace SML {

	TEST(LinearRegressionBuilder, training)
	{
		const double data[8] = {0, 0, 0, 1, 1, 0, 1, 1};
		const double target[4] = {0.7, 1, 0.9, 1.2};
		const double expected[3] = {0.7, 0.2, 0.3};

		LinearRegressionBuilder lrBuilder;
		lrBuilder.training(4, 2, data, target, 0.0);

		const LinearRegressionModel& lrModel = lrBuilder.getLinearRegressionModel();
		for (size_t i=0; i!=3; ++i) {
			EXPECT_NEAR(expected[i], lrModel.getWeights(i), 0.000001);
		}
	}

	TEST(LinearRegressionBuilder, sequence_training)
	{
		const double data[8] = {0, 0, 0, 1, 1, 0, 1, 1};
		const double target[4] = {0.7, 1, 0.9, 1.2};
		const double expected[3] = {0.7, 0.2, 0.3};

		LinearRegressionBuilder lrBuilder;
		lrBuilder.sequence_training<ForwardSelection>(4, 2, data, target, 0.005, 100, 3000);

		const LinearRegressionModel& lrModel = lrBuilder.getLinearRegressionModel();
		for (size_t i=0; i!=3; ++i) {
			EXPECT_NEAR(expected[i], lrModel.getWeights(i), 0.001);
		}
	}



}	//end of SML