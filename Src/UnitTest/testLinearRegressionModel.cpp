
#include "gtest\gtest.h"
#include "sml\Regression\BasisFunctions.h"
#include "sml\Regression\LinearRegressionModel.h"

namespace SML {

	TEST(LinearRegressionModel, calcDesignMatrix)
	{
		const double gMean[4] = {0, 0, 0, 0};
		const double gVariance[16] = {1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4};
		const double data[16] = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3};
		const double expected[20] = {1, 1, 1, 1, 0, 1, 2, 3, 0.5, 0.731059, 0.880797,
									0.952574, 1, 0.606531, 0.135335, 0.011109, 1,
									0.352866, 0.0155039, 8.48182e-5};
		double designMatrix[20];

		LinearRegressionModel lrModel;
		lrModel.addBasisFunction(new BiasFunction(1.0));
		lrModel.addBasisFunction(new LinearBasisFunction(0));
		lrModel.addBasisFunction(new SigmodBasisFunction(1, 0, 1));
		lrModel.addBasisFunction(new SingleVarGussian(2, 0, 1));
		lrModel.addBasisFunction(new GussianBasisFunction(4, gMean, gVariance));
		lrModel.calcDesignMatrix(4, 5, 4, data, designMatrix);

		for (int i=0; i!=20; ++i)
		{
			EXPECT_NEAR(expected[i], designMatrix[i], 0.000001);
		}
	}



}	//end of SML