
#include "gtest/gtest.h"
#include "sml/Regression/BasisFunctions.h"

TEST(GussianBasisFunction, phi)
{
	const double PI = 3.1415926;
	const int dimension = 2;
	const double mean[dimension] = {1, 2};
	const double covariance[] = {1, 0, 0, 4};
	const double data[] = {0, 0};
	const double expected = 1/exp((double)1);

	SML::GussianBasisFunction gbf(dimension, mean, covariance);
	EXPECT_NEAR(expected, gbf.phi(data, dimension), 0.000001);
}

TEST(SigmodBasisFunction, phi)
{
	const double u = 0, s = 1;
	const double data[] = {0, 1};
	const double expected[] = {0.5, 1/(1+exp((double)-1))};

	SML::SigmodBasisFunction sbf(0, u, s);
	EXPECT_NEAR(expected[0], sbf.phi(data, 1), 0.000001);
	EXPECT_NEAR(expected[1], sbf.phi(data + 1, 1), 0.000001);
}