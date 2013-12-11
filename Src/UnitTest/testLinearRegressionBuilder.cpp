
#include "gtest/gtest.h"

#include "sml/Regression/LinearRegressionBuilder.h"
#include "sml/Regression/BayesianLinearRegressionBuilder.h"
#include "sml/Regression/ValidateRegression.h"
#include "sml/Common/SmlSequenceSelection.h"

#include <vector>

namespace SML {
	using std::vector;
	using std::cout;
	using std::endl;

	/**
	* y = sin(x), error ~ N(0, 0.4)
	**/
	static const double x_value[20] = {0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5,
									   5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5};
	static const double y_value[20] = {0.00000000, 0.47942554, 0.84147098, 0.99749499, 0.90929743,
									   0.59847214, 0.14112001, -0.35078323, -0.75680250, -0.97753012,
									   -0.95892427, -0.70554033, -0.27941550, 0.21511999, 0.65698660,
									   0.93799998, 0.98935825, 0.79848711, 0.41211849, -0.07515112};
	static const double error[20] = {-0.6305681, 0.2702860, -0.2275811, 0.1804783, -0.6452688,
									0.2609742, -0.3808075, 0.5651586, 0.2827417, -0.4981668,
									-0.4973483, -0.1696667, -0.1234159, -0.3372082, -0.3428417,
									-0.1981684, 0.6792128, -0.2458879, 0.1530129, -0.1759507};

	TEST(LinearRegressionBuilder, training)
	{
		const double data[8] = {0, 0, 0, 1, 1, 0, 1, 1};
		const double target[4] = {0.7, 1, 0.9, 1.2};
		const double expected[3] = {0.7, 0.2, 0.3};

		LinearRegressionBuilder lrBuilder;
		lrBuilder.training(4, 2, data, target, 0.0);

		const LinearRegressionModel& lrModel = lrBuilder.getLinearRegressionModel();
		for (size_t i=0; i!=3; ++i) {
			EXPECT_NEAR(expected[i], lrModel.getWeights(i), 0.00001);
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

	TEST(LinearRegressionBuilder, training02)
	{
		LinearRegressionBuilder lrBuilder;
		lrBuilder.addBiasFunction();
		lrBuilder.addSingleVarGussian(0, 0, 1);
		lrBuilder.addSingleVarGussian(0, 1, 2);
		lrBuilder.addSingleVarGussian(0, 8, 4);

		lrBuilder.training(20, 1, x_value, y_value, 0.0);

		const LinearRegressionModel& lrModel = lrBuilder.getLinearRegressionModel();
		vector<double> vecPredValues(20, 0.0);
		for (size_t i=0; i!=20; ++i) {
			vecPredValues[i] = lrModel.forecast(x_value+i, 1);
		}
			
		EXPECT_NEAR(-1.49184, lrModel.getWeights(0), 0.00001);
		EXPECT_NEAR(-1.28299, lrModel.getWeights(1), 0.00001);
		EXPECT_NEAR(3.31208, lrModel.getWeights(2), 0.00001);
		EXPECT_NEAR(2.27591, lrModel.getWeights(3), 0.00001);
		
		RsquaredValidate v;
		const double rSquare = v.validate(y_value, y_value+20, vecPredValues.begin());
		EXPECT_NEAR(0.910849, rSquare, 0.00001);
	}

	TEST(LinearRegressionBuilder, training03)
	{
		LinearRegressionBuilder lrBuilder;
		lrBuilder.addBiasFunction();
		lrBuilder.addSingleVarGussian(0, 0, 1);
		lrBuilder.addSingleVarGussian(0, 1, 2);
		lrBuilder.addSingleVarGussian(0, 8, 4);
		lrBuilder.addSingleVarGussian(0, 2, 4);
		lrBuilder.addSingleVarGussian(0, 4, 8);

		lrBuilder.training(20, 1, x_value, y_value, 0.0);

		const LinearRegressionModel& lrModel = lrBuilder.getLinearRegressionModel();
		vector<double> vecTarget(20, 0.0);
		vector<double> vecPredValues(20, 0.0);
		double x = 0.25;
		for (size_t i=0; i!=20; ++i) {
			vecTarget[i] = std::sin(x);
			vecPredValues[i] = lrModel.forecast(&x, 1);
			x += 0.5;
		}
		
		RsquaredValidate v;
		const double rSquare = v.validate(vecTarget.begin(), vecTarget.end(), vecPredValues.begin());
		cout << "R-square=\t" << rSquare << endl;
		//EXPECT_NEAR(0.910849, rSquare, 0.00001);
	}

	TEST(LinearRegressionBuilder, training04)
	{
		LinearRegressionBuilder lrBuilder;
		lrBuilder.addBiasFunction();
		lrBuilder.addSingleVarGussian(0, 0, 1);
		lrBuilder.addSingleVarGussian(0, 1, 2);
		lrBuilder.addSingleVarGussian(0, 2, 4);
		lrBuilder.addSingleVarGussian(0, 8, 4);

		double target[20];
		for (size_t i=0; i!=20; ++i) {
			target[i] = y_value[i] + error[i];
		}
		lrBuilder.training(20, 1, x_value, target, 0.0);

		const LinearRegressionModel& lrModel = lrBuilder.getLinearRegressionModel();
		vector<double> vecPredValues(20, 0.0);
		for (size_t i=0; i!=20; ++i) {
			vecPredValues[i] = lrModel.forecast(x_value+i, 1);
		}
		
		RsquaredValidate v;
		const double rSquare = v.validate(target, target+20, vecPredValues.begin());
		cout << "R-square=\t" << rSquare << endl;
		//EXPECT_NEAR(0.910849, rSquare, 0.00001);
	}

	TEST(LinearRegressionBuilder, sequence_training02)
	{
		LinearRegressionBuilder lrBuilder;
		lrBuilder.addBiasFunction();
		lrBuilder.addSingleVarGussian(0, 0, 1);
		lrBuilder.addSingleVarGussian(0, 1, 2);
		lrBuilder.addSingleVarGussian(0, 8, 4);

		lrBuilder.sequence_training<ForwardSelection>(20, 1, x_value, y_value, 0.005, 100, 2000);

		const LinearRegressionModel& lrModel = lrBuilder.getLinearRegressionModel();
		vector<double> vecPredValues(20, 0.0);
		for (size_t i=0; i!=20; ++i) {
			vecPredValues[i] = lrModel.forecast(x_value+i, 1);
		}

		EXPECT_NEAR(-1.44226, lrModel.getWeights(0), 0.00001);
		EXPECT_NEAR(-1.17976, lrModel.getWeights(1), 0.00001);
		EXPECT_NEAR(3.17949, lrModel.getWeights(2), 0.00001);
		EXPECT_NEAR(2.21056, lrModel.getWeights(3), 0.00001);
		
		RsquaredValidate v;
		const double rSquare = v.validate(y_value, y_value+20, vecPredValues.begin());
		EXPECT_NEAR(0.909608, rSquare, 0.00001);
	}

	TEST(LinearRegressionBuilder, sequence_training03)
	{
		LinearRegressionBuilder lrBuilder;
		lrBuilder.addBiasFunction();
		lrBuilder.addSingleVarGussian(0, 0, 1);
		lrBuilder.addSingleVarGussian(0, 1, 2);
		lrBuilder.addSingleVarGussian(0, 8, 4);

		lrBuilder.sequence_training<ForwardSelection>(20, 1, x_value, y_value, 0.005, 100, 2000);

		const LinearRegressionModel& lrModel = lrBuilder.getLinearRegressionModel();
		vector<double> vecTarget(20, 0.0);
		vector<double> vecPredValues(20, 0.0);
		double x = 9.25;
		for (size_t i=0; i!=20; ++i) {
			vecTarget[i] = std::sin(x);
			vecPredValues[i] = lrModel.forecast(&x, 1);
			x += 0.5;
		}
		
		RsquaredValidate v;
		const double rSquare = v.validate(y_value, y_value+20, vecPredValues.begin());
		cout << "R-square=\t" << rSquare << endl;
		//EXPECT_NEAR(0.909608, rSquare, 0.00001);
	}

	TEST(BayesianLrRegressionBuilder, training01)
	{
		LinearRegressionBuilder lrBuilder;
		lrBuilder.addBiasFunction();
		lrBuilder.addSingleVarGussian(0, 0, 1);
		lrBuilder.addSingleVarGussian(0, 1, 0.5);
		lrBuilder.addSingleVarGussian(0, 2, 4);
		lrBuilder.addSingleVarGussian(0, 3, 1.5);
		lrBuilder.addSingleVarGussian(0, 4, 8);
		lrBuilder.addSingleVarGussian(0, 5, 2.5);
		lrBuilder.addSingleVarGussian(0, 6, 8);
		lrBuilder.addSingleVarGussian(0, 7, 3.5);
		lrBuilder.addSingleVarGussian(0, 8, 4);

		BayesianLrRegressionBuilder bayesLrBuilder(1, 20);
		bayesLrBuilder.addBiasFunction();
		bayesLrBuilder.addSingleVarGussian(0, 0, 1);
		bayesLrBuilder.addSingleVarGussian(0, 1, 0.5);
		bayesLrBuilder.addSingleVarGussian(0, 2, 4);
		bayesLrBuilder.addSingleVarGussian(0, 3, 1.5);
		bayesLrBuilder.addSingleVarGussian(0, 4, 8);
		bayesLrBuilder.addSingleVarGussian(0, 5, 2.5);
		bayesLrBuilder.addSingleVarGussian(0, 6, 8);
		bayesLrBuilder.addSingleVarGussian(0, 7, 3.5);
		bayesLrBuilder.addSingleVarGussian(0, 8, 4);

		double target[20];
		for (size_t i=0; i!=20; ++i) {
			target[i] = y_value[i] + error[i];
		}
		lrBuilder.training(20, 1, x_value, target, 0.0);
		bayesLrBuilder.training(20, 1, x_value, target);

		vector<double> lrPred(20, 0), bayesLrPred(20, 0);
		const LinearRegressionModel& lrModel = lrBuilder.getLinearRegressionModel();
		const BayesianLrRegressionModel& bayesLrModel = bayesLrBuilder.getBayesianLinearRegressionModel();
		for (size_t i=0; i!=20; ++i) {
			lrPred[i] = lrModel.forecast(x_value+i, 1);
			bayesLrPred[i] = bayesLrModel.forecast(x_value+i, 1);
		}
		
		RsquaredValidate v;
		const double lrTrainingRSquare = v.validate(target, target+20, lrPred.begin());
		const double bayesLrTrainingRSquare = v.validate(target, target+20, bayesLrPred.begin());
		cout << "lrTrainingRSquare=\t" << lrTrainingRSquare << endl;
		cout << "bayesLrTrainingRSquare=\t" << bayesLrTrainingRSquare << endl;

		vector<double> vecTarget(20, 0);
		double x = 0.25;
		for (size_t i=0; i!=20; ++i) {
			lrPred[i] = lrModel.forecast(&x, 1);
			bayesLrPred[i] = bayesLrModel.forecast(&x, 1);
			vecTarget[i] = std::sin(x);
			x += 0.5;
		}
		const double lrTestRSquare = v.validate(vecTarget.begin(), vecTarget.end(), lrPred.begin());
		const double bayesLrTestRSquare = v.validate(vecTarget.begin(), vecTarget.end(), bayesLrPred.begin());
		cout << "lrTestRSquare=\t" << lrTestRSquare << endl;
		cout << "bayesLrTestRSquare=\t" << bayesLrTestRSquare << endl;
	}


}	//end of SML