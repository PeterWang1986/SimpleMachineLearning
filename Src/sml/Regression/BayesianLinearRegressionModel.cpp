
#include "sml/Regression/BayesianLinearRegressionModel.h"

#include "eigen/Core"

namespace SML {

	BayesianLrRegressionModel::BayesianLrRegressionModel(double a,
														double b)
														: m_fAlpha(a)
														, m_fBeta(b)
	{
	}

	BayesianLrRegressionModel::~BayesianLrRegressionModel()
	{
	}

	bool BayesianLrRegressionModel::setGaussianMean(const Eigen::VectorXd& m)
	{
		const size_t n = this->getNumBasisFunctions();
		if (m.rows() != n) return false;

		for (size_t i=0; i!=n; ++i) {
			m_lrModel.setWeights(i, m(i));
		}

		m_Means = m;
		return true;
	}

	bool BayesianLrRegressionModel::setGaussianVarianceMatrix(const Eigen::MatrixXd& s)
	{
		if (s.rows() != getNumBasisFunctions()) return false;

		m_CovarianceMatrix = s;
		return (m_CovarianceMatrix.rows() == m_CovarianceMatrix.cols());
	}

}	//end of SML