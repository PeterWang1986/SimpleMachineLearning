#ifndef	_BAYESIAN_LINEAR_REGRESSION_MODEL_H
#define _BAYESIAN_LINEAR_REGRESSION_MODEL_H

#include "sml/Regression/LinearRegressionModel.h"

#include "eigen/Core"

namespace SML {

	/**
	*assume the prior distribution of 'w' is N(w|0, (1/a)*I)
	*here 'w' is a vector, and a := m_fAlpha.
	*assume t(i) = y(i) + e(i), y(i) = tr(w)*Phi(x(i)), e(i) ~ N(0, 1/b)
	*here b := m_fBeta, using y(i) to predict t(i).

	*@para m_fAlpha				represent 'a'
	*@para m_fBeta				represent 'b'
	*@para m_Means				the mean of gaussian distribution
	*@para m_CovarianceMatrix	the covariance matrix of gaussian distribution
	*@para m_lrModel			be used to store the basis functions
	**/
	class BayesianLrRegressionModel
	{
	public:
		BayesianLrRegressionModel(double a, double b);
		~BayesianLrRegressionModel();

		inline void calcDesignMatrix(size_t rows,
									size_t cols,
									size_t dimension,
									const double* data,
									double* matrix)
		{
			m_lrModel.calcDesignMatrix(rows, cols, dimension, data, matrix);
		}
		inline double forecast(const double* data, size_t dimension) const
		{
			return m_lrModel.forecast(data, dimension);
		}

		inline void setAlpha(double a) { m_fAlpha = a; }
		inline void setBeta(double b) { m_fBeta = b; }
		inline void addBasisFunction(BasisFunctionBase* func) { m_lrModel.addBasisFunction(func); }
		inline bool setBasisFunction(size_t pos, BasisFunctionBase* func) { return m_lrModel.setBasisFunction(pos, func); }
		bool setGaussianMean(const Eigen::VectorXd& m);
		bool setGaussianVarianceMatrix(const Eigen::MatrixXd& s);

		inline double getAlpha() const { return m_fAlpha; }
		inline double getBeta() const { return m_fBeta; }
		inline size_t getNumBasisFunctions() const { return m_lrModel.getNumBasisFunctions(); }
		inline BasisFunctionBase* getBasisFunction(size_t pos) { return m_lrModel.getBasisFunction(pos); }
		inline const BasisFunctionBase* getBasisFunction(size_t pos) const { return m_lrModel.getBasisFunction(pos); }

	private:
		double					m_fAlpha;
		double					m_fBeta;
		Eigen::VectorXd			m_Means;
		Eigen::MatrixXd			m_CovarianceMatrix;
		LinearRegressionModel	m_lrModel;
	};

}	//end of SML

#endif