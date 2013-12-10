
#include "sml/Regression/BasisFunctions.h"
#include "eigen/LU"
#include <cmath>

namespace SML {
	using Eigen::MatrixXd;
	using Eigen::VectorXd;

	BasisFunctionBase::BasisFunctionBase()
	{
	}

	BasisFunctionBase::~BasisFunctionBase()
	{
	}

	BiasFunction::BiasFunction(double bias)
					: BasisFunctionBase()
					, m_fBias(bias)
	{
	}

	BiasFunction::~BiasFunction()
	{
	}

	LinearBasisFunction::LinearBasisFunction(size_t pos)
						: BasisFunctionBase()
						, m_iPosition(pos)
	{
	}

	LinearBasisFunction::~LinearBasisFunction()
	{
	}

	SigmodBasisFunction::SigmodBasisFunction(size_t pos,
											double mean,
											double variance)
											: BasisFunctionBase()
											, m_iPosition(pos)
											, m_fMean(mean)
											, m_fStdVariance(variance)
	{
	}

	SigmodBasisFunction::~SigmodBasisFunction()
	{
	}

	SingleVarGussian::SingleVarGussian(size_t pos,
									double mean,
									double variance)
									: BasisFunctionBase()
									, m_iPosition(pos)
									, m_fMean(mean)
									, m_fDenominator(0 - 2*variance)
	{
	}

	SingleVarGussian::~SingleVarGussian()
	{
	}

	GussianBasisFunction::GussianBasisFunction(size_t dimension,
											const double* mean,
											const double* covariance)
											: BasisFunctionBase()
											, m_Mean(dimension)
	{
		initialize(dimension, mean, covariance);
	}

	GussianBasisFunction::~GussianBasisFunction()
	{
	}

	double GussianBasisFunction::phi(const double* data, size_t len) const
	{
		VectorXd v(len);
		for (size_t i=0; i!=len; ++i) {
			v[i] = data[i];
		}
		v -= m_Mean;
		const double f = 0 - v.transpose() * m_PrecisionMatrix * v;

		return exp(0.5 * f);
	}

	void GussianBasisFunction::initialize(size_t dim, const double* mean, const double* covariance)
	{
		for (size_t i=0; i!=dim; ++i) {
			m_Mean[i] = mean[i];
		}

		MatrixXd covarianceMatrix(dim, dim);
		for (size_t i=0; i!=dim; ++i)
		{
			const size_t base = i * dim;
			for (size_t j=0; j!=dim; ++j) {
				covarianceMatrix(i, j) = covariance[base + j];
			}
		}

		m_PrecisionMatrix = covarianceMatrix.inverse();
	}

}	//end of SML