
#include "sml/Regression/LinearRegressionBuilder.h"
#include "sml/Regression/BasisFunctions.h"

#include "eigen/Core"
#include "eigen/SVD"

#include <numeric>

namespace SML {

	using std::vector;

	using Eigen::MatrixXd;
	using Eigen::VectorXd;
	using Eigen::Map;
	using Eigen::JacobiSVD;

	LinearRegressionBuilder::LinearRegressionBuilder()
	{
	}

	LinearRegressionBuilder::~LinearRegressionBuilder()
	{
	}

	void LinearRegressionBuilder::training(size_t rows, size_t cols,
										const double* data,
										const double* target,
										double alpha)
	{
		if (0 == m_model.getNumBasisFunctions()) {
			useDefaultBasisFunctions(cols);
		}
		const size_t DesignMatrixRows = rows;
		const size_t DesignMatrixCols = m_model.getNumBasisFunctions();

		MatrixXd design_matrix(DesignMatrixRows, DesignMatrixCols);
		m_model.calcDesignMatrix(DesignMatrixRows, DesignMatrixCols, cols, data, design_matrix.data());
		MatrixXd trans_matrix = design_matrix.transpose() * design_matrix;
		Eigen::Map<const VectorXd> vector_target(target, rows);
		VectorXd b = design_matrix.transpose() * vector_target;

		JacobiSVD<MatrixXd> jSVD(trans_matrix, Eigen::ComputeThinU | Eigen::ComputeThinV);
		VectorXd result = jSVD.solve(b);

		const double* pWeights = result.data();
		for (size_t i=0; i!=DesignMatrixCols; ++i) {
			m_model.setWeights(i, pWeights[i]);
		}
	}

	double LinearRegressionBuilder::dissimilarity(const vector<double>& v1,
												const vector<double>& v2) const
	{
		const size_t num = v1.size();
		double sum = 0.0;
		for (size_t i=0; i!=num; ++i)
		{
			const double diff = v1[i] - v2[i];
			sum += diff * diff;
		}

		return std::sqrt(sum);
	}

	void LinearRegressionBuilder::sequence_training(size_t dimension,
													const double* data,
													const double target,
													double alpha)
	{
		const size_t num = m_model.getNumBasisFunctions();
		vector<double> phiVector(num, 0.0);
		for (size_t i=0; i!=num; ++i)
		{
			const BasisFunctionBase* pFun = m_model.getBasisFunction(i);
			phiVector[i] = pFun->phi(data, dimension);
		}

		const vector<double>& weights = m_model.getWeights();
		double diff = std::inner_product(weights.begin(), weights.end(), phiVector.begin(), 0.0);
		diff -= target;
		diff *= alpha;

		for (size_t i=0; i!=num; ++i)
		{
			const double w = weights[i] - (diff * phiVector[i]);
			m_model.setWeights(i, w);
		}
	}

	void LinearRegressionBuilder::useDefaultBasisFunctions(size_t dimension)
	{
		m_model.addBasisFunction(new BiasFunction(1.0));
		for (size_t i=0; i!=dimension; ++i)
		{
			m_model.addBasisFunction(new LinearBasisFunction(i));
		}
	}

	void LinearRegressionBuilder::addBiasFunction()
	{
		addBasisFunction(new BiasFunction(1.0));
	}
	
	void LinearRegressionBuilder::addLinearBasisFunction(size_t pos)
	{
		addBasisFunction(new LinearBasisFunction(pos));
	}

	void LinearRegressionBuilder::addSigmodBasisFunction(size_t pos,
														double mean,
														double variance)
	{
		addBasisFunction(new SigmodBasisFunction(pos, mean, variance));
	}
		
	void LinearRegressionBuilder::addSingleVarGussian(size_t pos,
													double mean,
													double variance)
	{
		addBasisFunction(new SingleVarGussian(pos, mean, variance));
	}
		
	void LinearRegressionBuilder::addGussianBasisFunction(size_t dimension,
														const double* mean,
														const double* covariance)
	{
		addBasisFunction(new GussianBasisFunction(dimension, mean, covariance));
	}


}	//end of SML