
#include "sml\Regression\LinearRegressionBuilder.h"
#include "sml\Regression\BasisFunctions.h"

#include "eigen\Core"
#include "eigen\SVD"

namespace SML {

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

	void LinearRegressionBuilder::useDefaultBasisFunctions(size_t dimension)
	{
		m_model.addBasisFunction(new BiasFunction(1.0));
		for (size_t i=0; i!=dimension; ++i)
		{
			m_model.addBasisFunction(new LinearBasisFunction(i));
		}
	}


}	//end of SML