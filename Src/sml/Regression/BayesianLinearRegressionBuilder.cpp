
#include "sml/Regression/BayesianLinearRegressionBuilder.h"
#include "sml/Regression/BasisFunctions.h"

#include "eigen/Core"
#include "eigen/Dense"

namespace SML {
	using Eigen::VectorXd;
	using Eigen::MatrixXd;
	using Eigen::DiagonalMatrix;

	BayesianLrRegressionBuilder::BayesianLrRegressionBuilder() : m_model(1, 2)
	{
	}

	BayesianLrRegressionBuilder::BayesianLrRegressionBuilder(double a,
															double b)
															: m_model(a, b)
	{
	}

	BayesianLrRegressionBuilder::~BayesianLrRegressionBuilder()
	{
	}

	void BayesianLrRegressionBuilder::training(size_t rows,
											size_t cols,
											const double* data,
											const double* target)
	{
		if (0 == m_model.getNumBasisFunctions()) {
			useDefaultBasisFunctions(cols);
		}

		const size_t DesignMatrixRows = rows;
		const size_t DesignMatrixCols = m_model.getNumBasisFunctions();
		MatrixXd design_matrix(DesignMatrixRows, DesignMatrixCols);
		m_model.calcDesignMatrix(DesignMatrixRows, DesignMatrixCols, cols, data, design_matrix.data());
		MatrixXd trDeisgnMatrix(design_matrix.transpose());
		MatrixXd trans_matrix = trDeisgnMatrix * design_matrix;

		MatrixXd inverseCoVarMatrix = m_model.getBeta() * trans_matrix;
		DiagonalMatrix<double, Eigen::Dynamic> dia(trans_matrix.rows());
		dia.setIdentity();
		inverseCoVarMatrix += (m_model.getAlpha() * dia);
		MatrixXd CoVarMatrix(inverseCoVarMatrix.inverse());

		Eigen::Map<const VectorXd> vector_target(target, rows);
		VectorXd b = trDeisgnMatrix * vector_target;
		VectorXd MeanVector(CoVarMatrix * b);
		MeanVector *= m_model.getBeta();

		m_model.setGaussianMean(MeanVector);
		m_model.setGaussianVarianceMatrix(CoVarMatrix);
	}

	void BayesianLrRegressionBuilder::useDefaultBasisFunctions(size_t dimension)
	{
		m_model.addBasisFunction(new BiasFunction(1.0));
		for (size_t i=0; i!=dimension; ++i)
		{
			m_model.addBasisFunction(new LinearBasisFunction(i));
		}
	}

	void BayesianLrRegressionBuilder::addBiasFunction()
	{
		addBasisFunction(new BiasFunction(1.0));
	}
		
	void BayesianLrRegressionBuilder::addLinearBasisFunction(size_t pos)
	{
		addBasisFunction(new LinearBasisFunction(pos));
	}
		
	void BayesianLrRegressionBuilder::addSigmodBasisFunction(size_t pos,
															double mean,
															double variance)
	{
		addBasisFunction(new SigmodBasisFunction(pos, mean, variance));
	}
		
	void BayesianLrRegressionBuilder::addSingleVarGussian(size_t pos,
														double mean,
														double variance)
	{
		addBasisFunction(new SingleVarGussian(pos, mean, variance));
	}
		
	void BayesianLrRegressionBuilder::addGussianBasisFunction(size_t dimension,
															const double* mean,
															const double* covariance)
	{
		addBasisFunction(new GussianBasisFunction(dimension, mean, covariance));
	}


}	//end of SML

