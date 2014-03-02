#ifndef SGD_COMPONENT_H
#define SGD_COMPONENT_H

namespace SML {

	class BiLogisticRegGradient
	{
	public:
		BiLogisticRegGradient(size_t numWeights,
							const double* weights,
							const double* data,
							const int* label);
		~BiLogisticRegGradient();

		void calcGradient(size_t pos);

		inline double getGradient(size_t pos) const { return m_gradient[pos]; }
		inline const double* getGradient() const { return m_gradient; }

	private:
		const size_t	m_iNumWeights; /*also equal the number of columns*/
		const double*	m_weights;
		const double*	m_data;
		const int*		m_label;
		double*			m_gradient;
	};

	class GaussNewtonBbprop
	{
	public:
		GaussNewtonBbprop(size_t numWeights,
						const double* weights,
						const double* data,
						const int* label);
		~GaussNewtonBbprop();

		void calcDiagonalHessian(size_t pos);

		inline double getDiagonalHessian(size_t pos) const { return m_DiagonalHessian[pos]; }
		inline const double* getDiagonalHessian() const { return m_DiagonalHessian; }

	private:
		const size_t	m_iNumWeights; /*also equal the number of columns*/
		const double*	m_weights;
		const double*	m_data;
		const int*		m_label;
		double*			m_DiagonalHessian;
	};

	class BiLRStandErrorFunction
	{
	public:
		BiLRStandErrorFunction(size_t rows, size_t cols,
							const double* weights,
							const double* data,
							const int* label);

		double operator() (void) const;

	private:
		const size_t	m_iRows;
		const size_t	m_iCols;
		const double*	m_weights;
		const double*	m_data;
		const int*		m_label;
	};


}	//end of namespace SML

#endif