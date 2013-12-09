#ifndef	_BASIS_FUNCTIONS_H
#define _BASIS_FUNCTIONS_H

#include "sml\Common\SmlMacro.h"

#include "eigen\Core"

namespace SML {

	class BasisFunctionBase
	{
	public:
		BasisFunctionBase();
		virtual ~BasisFunctionBase();

		virtual double phi(const double* data, size_t len) const =0;
	};

	class BiasFunction : public BasisFunctionBase
	{
	public:
		explicit BiasFunction(double bias =1.0);
		virtual ~BiasFunction();

		virtual double phi(const double* data, size_t len) const {
			return phiAux(data, data + len);
		}
		
		template <class RandomAccessIter>
		inline ITER_VALUE_TYPE(RandomAccessIter) phiAux(RandomAccessIter beg, RandomAccessIter end) const
		{
			return m_fBias;
		}

	private:
		const double m_fBias;
	};

	class LinearBasisFunction : public BasisFunctionBase
	{
	public:
		explicit LinearBasisFunction(size_t pos);
		virtual ~LinearBasisFunction();

		virtual double phi(const double* data, size_t len) const {
			return phiAux(data, data + len);
		}

		template <class RandomAccessIter>
		inline ITER_VALUE_TYPE(RandomAccessIter) phiAux(RandomAccessIter beg, RandomAccessIter end) const
		{
			return *(beg + m_iPosition);
		}

	private:
		const size_t m_iPosition;
	};

	class SigmodBasisFunction : public BasisFunctionBase
	{
	public:
		SigmodBasisFunction(size_t pos, double mean, double variance);
		virtual ~SigmodBasisFunction();

		virtual double phi(const double* data, size_t len) const {
			return phiAux(data, data + len);
		}

		template <class RandomAccessIter>
		inline ITER_VALUE_TYPE(RandomAccessIter) phiAux(RandomAccessIter beg, RandomAccessIter end) const
		{
			std::advance(beg, m_iPosition);
			const double a = 0 - (*beg - m_fMean)/m_fStdVariance;

			return (1/(1+exp(a)));
		}

	private:
		const size_t	m_iPosition;
		const double	m_fMean;
		const double	m_fStdVariance;
	};

	class SingleVarGussian : public BasisFunctionBase
	{
	public:
		SingleVarGussian(size_t pos, double mean, double variance);
		virtual ~SingleVarGussian();

		virtual double phi(const double* data, size_t len) const {
			return phiAux(data, data + len);
		}

		template <class RandomAccessIter>
		inline ITER_VALUE_TYPE(RandomAccessIter) phiAux(RandomAccessIter beg, RandomAccessIter end) const
		{
			std::advance(beg, m_iPosition);
			const double diff = *beg - m_fMean;
			const double f = (diff * diff)/m_fDenominator;

			return exp(f);
		}

	private:
		const size_t	m_iPosition;
		const double	m_fMean;
		const double	m_fDenominator;		/* 0 - 2*variance */
	};

	/*
	NOTE:	here we remove 1/(pow(2*PI, D/2)*sqrt(|covariance matrix|)),
			because it can be adapted by w
	*/
	class GussianBasisFunction : public BasisFunctionBase
	{
	public:
		GussianBasisFunction(size_t dimension,
							const double* mean,
							const double* covariance);
		virtual ~GussianBasisFunction();

		virtual double phi(const double* data, size_t len) const;

	private:
		void initialize(size_t dim, const double* mean, const double* covariance);

	private:
		Eigen::VectorXd		m_Mean;
		Eigen::MatrixXd		m_PrecisionMatrix;
	};


}	//end of SML
#endif