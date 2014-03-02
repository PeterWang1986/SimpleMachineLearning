#ifndef	_STOCHASTIC_GRADIENT_DESCENT_H
#define _STOCHASTIC_GRADIENT_DESCENT_H

#include <algorithm>
#include <random>
#include <ctime>
#include <vector>
#include <numeric>

namespace Optimization {

	template <typename DiagonalHessian>
	class ExpectedAdapRates
	{
		typedef ExpectedAdapRates<DiagonalHessian>		Self;

		ExpectedAdapRates(const Self& rhs);
		Self& operator= (const Self& rhs);

	public:
		typedef	DiagonalHessian				EstimateDigHessian;

		ExpectedAdapRates(size_t numWeights,
						const double* gradient);
		ExpectedAdapRates(size_t numWeights,
						const double* pGV,
						const double* pG,
						const double* pV,
						const double* pT,
						const double* pH);
		~ExpectedAdapRates();
		
		inline EstimateDigHessian* getDiagonalHessian() { return m_pDiagHessian; }
		inline size_t getNumWeights() const { return m_iNumWeights; }
		inline const double* getExpectedGradient() const { return m_g; }
		inline const double* getExpectedSquareGradient() const { return m_v; }
		inline const double* getTimeConstant() const { return m_t; }
		inline const double* getCurvature() const { return m_h; }

		void setDiagonalHessian(DiagonalHessian* pDh);
		template <typename ForwardIter>
		void setExpectedGradient(ForwardIter beg, ForwardIter end) { std::copy(beg, end, m_g); }
		template <typename ForwardIter>
		void setExpectedSquareGradient(ForwardIter beg, ForwardIter end) { std::copy(beg, end, m_v); }
		template <typename ForwardIter>
		void setTimeConstant(ForwardIter beg, ForwardIter end) { std::copy(beg, end, m_t); }
		template <typename ForwardIter>
		void setCurvature(ForwardIter beg, ForwardIter end) { std::copy(beg, end, m_h); }

	private:
		void init();

	protected:
		const size_t		m_iNumWeights;
		const double*		m_gradient;
		double*				m_g; /*approximate E[g], 'g' denote gradient*/
		double*				m_v; /*approximate E[g^2]*/
		double*				m_t; /*adaptive time-constant*/
		double*				m_h; /*approximate curvature*/
		DiagonalHessian*	m_pDiagHessian;
	};

	template <typename DiagonalHessian>
	ExpectedAdapRates<DiagonalHessian>::ExpectedAdapRates(size_t numWeights,
														const double* gradient)
														: m_iNumWeights(numWeights)
														, m_gradient(gradient)
														, m_g(new double[numWeights])
														, m_v(new double[numWeights])
														, m_t(new double[numWeights])
														, m_h(new double[numWeights])
														, m_pDiagHessian(0)
	{
		init();
	}

	template <typename DiagonalHessian>
	ExpectedAdapRates<DiagonalHessian>::ExpectedAdapRates(size_t numWeights,
														const double* pGV,
														const double* pG,
														const double* pV,
														const double* pT,
														const double* pH)
														: m_iNumWeights(numWeights)
														, m_gradient(pGV)
														, m_g(new double[numWeights])
														, m_v(new double[numWeights])
														, m_t(new double[numWeights])
														, m_h(new double[numWeights])
														, m_pDiagHessian(0)
	{
		this->setExpectedGradient(pG, pG + numWeights);
		this->setExpectedSquareGradient(pV, pV + numWeights);
		this->setTimeConstant(pT, pT + numWeights);
		this->setCurvature(pH, pH + numWeights);
	}

	template <typename DiagonalHessian>
	ExpectedAdapRates<DiagonalHessian>::~ExpectedAdapRates()
	{
		delete [] m_g;
		delete [] m_v;
		delete [] m_t;
		delete [] m_h;
		delete m_pDiagHessian;
	}

	template <typename DiagonalHessian>
	void ExpectedAdapRates<DiagonalHessian>::setDiagonalHessian(DiagonalHessian* pDh)
	{
		if (0 != m_pDiagHessian)
		{
			delete m_pDiagHessian;
		}

		m_pDiagHessian = pDh;
	}

	template <typename DiagonalHessian>
	void ExpectedAdapRates<DiagonalHessian>::init()
	{
		unsigned long seeds = (unsigned long)std::time(0);
		std::default_random_engine generator(seeds);
		std::default_random_engine generator1(generator());
		std::default_random_engine generator2(generator());
		std::default_random_engine generator3(generator());
		std::default_random_engine generator4(generator());
		std::uniform_real_distribution<double> distribution1(-0.05, 0.05);
		std::uniform_real_distribution<double> distribution2(0.001, 0.05);
		std::uniform_real_distribution<double> distribution3(1, 2);
		std::uniform_real_distribution<double> distribution4(0.001, 0.05);
		for (size_t i=0; i!=m_iNumWeights; ++i)
		{
			m_g[i] = distribution1(generator1);
			m_v[i] = distribution2(generator2);
			m_t[i] = distribution3(generator3);
			m_h[i] = distribution4(generator4);
		}
	}

	/*******************************************************************
	*						ExpectedAdapLocalRates
	**********************************************************************/
	template <typename DiagonalHessian>
	class ExpectedAdapLocalRates : public ExpectedAdapRates<DiagonalHessian>
	{
	public:
		typedef	ExpectedAdapRates<DiagonalHessian>		Parent;

		ExpectedAdapLocalRates(size_t numWeights,
							const double* gradient);
		ExpectedAdapLocalRates(size_t numWeights,
							const double* pGV,
							const double* pG,
							const double* pV,
							const double* pT,
							const double* pH);
		~ExpectedAdapLocalRates();

		void calcLearningRates(size_t pos);

		inline double getLearningRates(size_t pos) const { return m_rates[pos]; }

	private:
		double*				m_rates; /*learning rates*/
	};

	template <typename DiagonalHessian>
	ExpectedAdapLocalRates<DiagonalHessian>::ExpectedAdapLocalRates(size_t numWeights,
																	const double* gradient)
																	: Parent(numWeights, gradient)
																	, m_rates(new double[numWeights])
	{
	}

	template <typename DiagonalHessian>
	ExpectedAdapLocalRates<DiagonalHessian>::ExpectedAdapLocalRates(size_t numWeights,
																	const double* pGV,
																	const double* pG,
																	const double* pV,
																	const double* pT,
																	const double* pH)
																	: Parent(numWeights, pGV, pG, pV, pT, pH)
																	, m_rates(new double[numWeights])
	{
	}

	template <typename DiagonalHessian>
	ExpectedAdapLocalRates<DiagonalHessian>::~ExpectedAdapLocalRates()
	{
		delete [] m_rates;
	}

	template <typename DiagonalHessian>
	void ExpectedAdapLocalRates<DiagonalHessian>::calcLearningRates(size_t pos)
	{
		m_pDiagHessian->calcDiagonalHessian(pos);
		for (size_t i=0; i!=m_iNumWeights; ++i)
		{
			const double p = 1.0/m_t[i];	//inverse time constant
			const double q = 1 - p;
			const double gradient = m_gradient[i];
			m_g[i] = q * m_g[i] + p * gradient;
			m_v[i] = q * m_v[i] + p * gradient * gradient;

			const double h = q * m_h[i] + p * std::fabs(m_pDiagHessian->getDiagonalHessian(i));
			m_h[i] = (h < 0.01) ? 0.01 : h;

			const double gg = m_g[i] * m_g[i];
			m_rates[i] = gg/(m_h[i] * m_v[i]);
			m_t[i] = (1 - (gg/m_v[i])) * m_t[i] + 1;
		}
	}

	/*****************************************************************************
	*							ExpectedAdapGlobalRates
	*********************************************************************************/
	template <typename DiagonalHessian>
	class ExpectedAdapGlobalRates : public ExpectedAdapRates<DiagonalHessian>
	{
	public:
		typedef	ExpectedAdapRates<DiagonalHessian>			Parent;

		ExpectedAdapGlobalRates(size_t numWeights,
							const double* gradient);
		ExpectedAdapGlobalRates(size_t numWeights,
							const double* pGV,
							const double* pG,
							const double* pV,
							const double* pT,
							const double* pH,
							double ll);
		~ExpectedAdapGlobalRates();

		void calcLearningRates(size_t pos);

		inline double getLearningRates(size_t pos) const { return m_rate; }

	private:
		double				m_l;
		double				m_rate; /*learning rates*/
		double				m_global_t;
	};

	template <typename DiagonalHessian>
	ExpectedAdapGlobalRates<DiagonalHessian>::ExpectedAdapGlobalRates(size_t numWeights,
																	const double* gradient)
																	: Parent(numWeights, gradient)
																	, m_l(1.0)
																	, m_global_t(1)
	{
		const double* pESG = this->getExpectedSquareGradient();
		m_l = std::accumulate(pESG, pESG + numWeights, 0.0);
	}

	template <typename DiagonalHessian>
	ExpectedAdapGlobalRates<DiagonalHessian>::ExpectedAdapGlobalRates(size_t numWeights,
																	const double* pGV,
																	const double* pG,
																	const double* pV,
																	const double* pT,
																	const double* pH,
																	double ll)
																	: Parent(numWeights, pGV, pG, pV, pT, pH)
																	, m_l(ll)
																	, m_global_t(1)
	{
	}

	template <typename DiagonalHessian>
	ExpectedAdapGlobalRates<DiagonalHessian>::~ExpectedAdapGlobalRates()
	{
	}

	template <typename DiagonalHessian>
	void ExpectedAdapGlobalRates<DiagonalHessian>::calcLearningRates(size_t pos)
	{
		m_pDiagHessian->calcDiagonalHessian(pos);
		const double* hessian = m_pDiagHessian->getDiagonalHessian();
		double max_h = -1;
		double sum_square_gradient = 0.0, sum_square_g = 0.0, sum_v = 0.0;
		for (size_t i=0; i!=m_iNumWeights; ++i)
		{
			const double t = m_t[i];
			const double p = 1.0/t;	//inverse time constant
			const double q = 1 - p;
			const double gradient = m_gradient[i];
			const double square_gradient = gradient * gradient;
			m_g[i] = q * m_g[i] + p * gradient;
			m_v[i] = q * m_v[i] + p * square_gradient;

			const double g = m_g[i];
			const double v = m_v[i];
			const double gg = g * g;
			m_t[i] = (1 - (gg/v)) * t + 1;
			sum_square_gradient += square_gradient;
			sum_square_g += gg;
			sum_v += v;
			max_h = (max_h < hessian[i]) ? hessian[i] : max_h;
		}
		m_global_t = (1 - (sum_square_g/sum_v))*m_global_t + 1;
		const double p = 1/m_global_t;
		m_l = (1-p)*m_l + p*sum_square_gradient;
		max_h = (max_h < 0.01) ? 0.01 : max_h;
		m_rate = sum_square_g / (max_h * m_l);

		//std::cout << "max_h = " << max_h << std::endl;
		//std::cout << "m_l = " << m_l << std::endl;
		//std::cout << "m_rate = " << m_rate << std::endl;
	}


	/**********************************************************************************
	*									SGD
	************************************************************************************/
	template <typename Gradient, typename LearningRates, typename StopCriterion, typename Selection>
	class SGD
	{
		typedef SGD<Gradient, LearningRates, StopCriterion, Selection>		Self;

		SGD(const Self& rhs);
		Self& operator= (const Self& rhs);

	public:
		explicit SGD(size_t numWeights);
		SGD(size_t numWeights, const double* weights);
		~SGD();

		bool optimize(size_t rows, size_t cols, const double* data, const int* label=0);

		inline void setGradient(Gradient* pGt) { set<Gradient>(pGt, m_pGradient); }
		inline void setLearningRates(LearningRates* pLr) { set<LearningRates>(pLr, m_pLearningRates); }
		inline void setStopCriterion(StopCriterion* pSc) { set<StopCriterion>(pSc, m_pStopCriterion); }
		inline void setSelection(Selection* pSl) { set<Selection>(pSl, m_pSelection); }
		inline void setWeights(size_t pos, double val) { m_weights[pos] = val; }
		inline size_t getNumWeights() const { return m_iNumWeights; }
		inline const double* getWeighs() const { return m_weights; }
		inline double getWeights(size_t pos) const { return m_weights[pos]; }

	private:
		template <typename T> void set(T* pointer, T* &cur) {
			if (0 != cur) { delete cur; }
			cur = pointer;
		}
		inline bool checkSGDStatus() const {
			return (0!=m_pGradient && 0!=m_pLearningRates && 0!=m_pStopCriterion && 0!=m_pSelection);
		}
		void aditionalInitialize(size_t rows, size_t cols);

	private:
		const size_t	m_iNumWeights;
		double*			m_weights;

		Gradient*		m_pGradient;
		LearningRates*	m_pLearningRates;
		StopCriterion*	m_pStopCriterion;
		Selection*		m_pSelection;
	};

	template <typename Gradient, typename LearningRates, typename StopCriterion, typename Selection>
	SGD<Gradient, LearningRates, StopCriterion, Selection>::SGD(size_t numWeights)
																: m_iNumWeights(numWeights)
																, m_weights(new double[numWeights])
																, m_pGradient(0)
																, m_pLearningRates(0)
																, m_pStopCriterion(0)
																, m_pSelection(0)
	{
	}

	template <typename Gradient, typename LearningRates, typename StopCriterion, typename Selection>
	SGD<Gradient, LearningRates, StopCriterion, Selection>::SGD(size_t numWeights, const double* weights)
																: m_iNumWeights(numWeights)
																, m_weights(new double[numWeights])
																, m_pGradient(0)
																, m_pLearningRates(0)
																, m_pStopCriterion(0)
																, m_pSelection(0)
	{
		std::copy(weights, weights + numWeights, m_weights);
	}

	template <typename Gradient, typename LearningRates, typename StopCriterion, typename Selection>
	SGD<Gradient, LearningRates, StopCriterion, Selection>::~SGD()
	{
		delete [] m_weights;
		delete m_pGradient;
		delete m_pLearningRates;
		delete m_pStopCriterion;
		delete m_pSelection;
	}

	template <typename Gradient, typename LearningRates, typename StopCriterion, typename Selection>
	bool SGD<Gradient, LearningRates, StopCriterion, Selection>::optimize(size_t rows,
																		size_t cols,
																		const double* data,
																		const int* label)
	{
		if (0==rows || 0==cols) return false;
		if (0 == data) return false;
		if ( !checkSGDStatus() ) return false;

		aditionalInitialize(rows, cols);

		do {
			while (m_pSelection->isContinue())
			{
				const size_t pos = m_pSelection->getNext();
				m_pGradient->calcGradient(pos);
				m_pLearningRates->calcLearningRates(pos);
				for (size_t i=0; i!=m_iNumWeights; ++i)
				{
					m_weights[i] -= (m_pLearningRates->getLearningRates(i)) * (m_pGradient->getGradient(i));
				}

				//std::cout << m_pGradient->getGradient(0) << ' ' << m_pGradient->getGradient(1) << ' ' << m_pGradient->getGradient(2) << std::endl;
				//std::cout << m_weights[0] << ' ' << m_weights[1] << ' ' << m_weights[2] << std::endl;
			}
			m_pSelection->update();
		}
		while ( !m_pStopCriterion->isStop() );

		return true;
	}
	
	template <typename Gradient, typename LearningRates, typename StopCriterion, typename Selection>
	void SGD<Gradient, LearningRates, StopCriterion, Selection>::aditionalInitialize(size_t rows, size_t cols)
	{
		typename typedef LearningRates::EstimateDigHessian		EstimateDigHessian;

		EstimateDigHessian* pDiagonal = m_pLearningRates->getDiagonalHessian();
		std::vector<double> expected_gradient(cols, 0.0);
		std::vector<double> square_expected_gradient(cols, 0.0);
		std::vector<double> diagonal_hessian(cols, 0.0);
		size_t count = (0.001*rows) < 20 ? 20 : (0.001*rows);
		count = (count < rows) ? count : rows;
		for (size_t i=0; i!=count; ++i)
		{
			m_pGradient->calcGradient(i);
			pDiagonal->calcDiagonalHessian(i);
			for (size_t j=0; j!=cols; ++j)
			{
				const double g = m_pGradient->getGradient(j);
				expected_gradient[j] += g;
				square_expected_gradient[j] += (g * g);
				diagonal_hessian[j] += pDiagonal->getDiagonalHessian(j);
			}
		}

		for (size_t i=0; i!=cols; ++i)
		{
			expected_gradient[i] /= count;
			square_expected_gradient[i] /= count;
			diagonal_hessian[i] /= count;
		}
		m_pLearningRates->setExpectedGradient(expected_gradient.begin(), expected_gradient.end());
		m_pLearningRates->setExpectedSquareGradient(square_expected_gradient.begin(), square_expected_gradient.end());
		m_pLearningRates->setCurvature(diagonal_hessian.begin(), diagonal_hessian.end());
	}

	class DifferenceVector
	{
	public:
		DifferenceVector(double threshold,
						size_t numWeights,
						const double* weights);
		DifferenceVector(double threshold,
						size_t numWeights,
						const double* weights,
						size_t maxTimes,
						size_t count);
		~DifferenceVector();

		bool isStop();

	private:
		const double	m_fExitThreshold;
		const size_t	m_iNumWeights;
		double*			m_previous; /*refer to the previous weights vector*/
		const double*	m_currents; /*refer to the latest weights vector*/
		size_t			m_times; /*denote the iterator steps*/
		size_t			m_count;
		const size_t	m_max_times;
		const size_t	m_max_count;
	};

}	//end of namespace Optimization

#endif