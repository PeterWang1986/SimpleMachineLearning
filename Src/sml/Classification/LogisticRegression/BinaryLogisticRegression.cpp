
	//BinaryLogisticRegression.cpp
#include "sml/Classification/LogisticRegression/BinaryLogisticRegression.h"
#include "sml/Classification/LogisticRegression/SgdComponent.h"
#include "sml/Utilities/Math/functions.hpp"
#include "sml/Utilities/Optimization/StochasticGradientDescent.hpp"
#include "sml/Utilities/Optimization/StopCriterions.hpp"
#include "sml/Common/SmlSequenceSelection.h"
#include "sml/Common/SmlMacro.h"

#include <numeric>
#include <ctime>

namespace SML {
	using std::vector;
	using std::cout;
	using std::endl;

	class ForwardUnitSelection
	{
		typedef	ForwardUnitSelection		Self;

	public:
		ForwardUnitSelection(size_t beg, size_t end)
							: m_state(true)
							, m_iCount(0)
							, m_iWindow(end - beg)
		{
		}

		inline bool isContinue() const { return m_state; }
		inline size_t getNext()
		{ 
			m_state = false;
			return (m_iCount++)%m_iWindow;
		}
		void update() { m_state = true; }

	private:
		bool			m_state;
		size_t			m_iCount;
		const size_t	m_iWindow;
	};

	BiLogisticRegression::BiLogisticRegression()
	{
	}

	BiLogisticRegression::~BiLogisticRegression()
	{
	}

	/**
	* NOTE	we assume the first column of data are all the same with value 1
	**/
	bool BiLogisticRegression::stochasticGradientTraining(unsigned int rows, unsigned int cols,
														const double* data, const int* label,
														double learningRate,
														unsigned int maxIteration)
	{
		typedef Optimization::RelativeDiffValue<BiLRStandErrorFunction>				StopCriterion;

		if (0==data || 0==label) return false;
		
		initilizeWeights(cols);
		BiLRStandErrorFunction error(rows, cols, &(m_Weights[0]), data, label);
		StopCriterion sc(-1.0, maxIteration, error);
		const int window = (rows < 500) ? rows : 500;
		//RandomSelection sel(window, 0, rows);
		ForwardSelection sel(window, 0, rows);
		size_t count = 0;
		while (count < maxIteration)
		{
			while ( sel.isContinue() )
			{
				const size_t pos = sel.getNext();
				squence_training(cols, data+(pos*cols), *(label+pos), learningRate );
				++count;
			}

			if (sc.isStop()) break;
			sel.update();
		}

		return true;
	}

	/**
	* refer to the method vSGD-l from paper "No More Pesky Learning Rates"
	**/
	bool BiLogisticRegression::vlSGD(unsigned int rows, unsigned int cols,
									const double* data, const int* label,
									unsigned int maxIteration)
	{
		if (0==data || 0==label) return false;

		typedef BiLogisticRegGradient														Gradient;
		typedef Optimization::ExpectedAdapLocalRates<GaussNewtonBbprop>						LearningRates;
		typedef Optimization::RelativeDiffValue<BiLRStandErrorFunction>						StopCriterion;
		typedef ForwardSelection															Selection;
		typedef Optimization::SGD<Gradient, LearningRates, StopCriterion, Selection>		vSGD;

		initilizeWeights(cols);
		vSGD sgd(cols, &(m_Weights[0]));

		const double* weights = sgd.getWeighs();
		Gradient* gradient = new Gradient(cols, weights, data, label);
		LearningRates* lr = new LearningRates(cols, gradient->getGradient());
		lr->setDiagonalHessian(new GaussNewtonBbprop(cols, weights, data, label));
		BiLRStandErrorFunction error(rows, cols, weights, data, label);
		StopCriterion* sc = new StopCriterion(-1.0, 0.0000001, maxIteration, error);
		const int window = (rows < 500) ? rows : 500;
		Selection* sel = new Selection(window, 0, rows);

		sgd.setGradient(gradient);
		sgd.setLearningRates(lr);
		sgd.setStopCriterion(sc);
		sgd.setSelection(sel);
		bool isSuccess = sgd.optimize(rows, cols, data, label);

		std::copy(weights, weights + cols, m_Weights.begin());
		return isSuccess;
	}

	bool BiLogisticRegression::vgSGD(unsigned int rows, unsigned int cols,
									const double* data, const int* label,
									unsigned int maxIteration)
	{
		if (0==data || 0==label) return false;

		typedef BiLogisticRegGradient														Gradient;
		typedef Optimization::ExpectedAdapGlobalRates<GaussNewtonBbprop>					LearningRates;
		typedef Optimization::RelativeDiffValue<BiLRStandErrorFunction>						StopCriterion;
		typedef ForwardSelection															Selection;
		typedef Optimization::SGD<Gradient, LearningRates, StopCriterion, Selection>		vSGD;

		initilizeWeights(cols);
		vSGD sgd(cols, &(m_Weights[0]));

		const double* weights = sgd.getWeighs();
		Gradient* gradient = new Gradient(cols, weights, data, label);
		LearningRates* lr = new LearningRates(cols, gradient->getGradient());
		lr->setDiagonalHessian(new GaussNewtonBbprop(cols, weights, data, label));
		BiLRStandErrorFunction error(rows, cols, weights, data, label);
		StopCriterion* sc = new StopCriterion(-1.0, 0.0000001, maxIteration, error);
		const int window = (rows < 500) ? rows : 500;
		Selection* sel = new Selection(window, 0, rows);

		sgd.setGradient(gradient);
		sgd.setLearningRates(lr);
		sgd.setStopCriterion(sc);
		sgd.setSelection(sel);
		bool isSuccess = sgd.optimize(rows, cols, data, label);

		std::copy(weights, weights + cols, m_Weights.begin());
		return isSuccess;
	}

	void BiLogisticRegression::squence_training(unsigned int dimension, const double* data,
												int label, double learningRate)
	{
		const double a = std::inner_product(m_Weights.begin(), m_Weights.end(), data, 0.0);
		const double y = sigmoid(a);
		learningRate *= (label - y);

		for (unsigned int i=0; i!=dimension; ++i)
		{
			m_Weights[i] += (learningRate * data[i]);
		}
	}

	void BiLogisticRegression::initilizeWeights(unsigned int dimension)
	{
		unsigned long seeds = (unsigned long)std::time(0);
		std::default_random_engine generator(seeds);
		std::uniform_real_distribution<double> distribution(-0.01, 0.01);
		m_Weights.reserve(dimension);
		for (unsigned int i=0; i!=dimension; ++i)
		{
			m_Weights.push_back(distribution(generator));
		}
	}

	double BiLogisticRegression::dissimilarity(const vector<double>& v1,
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


}	//end of SML