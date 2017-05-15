/*
 * James Brundege
 * Date: 2017-05-01
 * MIT license: https://opensource.org/licenses/MIT
 */
package jmb.jcortex.strategies.regularization;

import jmb.jcortex.data.SynMatrix;

/**
 *
 */
public class L2Regularization implements WeightAdjuster {

    private double regularizationConstant;

    public L2Regularization(double regularizationConstant) {
        this.regularizationConstant = regularizationConstant;
    }

    @Override
    public SynMatrix adjustWeights(SynMatrix weights, int numExamples, double learningRate) {
        double regularizationFactor = 1.0 - learningRate * (regularizationConstant/numExamples);
        return weights.elementMultInPlace(regularizationFactor);
    }

}
