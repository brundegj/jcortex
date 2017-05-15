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
public interface WeightAdjuster {

    SynMatrix adjustWeights(SynMatrix weights, int numExamples, double learningRate);

}
