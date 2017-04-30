/*
 * James Brundege
 * Date: 2017-04-17
 * MIT license: https://opensource.org/licenses/MIT
 */
package jmb.jcortex.strategies.performanceevaluators;

import jmb.jcortex.data.DataSet;
import jmb.jcortex.neuralnet.NeuralNet;

/**
 *
 */
public interface PerformanceEvaluator {

    double getError(NeuralNet neuralNet, DataSet dataSet);

}
