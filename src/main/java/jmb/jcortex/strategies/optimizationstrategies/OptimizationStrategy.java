/*
 * James Brundege
 * Date: 2017-04-09
 * MIT license: https://opensource.org/licenses/MIT
 */
package jmb.jcortex.strategies.optimizationstrategies;

import jmb.jcortex.data.SynMatrix;

import java.util.List;

public interface OptimizationStrategy {

    List<SynMatrix> calcCorrections(List<SynMatrix> gradients);

    double getLearningRate();
}