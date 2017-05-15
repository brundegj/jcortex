/*
 * James Brundege
 * Date: 2017-04-25
 * MIT license: https://opensource.org/licenses/MIT
 */
package jmb.jcortex.strategies.optimizationstrategies;

import jmb.jcortex.data.SynMatrix;

import java.util.List;

import static java.util.stream.Collectors.toList;

/**
 *
 */
public class FixedLearningRateOptimizationStrategy implements OptimizationStrategy {

    private final double learningRate;

    public FixedLearningRateOptimizationStrategy(double learningRate) {
        this.learningRate = learningRate;
    }

    @Override
    public List<SynMatrix> calcCorrections(List<SynMatrix> gradients) {
        return gradients.parallelStream()
                .map(gradient -> gradient.elementMult(learningRate))
                .collect(toList());
    }

    @Override
    public double getLearningRate() {
        return learningRate;
    }
}
