/*
 * James Brundege
 * Date: 2017-04-09
 * MIT license: https://opensource.org/licenses/MIT
 */
package jmb.jcortex.strategies.optimizationstrategies;

import jmb.jcortex.data.SynMatrix;

import java.util.List;

import static java.util.stream.Collectors.toList;

public class MomentumOptimizationStrategy implements OptimizationStrategy {

    private double learningRate;
    private double momentum;
    private List<SynMatrix> previousCorrections;

    public MomentumOptimizationStrategy(double learningRate, double momentum) {
        this.learningRate = learningRate;
        this.momentum = momentum;
    }

    @Override
    public List<SynMatrix> calcCorrections(List<SynMatrix> gradients) {
        List<SynMatrix> corrections = gradients.parallelStream()
                .map(gradient -> gradient.elementMult(learningRate))
                .collect(toList());
        if (previousCorrections != null) {
            for (int i = 0; i < previousCorrections.size(); i++) {
                previousCorrections.get(i).elementMultInPlace(momentum);
                corrections.get(i).plusInPlace(previousCorrections.get(i));
            }
        }
        previousCorrections = corrections;
        return corrections;
    }
}
