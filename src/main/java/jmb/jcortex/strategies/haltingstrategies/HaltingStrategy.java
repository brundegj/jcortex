package jmb.jcortex.strategies.haltingstrategies;

import jmb.jcortex.neuralnet.NeuralNet;

/**
 * James Brundege
 * Date: 2017-04-09
 * MIT license: https://opensource.org/licenses/MIT
 */
public interface HaltingStrategy {

    /** Return true if training should stop */
    boolean shouldHalt();

    /** Return the percent correct (0-1), and store the current best neural net */
    double evaluate(NeuralNet neuralNet);

    /** Return the best neural net, typically determined by the percent correct against a validation set */
    NeuralNet getBestNeuralNet();

}