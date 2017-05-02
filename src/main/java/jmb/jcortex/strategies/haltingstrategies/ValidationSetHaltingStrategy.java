/*
 * James Brundege
 * Date: 2017-04-09
 * MIT license: https://opensource.org/licenses/MIT
 */
package jmb.jcortex.strategies.haltingstrategies;

import jmb.jcortex.data.DataSet;
import jmb.jcortex.neuralnet.NeuralNet;
import jmb.jcortex.strategies.performanceevaluators.PerformanceEvaluator;
import jmb.jcortex.strategies.performanceevaluators.PerformanceListener;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static jmb.jcortex.JCortexConstants.TRAINING_SET_PERCENT_WRONG;
import static jmb.jcortex.JCortexConstants.VALIDATION_SET_PERCENT_WRONG;

public class ValidationSetHaltingStrategy implements HaltingStrategy {

    private final DataSet validationSet;
    private final DataSet trainingSet;
    private final int maxIterationSinceBestResult;
    private final PerformanceEvaluator performanceEvaluator;
    private List<PerformanceListener> performanceListeners = new ArrayList<>();

    private NeuralNet best;
    private int iterationsSinceBest = 0;
    private double bestError = Double.MAX_VALUE;

    public ValidationSetHaltingStrategy(DataSet trainingSet, DataSet validationSet,
                                        PerformanceEvaluator performanceEvaluator, int maxIterationSinceBestResult)
    {
        this.validationSet = validationSet;
        this.trainingSet = trainingSet;
        this.performanceEvaluator = performanceEvaluator;
        this.maxIterationSinceBestResult = maxIterationSinceBestResult;
    }

    @Override
    public boolean shouldHalt() {
        return bestError == 0 || iterationsSinceBest > maxIterationSinceBestResult;
    }

    @Override
    public double evaluate(NeuralNet neuralNet) {
        double trainingSetError = performanceEvaluator.getError(neuralNet, trainingSet);
        double validationSetError = performanceEvaluator.getError(neuralNet, validationSet);
        reportPerformance(trainingSetError, validationSetError);
        if (validationSetError < bestError) {
            bestError = validationSetError;
            best = neuralNet.copy();
            iterationsSinceBest = 0;
        }
        iterationsSinceBest++;
        return validationSetError;
    }

    private void reportPerformance(double trainingSetError, double validationSetError) {
        if (performanceListeners.isEmpty()) return;

        Map<String, Double> performanceData = new HashMap<>();
        performanceData.put(TRAINING_SET_PERCENT_WRONG, trainingSetError);
        performanceData.put(VALIDATION_SET_PERCENT_WRONG, validationSetError);
        performanceListeners.forEach(listener -> listener.performanceEvent(performanceData));
    }

    @Override
    public NeuralNet getBestNeuralNet() {
        return best;
    }

    public void addPerformanceListener(PerformanceListener performanceListener) {
        performanceListeners.add(performanceListener);
    }
}
