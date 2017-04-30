/*
 * James Brundege
 * Date: 2017-04-09
 * MIT license: https://opensource.org/licenses/MIT
 */
package jmb.jcortex.strategies.haltingstrategies;

import jmb.jcortex.data.DataSet;
import jmb.jcortex.neuralnet.NeuralNet;
import jmb.jcortex.strategies.performanceevaluators.PerformanceEvaluator;

import java.text.NumberFormat;

import static java.lang.String.format;

public class ValidationSetHaltingStrategy implements HaltingStrategy {
    public static final NumberFormat ONE_DECIMAL = NumberFormat.getNumberInstance();
    static {
        ONE_DECIMAL.setMinimumFractionDigits(1);
        ONE_DECIMAL.setMaximumFractionDigits(1);
    }

    private final DataSet validationSet;
    private final DataSet trainingSet;
    private final int maxIterationSinceBestResult;
    private final PerformanceEvaluator performanceEvaluator;

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
        // FIXME add logging or event listeners
//        System.out.println(format("Training set %% wrong: %s \tValidation set %% wrong: %s",
//                ONE_DECIMAL.format(trainingSetError * 100), ONE_DECIMAL.format(validationSetError * 100)));
    }

    @Override
    public NeuralNet getBestNeuralNet() {
        return best;
    }

}
