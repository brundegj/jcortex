/*
 * James Brundege
 * Date: 2017-04-09
 * MIT license: https://opensource.org/licenses/MIT
 */
package jmb.jcortex.trainers;

import jmb.jcortex.data.DataSet;
import jmb.jcortex.data.SynMatrix;
import jmb.jcortex.neuralnet.NeuralNet;
import jmb.jcortex.strategies.batchingstrategies.BatchedDataSet;
import jmb.jcortex.strategies.batchingstrategies.BatchingStrategy;
import jmb.jcortex.strategies.batchingstrategies.FixedNumBatchingStrategy;
import jmb.jcortex.strategies.haltingstrategies.HaltingStrategy;
import jmb.jcortex.strategies.optimizationstrategies.OptimizationStrategy;
import jmb.jcortex.strategies.regularization.WeightAdjuster;

import java.util.Arrays;
import java.util.List;

/**
 * A trainer for neural nets that uses gradient descent as the minimization function. Contains hooks to allow
 * different optimizations and regularizations.
 *
 * Use GradientDescentTrainerBuilder to conveniently create and configure an instance of this trainer.
 */
public class GradientDescentTrainer implements SupervisedTrainer {

    private BatchingStrategy batchingStrategy;
    private OptimizationStrategy optimizationStrategy;
    private HaltingStrategy haltingStrategy;
    private WeightAdjuster weightAdjuster;

    private DeltaCalculator deltaCalculator = new DeltaCalculator();
    private GradientCalculator gradientCalculator = new GradientCalculator();

    public GradientDescentTrainer(OptimizationStrategy optimizationStrategy, HaltingStrategy haltingStrategy) {
        this(new FixedNumBatchingStrategy(1), optimizationStrategy, haltingStrategy);
    }

    public GradientDescentTrainer(BatchingStrategy batchingStrategy, OptimizationStrategy optimizationStrategy, HaltingStrategy haltingStrategy) {
        this.optimizationStrategy = optimizationStrategy;
        this.haltingStrategy = haltingStrategy;
        this.batchingStrategy = batchingStrategy;
    }

    @Override
    public NeuralNet train(NeuralNet neuralNet, DataSet trainingSet) {
        haltingStrategy.evaluate(neuralNet);
        while (!haltingStrategy.shouldHalt()) {
            neuralNet = doTrainingIteration(neuralNet, trainingSet);
            haltingStrategy.evaluate(neuralNet);
        }
        return haltingStrategy.getBestNeuralNet();
    }

    private NeuralNet doTrainingIteration(NeuralNet neuralNet, DataSet trainingSet) {
        BatchedDataSet batches = batchingStrategy.getBatchedDataSet(trainingSet);
        while(batches.hasNext()) {
            DataSet batch = batches.getNextBatch();
            List<SynMatrix> nodeValues = neuralNet.trainForward(batch);
            neuralNet = doBackPropagation(neuralNet, nodeValues, batch.getLabels());
        }
        return neuralNet;
    }

    private NeuralNet doBackPropagation(NeuralNet neuralNet, List<SynMatrix> nodeValues, SynMatrix labels) {
        List<SynMatrix> deltas = deltaCalculator.calcDeltas(nodeValues, labels, neuralNet);
        List<SynMatrix> gradients = gradientCalculator.calcGradients(deltas, nodeValues);
        List<SynMatrix> newLayers = updateParameters(neuralNet.getLayers(), gradients, labels.numRows());
        neuralNet.setLayers(newLayers);
        return neuralNet;
    }

    private List<SynMatrix> updateParameters(List<SynMatrix> layers, List<SynMatrix> gradients, int numExamples) {
        SynMatrix[] corrections = optimizationStrategy.calcCorrections(gradients).toArray(new SynMatrix[0]);
        SynMatrix[] weights = layers.toArray(new SynMatrix[layers.size()]);
        for (int i = 0; i < corrections.length; i++) {
            weights[i] = applyRegularization(weights[i], numExamples);
            weights[i].minusInPlace(corrections[i]);
        }
        return Arrays.asList(weights);
    }

    private SynMatrix applyRegularization(SynMatrix weights, int numExamples) {
        if (weightAdjuster != null) {
            double learningRate = optimizationStrategy.getLearningRate();
            weights = weightAdjuster.adjustWeights(weights, numExamples, learningRate);
        }
        return weights;
    }

    public BatchingStrategy getBatchingStrategy() {
        return batchingStrategy;
    }

    public OptimizationStrategy getOptimizationStrategy() {
        return optimizationStrategy;
    }

    public HaltingStrategy getHaltingStrategy() {
        return haltingStrategy;
    }

    public DeltaCalculator getDeltaCalculator() {
        return deltaCalculator;
    }

    public void setDeltaCalculator(DeltaCalculator deltaCalculator) {
        this.deltaCalculator = deltaCalculator;
    }

    public GradientCalculator getGradientCalculator() {
        return gradientCalculator;
    }

    public void setGradientCalculator(GradientCalculator gradientCalculator) {
        this.gradientCalculator = gradientCalculator;
    }

    public WeightAdjuster getWeightAdjuster() {
        return weightAdjuster;
    }

    public void setWeightAdjuster(WeightAdjuster weightAdjuster) {
        this.weightAdjuster = weightAdjuster;
    }
}
