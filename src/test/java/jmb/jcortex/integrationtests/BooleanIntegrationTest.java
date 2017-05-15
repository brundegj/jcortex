/*
 * James Brundege
 * Date: 2017-04-29
 * MIT license: https://opensource.org/licenses/MIT
 */
package jmb.jcortex.integrationtests;

import jmb.jcortex.data.DataSet;
import jmb.jcortex.data.SynMatrix;
import jmb.jcortex.neuralnet.NeuralNet;
import jmb.jcortex.neuralnet.NeuralNetBuilder;
import jmb.jcortex.strategies.batchingstrategies.FullTrainingSetBatchingStrategy;
import jmb.jcortex.strategies.haltingstrategies.ValidationSetHaltingStrategy;
import jmb.jcortex.strategies.optimizationstrategies.MomentumOptimizationStrategy;
import jmb.jcortex.strategies.performanceevaluators.ThresholdPerformanceEvaluator;
import jmb.jcortex.strategies.weightinitializers.LinearRandomWeightInitializer;
import jmb.jcortex.trainers.GradientDescentTrainerBuilder;
import jmb.jcortex.trainers.SupervisedTrainer;
import org.junit.Test;

import static jmb.jcortex.mapfunctions.MatrixFunctions.LINEAR_MATRIX_FUNCTION;
import static jmb.jcortex.mapfunctions.MatrixFunctions.SIGMOID_MATRIX_FUNCTION;
import static org.assertj.core.api.Assertions.assertThat;

/**
 *
 */
public class BooleanIntegrationTest {

    @Test
    public void shouldComputeBooleanTrue() {
        SynMatrix trainingData = new SynMatrix(new double[][]{
                {0, 0},
                {1, 0},
                {0, 1},
                {1, 1}
        });
        SynMatrix labels = new SynMatrix(1, 1, 1, 1);
        DataSet dataSet = new DataSet(trainingData, labels);
        runTest(dataSet, 2, 1);
    }

    @Test
    public void shouldComputeBooleanFalse() {
        SynMatrix trainingData = new SynMatrix(new double[][]{
                {0, 0},
                {1, 0},
                {0, 1},
                {1, 1}
        });
        SynMatrix labels = new SynMatrix(0, 0, 0, 0);
        DataSet dataSet = new DataSet(trainingData, labels);
        runTest(dataSet, 2, 1);
    }

    @Test
    public void shouldComputeBooleanAnd() {
        SynMatrix trainingData = new SynMatrix(new double[][]{
                {0, 0},
                {1, 0},
                {0, 1},
                {1, 1}
        });
        SynMatrix labels = new SynMatrix(0, 0, 0, 1);
        DataSet dataSet = new DataSet(trainingData, labels);
        runTest(dataSet, 2, 1);
    }

    @Test
    public void shouldComputeBooleanOr() {
        SynMatrix trainingData = new SynMatrix(new double[][]{
                {0, 0},
                {1, 0},
                {0, 1},
                {1, 1}
        });
        SynMatrix labels = new SynMatrix(0, 1, 1, 1);
        DataSet dataSet = new DataSet(trainingData, labels);
        runTest(dataSet, 2, 1);
    }

    @Test
    public void shouldComputeBooleanNand() {
        SynMatrix trainingData = new SynMatrix(new double[][]{
                {0, 0},
                {1, 0},
                {0, 1},
                {1, 1}
        });
        SynMatrix labels = new SynMatrix(1, 1, 1, 0);
        DataSet dataSet = new DataSet(trainingData, labels);
        runTest(dataSet, 2, 1);
    }

    @Test
    public void shouldComputeBooleanNor() {
        SynMatrix trainingData = new SynMatrix(new double[][]{
                {0, 0},
                {1, 0},
                {0, 1},
                {1, 1}
        });
        SynMatrix labels = new SynMatrix(1, 0, 0, 0);
        DataSet dataSet = new DataSet(trainingData, labels);
        runTest(dataSet, 2, 1);
    }

    @Test
    public void shouldComputeBooleanXor() {
        SynMatrix trainingData = new SynMatrix(new double[][]{
                {0, 0},
                {1, 0},
                {0, 1},
                {1, 1}
        });
        SynMatrix labels = new SynMatrix(0, 1, 1, 0);
        DataSet dataSet = new DataSet(trainingData, labels);
        runTest(dataSet, 2, 2, 1);
    }

    @Test
    public void shouldComputeBooleanNxor() {
        SynMatrix trainingData = new SynMatrix(new double[][]{
                {0, 0},
                {1, 0},
                {0, 1},
                {1, 1}
        });
        SynMatrix labels = new SynMatrix(1, 0, 0, 1);
        DataSet dataSet = new DataSet(trainingData, labels);
        runTest(dataSet, 2, 2, 1);
    }


    private void runTest(DataSet dataSet, int... dimensions) {
        NeuralNet untrainedNetwork = NeuralNetBuilder.createNeuralNet()
                .withDimensions(dimensions)
                .withActivationFunction(SIGMOID_MATRIX_FUNCTION)
                .withOutputFunction(LINEAR_MATRIX_FUNCTION)
                .withWeightInitializer(new LinearRandomWeightInitializer(-0.8, 0.8))
                .build();

        SupervisedTrainer trainer = GradientDescentTrainerBuilder.createTrainer()
                .withBatchingStrategy(new FullTrainingSetBatchingStrategy())
                .withOptimizationStrategy(new MomentumOptimizationStrategy(0.5, 0.9))
                .withHaltingStrategy(new ValidationSetHaltingStrategy(dataSet, dataSet,
                        new ThresholdPerformanceEvaluator(0.5), 10000))
                .build();

        NeuralNet trainedNetwork = trainer.train(untrainedNetwork, dataSet);

        double error = new ThresholdPerformanceEvaluator(0.5).getError(trainedNetwork, dataSet);
        assertThat(error).isLessThan(0.001);
    }

}
