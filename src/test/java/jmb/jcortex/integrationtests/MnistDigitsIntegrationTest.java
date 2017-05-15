/*
 * James Brundege
 * Date: 2017-04-09
 * MIT license: https://opensource.org/licenses/MIT
 */
package jmb.jcortex.integrationtests;

import jmb.jcortex.data.DataSet;
import jmb.jcortex.data.DataSetSplitter;
import jmb.jcortex.datasource.MnistDigitsDataService;
import jmb.jcortex.neuralnet.NeuralNet;
import jmb.jcortex.neuralnet.NeuralNetBuilder;
import jmb.jcortex.strategies.batchingstrategies.FixedNumBatchingStrategy;
import jmb.jcortex.strategies.haltingstrategies.ValidationSetHaltingStrategy;
import jmb.jcortex.strategies.optimizationstrategies.MomentumOptimizationStrategy;
import jmb.jcortex.strategies.performanceevaluators.ClassificationPerformanceEvaluator;
import jmb.jcortex.strategies.performanceevaluators.PrintlnPerformanceListener;
import jmb.jcortex.strategies.weightinitializers.LinearRandomWeightInitializer;
import jmb.jcortex.trainers.GradientDescentTrainerBuilder;
import jmb.jcortex.trainers.SupervisedTrainer;
import org.junit.Test;

import static jmb.jcortex.mapfunctions.MatrixFunctions.SIGMOID_MATRIX_FUNCTION;
import static jmb.jcortex.mapfunctions.MatrixFunctions.SOFTMAX_MATRIX_FUNCTION;
import static org.assertj.core.api.Assertions.assertThat;

public class MnistDigitsIntegrationTest {

    @Test
    public void trainMnistDigitClassifier() {
        DataSet[] sets = loadData();
        DataSet training = sets[0];
        DataSet validation = sets[1];
        DataSet testing = sets[2];

        NeuralNet untrainedNetwork = NeuralNetBuilder.createNeuralNet()
                .withDimensions(784, 200, 200, 10)
                .withActivationFunction(SIGMOID_MATRIX_FUNCTION)
                .withOutputFunction(SOFTMAX_MATRIX_FUNCTION)
                .withWeightInitializer(new LinearRandomWeightInitializer(-0.3, 0.3))
                .build();

        ValidationSetHaltingStrategy haltingStrategy = new ValidationSetHaltingStrategy(training, validation,
                new ClassificationPerformanceEvaluator(), 3);
        haltingStrategy.addPerformanceListener(new PrintlnPerformanceListener());

        SupervisedTrainer trainer = GradientDescentTrainerBuilder.createTrainer()
                .withBatchingStrategy(new FixedNumBatchingStrategy(100))
                .withOptimizationStrategy(new MomentumOptimizationStrategy(0.08, 0.9))
                .withHaltingStrategy(haltingStrategy)
                .build();

        NeuralNet trainedNetwork = trainer.train(untrainedNetwork, training);

        double error = new ClassificationPerformanceEvaluator().getError(trainedNetwork, testing);

        // a reasonable score assuming only 3000 images in the training set and no regularization
        assertThat(error).isLessThan(0.15);
    }

    private DataSet[] loadData() {
        DataSet allMnistDigits = new MnistDigitsDataService().loadDataFile();
        // Only use 5% of images in each set to speed up test
        return new DataSetSplitter().split(allMnistDigits.shuffleRows(), 0.05, 0.1, 0.1);
    }

}
