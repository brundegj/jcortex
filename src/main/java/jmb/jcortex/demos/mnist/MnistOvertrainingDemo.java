/*
 * James Brundege
 * Date: 2017-05-01
 * MIT license: https://opensource.org/licenses/MIT
 */
package jmb.jcortex.demos.mnist;

import com.codepoetics.protonpack.maps.MapStream;
import jmb.jcortex.data.DataSet;
import jmb.jcortex.data.DataSetSplitter;
import jmb.jcortex.datasource.MnistDigitsDataService;
import jmb.jcortex.neuralnet.NeuralNet;
import jmb.jcortex.neuralnet.NeuralNetBuilder;
import jmb.jcortex.strategies.batchingstrategies.FixedNumBatchingStrategy;
import jmb.jcortex.strategies.haltingstrategies.ValidationSetHaltingStrategy;
import jmb.jcortex.strategies.optimizationstrategies.FixedLearningRateOptimizationStrategy;
import jmb.jcortex.strategies.performanceevaluators.ChartingPerformanceListener;
import jmb.jcortex.strategies.performanceevaluators.ClassificationPerformanceEvaluator;
import jmb.jcortex.strategies.performanceevaluators.PrintlnPerformanceListener;
import jmb.jcortex.strategies.weightinitializers.LinearRandomWeightInitializer;
import jmb.jcortex.trainers.GradientDescentTrainerBuilder;
import jmb.jcortex.trainers.SupervisedTrainer;

import static jmb.jcortex.JCortexConstants.TEST_SET_PERCENT_WRONG;
import static jmb.jcortex.mapfunctions.MatrixFunctions.RECIFIED_LINEAR_MATRIX_FUNCTION;
import static jmb.jcortex.mapfunctions.MatrixFunctions.SOFTMAX_MATRIX_FUNCTION;

/**
 *
 */
public class MnistOvertrainingDemo {

    public static void main(String[] args) {
        MnistOvertrainingDemo demo = new MnistOvertrainingDemo();
        demo.runDemo();
    }

    private void runDemo() {
        DataSet[] sets = loadData();
        DataSet training = sets[0];
        DataSet validation = sets[1];
        DataSet test = sets[2];

        NeuralNet untrainedNetwork = NeuralNetBuilder.createNeuralNet()
                .withDimensions(784, 200, 200, 10)
                .withActivationFunction(RECIFIED_LINEAR_MATRIX_FUNCTION)
                .withOutputFunction(SOFTMAX_MATRIX_FUNCTION)
                .withWeightInitializer(new LinearRandomWeightInitializer(-0.3, 0.3))
                .withDropout(0.2)
                .build();

        ClassificationPerformanceEvaluator performanceEvaluator = new ClassificationPerformanceEvaluator();
        PrintlnPerformanceListener printlnPerformanceListener = new PrintlnPerformanceListener();

        ValidationSetHaltingStrategy haltingStrategy = new ValidationSetHaltingStrategy(training, validation,
                performanceEvaluator, 10);
        haltingStrategy.addPerformanceListener(printlnPerformanceListener);
        haltingStrategy.addPerformanceListener(new ChartingPerformanceListener(100));

        SupervisedTrainer trainer = GradientDescentTrainerBuilder.createTrainer()
                .withBatchingStrategy(new FixedNumBatchingStrategy(100))
                .withOptimizationStrategy(new FixedLearningRateOptimizationStrategy(0.05))
//                .withOptimizationStrategy(new MomentumOptimizationStrategy(0.08, 0.7))
                .withHaltingStrategy(haltingStrategy)
                .build();

        NeuralNet trained = trainer.train(untrainedNetwork, training);

        // Report the performance on the test set
        printlnPerformanceListener.performanceEvent(MapStream.of(TEST_SET_PERCENT_WRONG, performanceEvaluator.getError(trained, test)).collect());
    }

    private DataSet[] loadData() {
        DataSet allMnistDigits = new MnistDigitsDataService().loadDataFile();
        // Only use 2% of images in the training set. This exacerbates the overfitting for the demo.
        return new DataSetSplitter().split(allMnistDigits.shuffleRows(), 0.02, 0.1, 0.1);
    }
}
