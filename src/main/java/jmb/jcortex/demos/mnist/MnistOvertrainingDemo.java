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
import jmb.jcortex.strategies.batchingstrategies.GaussianNoiseBatchingStrategy;
import jmb.jcortex.strategies.haltingstrategies.HaltingStrategy;
import jmb.jcortex.strategies.haltingstrategies.ValidationSetHaltingStrategy;
import jmb.jcortex.strategies.optimizationstrategies.MomentumOptimizationStrategy;
import jmb.jcortex.strategies.performanceevaluators.ChartingPerformanceListener;
import jmb.jcortex.strategies.performanceevaluators.ClassificationPerformanceEvaluator;
import jmb.jcortex.strategies.performanceevaluators.PrintlnPerformanceListener;
import jmb.jcortex.strategies.regularization.L2Regularization;
import jmb.jcortex.strategies.weightinitializers.LinearRandomWeightInitializer;
import jmb.jcortex.trainers.GradientDescentTrainerBuilder;
import jmb.jcortex.trainers.SupervisedTrainer;

import static jmb.jcortex.JCortexConstants.TEST_SET_PERCENT_WRONG;
import static jmb.jcortex.mapfunctions.MatrixFunctions.RECIFIED_LINEAR_MATRIX_FUNCTION;
import static jmb.jcortex.mapfunctions.MatrixFunctions.SOFTMAX_MATRIX_FUNCTION;

/**
 *  A Demo to demonstrate overfitting and regularization. Trains a digit recognizer on the MNIST digit data set.
 *  This demo trains with only 5% of the MNIST training samples (3000 images) to make the task much harder, thus showing
 *  severe overfitting. A variety of regularization techniques can be combined (L2 weight penalty, dropout,
 *  synthetic training data using gaussian noise) to reduce the overfitting.
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

        NeuralNet trainedNoRegularization = trainWithoutRegularization(training, getHaltingStrategy("MNIST training WITHOUT regularization", training, validation, 10));
        NeuralNet trainedWithRegularization = trainWithRegularization(training, getHaltingStrategy("MNIST training WITH regularization", training, validation, 30));

        // Report the performance on the test set
        ClassificationPerformanceEvaluator performanceEvaluator = new ClassificationPerformanceEvaluator();
        PrintlnPerformanceListener performanceListener = new PrintlnPerformanceListener();
        System.out.println("Without regularization:");
        performanceListener.performanceEvent(MapStream.of(TEST_SET_PERCENT_WRONG, performanceEvaluator.getError(trainedNoRegularization, test)).collect());
        System.out.println("With regularization:");
        performanceListener.performanceEvent(MapStream.of(TEST_SET_PERCENT_WRONG, performanceEvaluator.getError(trainedWithRegularization, test)).collect());
    }

    private NeuralNet trainWithoutRegularization(DataSet trainingSet, HaltingStrategy haltingStrategy) {
        NeuralNet untrained = NeuralNetBuilder.createNeuralNet()
                .withDimensions(784, 200, 200, 10)
                .withActivationFunction(RECIFIED_LINEAR_MATRIX_FUNCTION)
                .withOutputFunction(SOFTMAX_MATRIX_FUNCTION)
                .withWeightInitializer(new LinearRandomWeightInitializer(-0.3, 0.3))
                .build();

        SupervisedTrainer trainer = GradientDescentTrainerBuilder.createTrainer()
                .withBatchingStrategy(new FixedNumBatchingStrategy(100))
                .withOptimizationStrategy(new MomentumOptimizationStrategy(0.08, 0.7))
                .withHaltingStrategy(haltingStrategy)
                .build();

        return trainer.train(untrained, trainingSet);
    }

    private NeuralNet trainWithRegularization(DataSet training, HaltingStrategy haltingStrategy) {
        NeuralNet untrained = NeuralNetBuilder.createNeuralNet()
                .withDimensions(784, 200, 200, 10)
                .withActivationFunction(RECIFIED_LINEAR_MATRIX_FUNCTION)
                .withOutputFunction(SOFTMAX_MATRIX_FUNCTION)
                .withWeightInitializer(new LinearRandomWeightInitializer(-0.3, 0.3))
                .withDropout(0.2)   // dropout
                .build();

        SupervisedTrainer trainer = GradientDescentTrainerBuilder.createTrainer()
                .withBatchingStrategy(new GaussianNoiseBatchingStrategy(100, 0.4, training))    // gaussian noise synthetic data
                .withOptimizationStrategy(new MomentumOptimizationStrategy(0.08, 0.7))
                .withHaltingStrategy(haltingStrategy)
                .withWeightAdjuster(new L2Regularization(0.1))  // L2 weight decay
                .build();

        return trainer.train(untrained, training);
    }

    private HaltingStrategy getHaltingStrategy(String title, DataSet training, DataSet validation, int maxIterationSinceBestResult) {
        ClassificationPerformanceEvaluator performanceEvaluator = new ClassificationPerformanceEvaluator();
        PrintlnPerformanceListener printlnPerformanceListener = new PrintlnPerformanceListener();
        ValidationSetHaltingStrategy haltingStrategy = new ValidationSetHaltingStrategy(
                training, validation, performanceEvaluator, maxIterationSinceBestResult);
        haltingStrategy.addPerformanceListener(printlnPerformanceListener);
        haltingStrategy.addPerformanceListener(new ChartingPerformanceListener(title, 30));
        return haltingStrategy;
    }

    private DataSet[] loadData() {
        DataSet allMnistDigits = new MnistDigitsDataService().loadDataFile();
        // Only use 5% of images in the training set (3000 images). This exacerbates overfitting for the demo.
        return new DataSetSplitter().split(allMnistDigits.shuffleRows(), 0.05, 0.1, 0.1);
    }
}
