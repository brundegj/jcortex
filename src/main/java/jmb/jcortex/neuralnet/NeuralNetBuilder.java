/*
 * James Brundege
 * Date: 2017-04-09
 * MIT license: https://opensource.org/licenses/MIT
 */
package jmb.jcortex.neuralnet;

import jmb.jcortex.mapfunctions.DifferentiableMatrixFunction;
import jmb.jcortex.strategies.weightinitializers.LinearRandomWeightInitializer;
import jmb.jcortex.strategies.weightinitializers.WeightInitializer;

import static jmb.jcortex.mapfunctions.MatrixFunctions.LINEAR_MATRIX_FUNCTION;
import static jmb.jcortex.mapfunctions.MatrixFunctions.SIGMOID_MATRIX_FUNCTION;

public class NeuralNetBuilder {

    public static NeuralNetBuilder createNeuralNet() {
        return new NeuralNetBuilder();
    }

    private int[] dimensions;
    private DifferentiableMatrixFunction activationFunction;
    private DifferentiableMatrixFunction outputFunction;
    private WeightInitializer weightInitializer;

    private double hiddenDropoutPercent = 0.0;

    public NeuralNetBuilder withDimensions(int... dimensions) {
        this.dimensions = dimensions;

        // Set defaults
        activationFunction = SIGMOID_MATRIX_FUNCTION;
        if (dimensions[dimensions.length-1] == 1) {
            outputFunction = LINEAR_MATRIX_FUNCTION;
        } else {
            outputFunction = SIGMOID_MATRIX_FUNCTION;
        }
        weightInitializer = new LinearRandomWeightInitializer(-0.04, 0.04);

        return this;
    }

    public NeuralNetBuilder withActivationFunction(DifferentiableMatrixFunction activationFunction) {
        this.activationFunction = activationFunction;
        return this;
    }

    public NeuralNetBuilder withOutputFunction(DifferentiableMatrixFunction outputFunction) {
        this.outputFunction = outputFunction;
        return this;
    }

    public NeuralNetBuilder withWeightInitializer(WeightInitializer weightInitializer) {
        this.weightInitializer = weightInitializer;
        return this;
    }

    public NeuralNetBuilder withDropout(double hiddenDropoutPercent) {
        this.hiddenDropoutPercent = hiddenDropoutPercent;
        return this;
    }

    public NeuralNet build() {
        NeuralNet neuralNet = new NeuralNet(dimensions);
        neuralNet.setActivationFunction(activationFunction);
        neuralNet.setOutputFunction(outputFunction);
        neuralNet.setWeightInitializer(weightInitializer);
        neuralNet.setHiddenDropoutPercent(hiddenDropoutPercent);
        return neuralNet;
    }

}
