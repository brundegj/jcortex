/*
 * James Brundege
 * Date: 2017-04-09
 * MIT license: https://opensource.org/licenses/MIT
 */
package jmb.jcortex.neuralnet;

import jmb.jcortex.data.Copyable;
import jmb.jcortex.data.DataSet;
import jmb.jcortex.data.SynMatrix;
import jmb.jcortex.mapfunctions.DifferentiableMatrixFunction;
import jmb.jcortex.strategies.weightinitializers.WeightInitializer;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.IntStream;

import static java.util.stream.Collectors.toList;

public class NeuralNet implements Copyable<NeuralNet> {

    private List<SynMatrix> layers;
    private DifferentiableMatrixFunction activationFunction;
    private DifferentiableMatrixFunction outputFunction;
    private WeightInitializer weightInitializer;

    public NeuralNet(List<SynMatrix> layers) {
        this.layers = layers;
    }

    public NeuralNet(int... dimensions) {
        if (dimensions.length < 2) {
            throw new IllegalArgumentException("Must have at least 2 layers in a NeuralNet");
        }

        layers = new ArrayList<>();
        for (int i = 1; i < dimensions.length; i++) {
            int input = dimensions[i-1];
            int output = dimensions[i];
            // input+1 to allow for the bias node
            layers.add(new SynMatrix(input+1, output));
        }
    }

    private void initializeWeights() {
        layers = layers.parallelStream().map(weightInitializer::initialize).collect(toList());
    }

    public DifferentiableMatrixFunction getActivationFunction() {
        return activationFunction;
    }

    public void setActivationFunction(DifferentiableMatrixFunction activationFunction) {
        this.activationFunction = activationFunction;
    }

    public DifferentiableMatrixFunction getOutputFunction() {
        return outputFunction;
    }

    public void setOutputFunction(DifferentiableMatrixFunction outputFunction) {
        this.outputFunction = outputFunction;
    }

    public WeightInitializer getWeightInitializer() {
        return weightInitializer;
    }

    /**
     * Sets the WeightInitializer AND initializes the weights. This will effectively mutate the layers of this
     * NeuralNet in place and wipe out any prior traing history.
     */
    public void setWeightInitializer(WeightInitializer weightInitializer) {
        this.weightInitializer = weightInitializer;
        initializeWeights();
    }

    public List<SynMatrix> getLayers() {
        return new ArrayList<>(layers);
    }

    public void setLayers(List<SynMatrix> layers) {
        this.layers = layers;
    }

    public List<SynMatrix> doForwardPass(DataSet batch) {
        List<SynMatrix> nodeValues = new ArrayList<>();
        nodeValues.add(batch.getFeatures());
        IntStream.range(0, layers.size()).forEach(index -> {
            SynMatrix nodeVector = nodeValues.get(index).addBiasColumn().multiply(layers.get(index));
            if (index < layers.size()-1) {
                nodeVector = nodeVector.apply(activationFunction.getFunction());
            } else {
                nodeVector = nodeVector.apply(outputFunction.getFunction());
            }
            nodeValues.add(nodeVector);
        });
        return nodeValues;
    }

    /**
     * Do a forward pass through the given DataSet and return the output vector.
     */
    public SynMatrix analyzeData(DataSet dataSet) {
        List<SynMatrix> nodeValues = doForwardPass(dataSet);
        return nodeValues.get(nodeValues.size() - 1);
    }

    public NeuralNet copy() {
        NeuralNet copy = new NeuralNet(this.layers.stream().map(SynMatrix::copy).collect(toList()));
        copy.activationFunction = this.activationFunction;
        copy.outputFunction = this.outputFunction;
        copy.weightInitializer = this.weightInitializer;
        return copy;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        NeuralNet neuralNet = (NeuralNet) o;

        if (!getLayers().equals(neuralNet.getLayers())) return false;
        if (!getActivationFunction().equals(neuralNet.getActivationFunction())) return false;
        return getOutputFunction().equals(neuralNet.getOutputFunction());
    }

    @Override
    public int hashCode() {
        int result = getLayers().hashCode();
        result = 31 * result + getActivationFunction().hashCode();
        result = 31 * result + getOutputFunction().hashCode();
        return result;
    }
}
