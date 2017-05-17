/*
 * James Brundege
 * Date: 2017-04-16
 * MIT license: https://opensource.org/licenses/MIT
 */
package jmb.jcortex.neuralnet;

import jmb.jcortex.data.DataSet;
import jmb.jcortex.data.SynMatrix;
import jmb.jcortex.strategies.weightinitializers.OnesWeightInitializer;
import org.junit.Test;

import java.util.Arrays;
import java.util.List;
import java.util.Random;

import static jmb.jcortex.mapfunctions.MatrixFunctions.LINEAR_MATRIX_FUNCTION;
import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

public class NeuralNetTest {

    @Test
    public void constructor_SetsDimensions() {
        NeuralNet neuralNet = new NeuralNet(8, 6, 4, 2);
        List<SynMatrix> layers = neuralNet.getLayers();
        assertThat(layers).hasSize(3);

        // verify dimensions
        assertThat(layers.get(0).numRows()).isEqualTo(9);   // features + 1 for bias node
        assertThat(layers.get(0).numCols()).isEqualTo(6);
        assertThat(layers.get(1).numRows()).isEqualTo(7);   // + 1 for bias node
        assertThat(layers.get(1).numCols()).isEqualTo(4);
        assertThat(layers.get(2).numRows()).isEqualTo(5);   // + 1 for bias node
        assertThat(layers.get(2).numCols()).isEqualTo(2);
    }

    @Test
    public void setWeightInitializer_InitializesAllWeights() {
        NeuralNet neuralNet = new NeuralNet(8, 6, 4, 2);
        neuralNet.setWeightInitializer(new OnesWeightInitializer());
        List<SynMatrix> layers = neuralNet.getLayers();
        assertThat(layers).hasSize(3);

        // verify every element is initialized
        layers.parallelStream()
                .forEach(layer -> Arrays.stream(layer.getAll())
                        .forEach(value -> assertThat(value).isEqualTo(1)));
    }

    @Test
    public void trainForward_MultipliesAndSumsInputsToNodeValues() {
        NeuralNet neuralNet = new NeuralNet(4, 3, 2);
        // Set every weight to 2
        neuralNet.setWeightInitializer(matrix -> SynMatrix.ones(matrix.numRows(), matrix.numCols()).elementMultInPlace(2));
        neuralNet.setActivationFunction(LINEAR_MATRIX_FUNCTION);
        neuralNet.setOutputFunction(LINEAR_MATRIX_FUNCTION);

        // For each sample, the above NeuralNet should take the 4 inputs plus the bias, multiply by 2, and sum into
        // each middle node. The resulting 3 identical values, plus the bias, should be multiplied by 2 and summed
        // into each of the output neurons.

        double[][] values = new double[][] {
                {1, 2, 3, 4},
                {5, 6, 7, 8}
        };
        SynMatrix features = new SynMatrix(values);
        DataSet dataSet = new DataSet(features, SynMatrix.ones(2, 4));

        // Sample 1: each middle node = 22, each output node = 22 * 2 * 3 + 2 = 134
        // Sample 2: each middle node = 54, each output node = 54 * 2 * 3 + 2 = 326

        List<SynMatrix> output = neuralNet.trainForward(dataSet);

        assertThat(output).hasSize(3);  // middle nodes and output nodes
        SynMatrix inputNodes = output.get(0);
        assertThat(inputNodes.numRows()).isEqualTo(2);
        assertThat(inputNodes.numCols()).isEqualTo(4);
        assertThat(inputNodes.getRow(0)).containsExactly(1, 2, 3, 4);
        assertThat(inputNodes.getRow(1)).containsExactly(5, 6, 7, 8);
        SynMatrix middleNodes = output.get(1);
        assertThat(middleNodes.numRows()).isEqualTo(2);
        assertThat(middleNodes.numCols()).isEqualTo(3);
        assertThat(middleNodes.getRow(0)).containsExactly(22, 22, 22);
        assertThat(middleNodes.getRow(1)).containsExactly(54, 54, 54);
        SynMatrix outputNodes = output.get(2);
        assertThat(outputNodes.numRows()).isEqualTo(2);
        assertThat(outputNodes.numCols()).isEqualTo(2);
        assertThat(outputNodes.getRow(0)).containsExactly(134, 134);
        assertThat(outputNodes.getRow(1)).containsExactly(326, 326);
    }

    @Test
    public void trainForward_WithDropout_BlanksMiddleLayerNodes() {
        NeuralNet neuralNet = new NeuralNet(4, 3, 2);
        // Set every weight to 2
        neuralNet.setWeightInitializer(matrix -> SynMatrix.ones(matrix.numRows(), matrix.numCols()).elementMultInPlace(2));
        neuralNet.setActivationFunction(LINEAR_MATRIX_FUNCTION);
        neuralNet.setOutputFunction(LINEAR_MATRIX_FUNCTION);

        neuralNet.setHiddenDropoutPercent(0.5);
        Random mockRandom = mock(Random.class);
        // This should create a dropoutMask that looks like:
        //      {{1.0, 0.0, 1.0},
        //      {0.0, 1.0, 0.0}}
        // Note that the order looks off because the apply to the 3x2 matrix does *columnwise*
        when(mockRandom.nextDouble()).thenReturn(0.6).thenReturn(0.4).thenReturn(0.4)
                                     .thenReturn(0.6).thenReturn(0.6).thenReturn(0.4);
        neuralNet.setDropoutRandomizer(mockRandom);

        double[][] values = new double[][] {
                {1, 2, 3, 4},
                {5, 6, 7, 8}
        };
        SynMatrix features = new SynMatrix(values);
        DataSet dataSet = new DataSet(features, SynMatrix.ones(2, 4));

        // Sample 1: middle nodes = {22, 0, 22}, each output node = 22 * 2 * 2 + 2 = 90
        // Sample 2: middle nodes = {0, 54, 0}, each output node = 54 * 2 * 1 + 2 = 110

        List<SynMatrix> output = neuralNet.trainForward(dataSet);

        assertThat(output).hasSize(3);  // middle nodes and output nodes
        SynMatrix inputNodes = output.get(0);
        assertThat(inputNodes.numRows()).isEqualTo(2);
        assertThat(inputNodes.numCols()).isEqualTo(4);
        assertThat(inputNodes.getRow(0)).containsExactly(1, 2, 3, 4);
        assertThat(inputNodes.getRow(1)).containsExactly(5, 6, 7, 8);
        SynMatrix middleNodes = output.get(1);
        assertThat(middleNodes.numRows()).isEqualTo(2);
        assertThat(middleNodes.numCols()).isEqualTo(3);
        assertThat(middleNodes.getRow(0)).containsExactly(22, 0, 22);
        assertThat(middleNodes.getRow(1)).containsExactly(0, 54, 0);
        SynMatrix outputNodes = output.get(2);
        assertThat(outputNodes.numRows()).isEqualTo(2);
        assertThat(outputNodes.numCols()).isEqualTo(2);
        assertThat(outputNodes.getRow(0)).containsExactly(90, 90);
        assertThat(outputNodes.getRow(1)).containsExactly(110, 110);
    }

    @Test
    public void analyzeData_DoesAForwardPass_AndReturnsOutputNodesOnly() {
        NeuralNet neuralNet = new NeuralNet(4, 3, 2);
        // Set every weight to 2
        neuralNet.setWeightInitializer(matrix -> SynMatrix.ones(matrix.numRows(), matrix.numCols()).elementMultInPlace(2));
        neuralNet.setActivationFunction(LINEAR_MATRIX_FUNCTION);
        neuralNet.setOutputFunction(LINEAR_MATRIX_FUNCTION);
        double[][] values = new double[][] {
                {1, 2, 3, 4},
                {5, 6, 7, 8}
        };
        SynMatrix features = new SynMatrix(values);
        DataSet dataSet = new DataSet(features, SynMatrix.ones(2, 4));

        // Sample 1: each middle node = 22, each output node = 22 * 2 * 3 + 2 = 134
        // Sample 2: each middle node = 54, each output node = 54 * 2 * 3 + 2 = 326

        SynMatrix outputNodes = neuralNet.analyzeData(dataSet);
        assertThat(outputNodes.numRows()).isEqualTo(2);
        assertThat(outputNodes.numCols()).isEqualTo(2);
        assertThat(outputNodes.getRow(0)).containsExactly(134, 134);
        assertThat(outputNodes.getRow(1)).containsExactly(326, 326);
    }

    @Test
    public void analyzeData_WithDropout_HalvesTheNodeValues() {
        NeuralNet neuralNet = new NeuralNet(4, 3, 2);
        // Set every weight to 2
        neuralNet.setWeightInitializer(matrix -> SynMatrix.ones(matrix.numRows(), matrix.numCols()).elementMultInPlace(2));
        neuralNet.setActivationFunction(LINEAR_MATRIX_FUNCTION);
        neuralNet.setOutputFunction(LINEAR_MATRIX_FUNCTION);
        neuralNet.setHiddenDropoutPercent(0.5);

        double[][] values = new double[][] {
                {1, 2, 3, 4},
                {5, 6, 7, 8}
        };
        SynMatrix features = new SynMatrix(values);
        DataSet dataSet = new DataSet(features, SynMatrix.ones(2, 4));

        // Sample 1: each middle node = 22, each output node = 22 * 1 * 3 + 2 = 68
        // Sample 2: each middle node = 54, each output node = 54 * 1 * 3 + 2 = 164

        SynMatrix outputNodes = neuralNet.analyzeData(dataSet);
        assertThat(outputNodes.numRows()).isEqualTo(2);
        assertThat(outputNodes.numCols()).isEqualTo(2);
        assertThat(outputNodes.getRow(0)).containsExactly(68, 68);
        assertThat(outputNodes.getRow(1)).containsExactly(164, 164);
    }

}