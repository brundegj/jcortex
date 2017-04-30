/*
 * James Brundege
 * Date: 2017-04-16
 * MIT license: https://opensource.org/licenses/MIT
 */
package jmb.jcortex.neuralnet;

import jmb.jcortex.data.SynMatrix;
import jmb.jcortex.strategies.weightinitializers.LinearRandomWeightInitializer;
import org.junit.Test;

import java.util.List;

import static jmb.jcortex.mapfunctions.MatrixFunctions.LINEAR_MATRIX_FUNCTION;
import static jmb.jcortex.mapfunctions.MatrixFunctions.SIGMOID_MATRIX_FUNCTION;
import static org.assertj.core.api.Assertions.assertThat;

public class NeuralNetBuilderTest {

    @Test
    public void testBuilder() {
        LinearRandomWeightInitializer weightInitializer = new LinearRandomWeightInitializer(-0.6, 0.6);

        NeuralNet neuralNet = NeuralNetBuilder.createNeuralNet()
                .withDimensions(8, 6, 4, 2)
                .withWeightInitializer(weightInitializer)
                .withActivationFunction(SIGMOID_MATRIX_FUNCTION)
                .withOutputFunction(LINEAR_MATRIX_FUNCTION)
                .build();

        assertThat(neuralNet.getWeightInitializer()).isSameAs(weightInitializer);
        assertThat(neuralNet.getActivationFunction()).isSameAs(SIGMOID_MATRIX_FUNCTION);
        assertThat(neuralNet.getOutputFunction()).isSameAs(LINEAR_MATRIX_FUNCTION);
        List<SynMatrix> layers = neuralNet.getLayers();
        assertThat(layers).hasSize(3);
        assertThat(layers.get(0).numRows()).isEqualTo(9);   // + 1 for bias node
        assertThat(layers.get(0).numCols()).isEqualTo(6);
        assertThat(layers.get(1).numRows()).isEqualTo(7);   // + 1 for bias node
        assertThat(layers.get(1).numCols()).isEqualTo(4);
        assertThat(layers.get(2).numRows()).isEqualTo(5);   // + 1 for bias node
        assertThat(layers.get(2).numCols()).isEqualTo(2);
    }

}