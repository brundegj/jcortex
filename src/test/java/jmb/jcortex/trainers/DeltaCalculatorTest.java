/*
 * James Brundege
 * Date: 2017-04-22
 * MIT license: https://opensource.org/licenses/MIT
 */
package jmb.jcortex.trainers;

import jmb.jcortex.data.SynMatrix;
import jmb.jcortex.mapfunctions.DifferentiableMatrixFunction;
import jmb.jcortex.mapfunctions.DoublingActivationFunction;
import jmb.jcortex.mapfunctions.SimpleDifferentiableMatrixFunction;
import jmb.jcortex.neuralnet.NeuralNet;
import org.junit.Test;

import java.util.Arrays;
import java.util.List;

import static java.util.Collections.singletonList;
import static org.assertj.core.api.Assertions.assertThat;

public class DeltaCalculatorTest {

    private DifferentiableMatrixFunction doublingActivationFunction
            = new SimpleDifferentiableMatrixFunction(new DoublingActivationFunction());

    @Test
    public void testCalcDeltas_OnSingleLayer() {

        SynMatrix input = new SynMatrix(new double[][]{
                {0.5, -1, -0.4},
                {1, -0.5, 0.4}
        });
        SynMatrix layer = new SynMatrix(new double[][]{
                {-0.1, 0.1},    // 1st row is the bias
                {0.1, -0.2},
                {-0.1, 0.2},
                {0.15, -0.15}
        });
        // output = input (plus bias column) x layer x activationDerivative
        SynMatrix output = new SynMatrix(new double[][]{
                {-0.01, -0.14},
                {0.11, -0.26}
        });
        SynMatrix labels = new SynMatrix(new double[][]{
                {-0.1, -0.3},
                {0.30, -0.1}
        });
        // deltas = output - labels
        SynMatrix deltas = new SynMatrix(new double[][]{
                {0.18, 0.32},
                {-0.38, -0.32}
        });

        DeltaCalculator deltaCalculator = new DeltaCalculator();
        List<SynMatrix> nodeValues = Arrays.asList(input, output);
        List<SynMatrix> layers = singletonList(layer);
        NeuralNet neuralNet = new NeuralNet();
        neuralNet.setLayers(layers);
        neuralNet.setActivationFunction(doublingActivationFunction);
        neuralNet.setOutputFunction(doublingActivationFunction);

        List<SynMatrix> actualDeltas = deltaCalculator.calcDeltas(nodeValues, labels, neuralNet);

        assertThat(actualDeltas).hasSize(1);
        assertThat(actualDeltas.get(0)).isEqualTo(deltas);
    }

    @Test
    public void testCalcDeltas_OnTwoLayers() {

        SynMatrix input = new SynMatrix(new double[][]{
                {0.5, -1, -0.4},
                {1, -0.5, 0.4}
        });
        SynMatrix layer1 = new SynMatrix(new double[][]{
                {-0.1, 0.1},    // 1st row is the bias
                {0.1, -0.2},
                {-0.1, 0.2},
                {0.15, -0.15}
        });
        // middle = input (plus bias column) x layer * 2
        SynMatrix middleNodeValues = new SynMatrix(new double[][]{
                {-0.02, -0.28},
                {0.22, -0.52}
        });
        SynMatrix layer2 = new SynMatrix(new double[][]{
                {-0.2, 0.1},    // 1st row is the bias
                {-0.1, 0.2},
                {0.15, -0.15}
        });
        SynMatrix output = new SynMatrix(new double[][]{
                {-0.24, -0.138},
                {-0.3, 0.222}
        });
        SynMatrix labels = new SynMatrix(new double[][]{
                {-0.1, -0.3},
                {0.30, -0.1}
        });
        // deltas1 = (output - labels) * activationDerivative
        SynMatrix deltas1 = new SynMatrix(new double[][]{
                {-0.28, 0.324},
                {-1.2, 0.644}
        });
        // deltas2 = deltas1 X layer2NoBias * 2
        SynMatrix deltas2 = new SynMatrix(new double[][]{
                {0.185600, -0.181200},
                {0.497600, -0.553200}
        });

        DeltaCalculator deltaCalculator = new DeltaCalculator();
        List<SynMatrix> nodeValues = Arrays.asList(input, middleNodeValues, output);
        List<SynMatrix> layers = Arrays.asList(layer1, layer2);
        NeuralNet neuralNet = new NeuralNet();
        neuralNet.setDropoutMasks(singletonList(SynMatrix.ones(middleNodeValues.numRows(), middleNodeValues.numCols())));
        neuralNet.setLayers(layers);
        neuralNet.setActivationFunction(doublingActivationFunction);
        neuralNet.setOutputFunction(doublingActivationFunction);

        List<SynMatrix> actualDeltas = deltaCalculator.calcDeltas(nodeValues, labels, neuralNet);

        assertThat(actualDeltas).hasSize(2);
        assertThat(actualDeltas.get(1)).isEqualTo(deltas1);
        assertThat(actualDeltas.get(0)).isEqualTo(deltas2);
    }

    @Test
    public void testCalcDeltas_WithDropout() {

        SynMatrix input = new SynMatrix(new double[][]{
                {0.5, -1, -0.4},
                {1, -0.5, 0.4}
        });
        SynMatrix layer1 = new SynMatrix(new double[][]{
                {-0.1, 0.1},    // 1st row is the bias
                {0.1, -0.2},
                {-0.1, 0.2},
                {0.15, -0.15}
        });
        // middle = input (plus bias column) x layer * 2
        SynMatrix middleNodeValues = new SynMatrix(new double[][]{
                {-0.02, -0.28},
                {0.22, -0.52}
        });
        SynMatrix dropoutMask = new SynMatrix(new double[][]{
                {1.0, 0.0},
                {0.0, 1.0}
        });
        SynMatrix layer2 = new SynMatrix(new double[][]{
                {-0.2, 0.1},    // 1st row is the bias
                {-0.1, 0.2},
                {0.15, -0.15}
        });
        SynMatrix output = new SynMatrix(new double[][]{
                {-0.24, -0.138},
                {-0.3, 0.222}
        });
        SynMatrix labels = new SynMatrix(new double[][]{
                {-0.1, -0.3},
                {0.30, -0.1}
        });
        // deltas1 = (output - labels) * activationDerivative
        SynMatrix deltas1 = new SynMatrix(new double[][]{
                {-0.28, 0.324},
                {-1.2, 0.644}
        });
        // deltas2 = deltas1 X layer2NoBias * 2 * dropoutMask
        SynMatrix deltas2 = new SynMatrix(new double[][]{
                {0.185600, 0.0},
                {0.0, -0.553200}
        });

        DeltaCalculator deltaCalculator = new DeltaCalculator();
        List<SynMatrix> nodeValues = Arrays.asList(input, middleNodeValues, output);
        List<SynMatrix> layers = Arrays.asList(layer1, layer2);
        NeuralNet neuralNet = new NeuralNet();
        neuralNet.setDropoutMasks(singletonList(dropoutMask));
        neuralNet.setLayers(layers);
        neuralNet.setActivationFunction(doublingActivationFunction);
        neuralNet.setOutputFunction(doublingActivationFunction);

        List<SynMatrix> actualDeltas = deltaCalculator.calcDeltas(nodeValues, labels, neuralNet);

        assertThat(actualDeltas).hasSize(2);
        assertThat(actualDeltas.get(1)).isEqualTo(deltas1);
        assertThat(actualDeltas.get(0)).isEqualTo(deltas2);
    }

}
