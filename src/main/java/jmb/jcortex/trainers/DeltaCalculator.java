/*
 * James Brundege
 * Date: 2017-04-22
 * MIT license: https://opensource.org/licenses/MIT
 */
package jmb.jcortex.trainers;

import jmb.jcortex.data.SynMatrix;
import jmb.jcortex.neuralnet.NeuralNet;

import java.util.Arrays;
import java.util.List;

/**
 *
 */
public class DeltaCalculator {

    public List<SynMatrix> calcDeltas(List<SynMatrix> nodeValues, SynMatrix labels, NeuralNet neuralNet) {
        List<SynMatrix> layers = neuralNet.getLayers();
        List<SynMatrix> dropoutMasks = neuralNet.getDropoutMasks();
        SynMatrix[] deltas = new SynMatrix[layers.size()];
        SynMatrix outputs = nodeValues.get(nodeValues.size() - 1);
        deltas[deltas.length - 1] = outputs.minus(labels).elementMultInPlace(outputs.apply(neuralNet.getOutputFunction().getDerivative()));
        for (int i = deltas.length - 1; i >= 1; i--) {
            SynMatrix weightsNoBias = layers.get(i).removeBiasRow();
            SynMatrix activationDerivative = nodeValues.get(i).apply(neuralNet.getActivationFunction().getDerivative());
            deltas[i - 1] = (deltas[i].multiply(weightsNoBias.transpose())).elementMultInPlace(activationDerivative).elementMultInPlace(dropoutMasks.get(i-1));
        }
        return Arrays.asList(deltas);
    }

}
