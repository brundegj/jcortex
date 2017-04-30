/*
 * James Brundege
 * Date: 2017-04-23
 * MIT license: https://opensource.org/licenses/MIT
 */
package jmb.jcortex.trainers;

import jmb.jcortex.data.SynMatrix;

import java.util.Arrays;
import java.util.List;

/**
 *
 */
public class GradientCalculator {

    public List<SynMatrix> calcGradients(List<SynMatrix> deltasList, List<SynMatrix> nodeValuesList) {
        SynMatrix[] nodeValues = nodeValuesList.toArray(new SynMatrix[nodeValuesList.size()]);
        SynMatrix[] deltas = deltasList.toArray(new SynMatrix[deltasList.size()]);
        SynMatrix[] gradients = new SynMatrix[deltas.length];
        for (int i = 0; i < deltas.length; i++) {
            gradients[i] = nodeValues[i].addBiasColumn().transpose().multiply(deltas[i]).elementDivideInPlace(deltas[i].numRows());
        }
        return Arrays.asList(gradients);
    }

//
//        return StreamUtils.zip(deltas.stream(), nodeValues.stream(), this::calcPartialDerivativeGradient)
//                .collect(toList());
//    }
//
//    private SynMatrix calcPartialDerivativeGradient(SynMatrix deltas, SynMatrix nodeValues) {
//        Theta2_grad = (Delta3' * A) ./ m;
//        return nodeValues.addBiasColumn().transpose().multiply(deltas).elementDivideInPlace(deltas.numRows());
//    }

}
