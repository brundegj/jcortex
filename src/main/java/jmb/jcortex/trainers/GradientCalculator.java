/*
 * James Brundege
 * Date: 2017-04-23
 * MIT license: https://opensource.org/licenses/MIT
 */
package jmb.jcortex.trainers;

import com.codepoetics.protonpack.StreamUtils;
import jmb.jcortex.data.SynMatrix;

import java.util.List;

import static java.util.stream.Collectors.toList;

/**
 *
 */
public class GradientCalculator {

    public List<SynMatrix> calcGradients(List<SynMatrix> deltasList, List<SynMatrix> nodeValuesList) {
        return StreamUtils.zip(deltasList.stream(), nodeValuesList.stream(), this::calcPartialDerivativeGradient)
                .collect(toList());
    }

    private SynMatrix calcPartialDerivativeGradient(SynMatrix deltas, SynMatrix nodeValues) {
        return nodeValues.addBiasColumn().transpose().multiply(deltas).elementDivideInPlace(deltas.numRows());
    }

}
