/*
 * James Brundege
 * Date: 2017-04-09
 * MIT license: https://opensource.org/licenses/MIT
 */
package jmb.jcortex.mapfunctions;

import jmb.jcortex.data.SynMatrix;
import org.apache.commons.math3.util.FastMath;

import java.util.Arrays;

public class SoftMaxActivationFunction implements MatrixFunction, RowFunction {

    @Override
    public SynMatrix apply(SynMatrix synMatrix) {
        return applyToRows(synMatrix, this);
    }

    @Override
    public double[] apply(double[] input) {
        double[] exp = Arrays.stream(input).map(FastMath::exp).toArray();
        double sum = Arrays.stream(exp).sum();
        return Arrays.stream(exp).map(value -> value/sum).toArray();
    }

}
