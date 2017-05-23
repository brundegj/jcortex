/*
 * James Brundege
 * Date: 2017-04-23
 * MIT license: https://opensource.org/licenses/MIT
 */
package jmb.jcortex.mapfunctions;

import jmb.jcortex.data.SynMatrix;

import java.util.function.DoubleUnaryOperator;

/**
 *
 */
public class SimpleMatrixFunction implements MatrixFunction {

    private final DoubleUnaryOperator function;

    public SimpleMatrixFunction(DoubleUnaryOperator function) {
        this.function = function;
    }

    @Override
    public SynMatrix apply(SynMatrix synMatrix) {
        return synMatrix.apply(function);
    }
}