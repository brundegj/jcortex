/*
 * James Brundege
 * Date: 2017-04-16
 * MIT license: https://opensource.org/licenses/MIT
 */
package jmb.jcortex.mapfunctions;

import java.util.function.DoubleUnaryOperator;

/**
 * A simplified calculation useful for testing. Simply doubles the inputs.
 */
public class DoublingActivationFunction implements DifferentiableFunction {

    @Override
    public DoubleUnaryOperator getFunction() {
        return x -> 2 * x;
    }

    @Override
    public DoubleUnaryOperator getDerivative() {
        return x -> 2;
    }

}
