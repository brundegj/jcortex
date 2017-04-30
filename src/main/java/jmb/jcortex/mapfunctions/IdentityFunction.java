/*
 * James Brundege
 * Date: 2017-04-16
 * MIT license: https://opensource.org/licenses/MIT
 */
package jmb.jcortex.mapfunctions;

import java.util.function.DoubleUnaryOperator;

/**
 * No-op. Simply returns the input value. Leaves all values unchanged.
 */
public class IdentityFunction implements DifferentiableFunction {

    @Override
    public DoubleUnaryOperator getFunction() {
        return x -> x;
    }

    @Override
    public DoubleUnaryOperator getDerivative() {
        return x -> 1;
    }

}
