/*
 * James Brundege
 * Date: 2017-05-14
 * MIT license: https://opensource.org/licenses/MIT
 */
package jmb.jcortex.mapfunctions;

import java.util.function.DoubleUnaryOperator;

/**
 *
 */
public class RectifiedLinearActivationFunction implements DifferentiableFunction {

    @Override
    public DoubleUnaryOperator getFunction() {
        return x -> Math.max(0.0, x);
    }

    @Override
    public DoubleUnaryOperator getDerivative() {
        return x -> x > 0 ? 1 : 0;
    }
}
