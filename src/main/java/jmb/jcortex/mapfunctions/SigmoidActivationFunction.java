/*
 * James Brundege
 * Date: 2017-04-09
 * MIT license: https://opensource.org/licenses/MIT
 */
package jmb.jcortex.mapfunctions;

import org.apache.commons.math3.util.FastMath;

import java.util.function.DoubleUnaryOperator;

public class SigmoidActivationFunction implements DifferentiableFunction {

    @Override
    public DoubleUnaryOperator getFunction() {
        return x -> 1 / (1 + FastMath.exp(-x));
    }

    /**
     * The derivative function assuming that the input (x) is the output of the original getFunction() call.
     * Note that this is not the same as the pure derivative of the getFunction(), but rather the derivate calculation
     * on the getFunction output, which is what we use during backpropagation by saving the original node values.
     * To get the actual sigmoid derivative result on an arbitrary x, you would need to call
     * getDerivative().applyAsDouble(getFunction().applyAsDouble(x))
     */
    @Override
    public DoubleUnaryOperator getDerivative() {
        return x -> x * (1.0 - x);
    }

}
