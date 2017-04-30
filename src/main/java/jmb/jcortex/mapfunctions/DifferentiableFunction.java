/*
 * James Brundege
 * Date: 2017-04-23
 * MIT license: https://opensource.org/licenses/MIT
 */
package jmb.jcortex.mapfunctions;

import java.util.function.DoubleUnaryOperator;

/**
 *
 */
public interface DifferentiableFunction {

    /**
     * @return a differentiable function implementing the DoubleUnaryOperator interface
     */
    DoubleUnaryOperator getFunction();

    /**
     * The derivative function assuming that the input (x) is the output of the original getFunction() call.
     * Note that this is not the same as the pure derivative of getFunction(), but rather the derivative calculation
     * on the getFunction output, which is what we use during backpropagation by saving the original node values.
     * To get the actual derivative result on an arbitrary x, you would need to call
     * getDerivative().applyAsDouble(getFunction().applyAsDouble(x))
     */
    DoubleUnaryOperator getDerivative();
}
