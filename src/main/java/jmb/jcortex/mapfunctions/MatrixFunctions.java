/*
 * James Brundege
 * Date: 2017-04-23
 * MIT license: https://opensource.org/licenses/MIT
 */
package jmb.jcortex.mapfunctions;

/**
 *
 */
public class MatrixFunctions {

    public static final DifferentiableMatrixFunction SIGMOID_MATRIX_FUNCTION
            = new SimpleDifferentiableMatrixFunction(new SigmoidActivationFunction());

    public static final DifferentiableMatrixFunction LINEAR_MATRIX_FUNCTION
            = new SimpleDifferentiableMatrixFunction(new IdentityFunction());

    public static final DifferentiableMatrixFunction SOFTMAX_MATRIX_FUNCTION
            = new SimpleDifferentiableMatrixFunction(new SoftMaxActivationFunction(),
            new SimpleMatrixFunction(new IdentityFunction().getDerivative()));
            // SoftMax is used for output and has no derivative, so use the derivative of the identityfunction
            // which is f(x) -> 1. This is effectively a no-op for calculating deltas at the output.

}
