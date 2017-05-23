/*
 * James Brundege
 * Date: 2017-04-23
 * MIT license: https://opensource.org/licenses/MIT
 */
package jmb.jcortex.mapfunctions;

/**
 * Common activation and output functions. These are stateless and threadsafe, and can thus be used as constants.
 */
public class MatrixFunctions {

    public static final DifferentiableMatrixFunction SIGMOID_MATRIX_FUNCTION
            = new SimpleDifferentiableMatrixFunction(new SigmoidActivationFunction());

    public static final DifferentiableMatrixFunction LINEAR_MATRIX_FUNCTION
            = new SimpleDifferentiableMatrixFunction(new IdentityFunction());

    public static final DifferentiableMatrixFunction RECIFIED_LINEAR_MATRIX_FUNCTION
            = new SimpleDifferentiableMatrixFunction(new RectifiedLinearActivationFunction());

    public static final DifferentiableMatrixFunction SOFTMAX_MATRIX_FUNCTION
            = new SimpleDifferentiableMatrixFunction(new SoftMaxActivationFunction(),
            new SimpleMatrixFunction(x -> 1));

}
