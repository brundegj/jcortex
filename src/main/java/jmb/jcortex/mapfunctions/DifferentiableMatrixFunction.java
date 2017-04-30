/*
 * James Brundege
 * Date: 2017-04-23
 * MIT license: https://opensource.org/licenses/MIT
 */
package jmb.jcortex.mapfunctions;

/**
 *
 */
public interface DifferentiableMatrixFunction {

    MatrixFunction getFunction();
    MatrixFunction getDerivative();

}
