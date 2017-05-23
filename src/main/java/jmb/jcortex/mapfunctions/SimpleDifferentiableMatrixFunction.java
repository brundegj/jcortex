/*
 * James Brundege
 * Date: 2017-04-23
 * MIT license: https://opensource.org/licenses/MIT
 */
package jmb.jcortex.mapfunctions;

/**
 *
 */
public class SimpleDifferentiableMatrixFunction implements DifferentiableMatrixFunction {

    private final MatrixFunction function;
    private final MatrixFunction derivative;

    public SimpleDifferentiableMatrixFunction(DifferentiableFunction differentiableFunction) {
        function = new SimpleMatrixFunction(differentiableFunction.getFunction());
        derivative = new SimpleMatrixFunction(differentiableFunction.getDerivative());
    }

    public SimpleDifferentiableMatrixFunction(MatrixFunction function, MatrixFunction derivative) {
        this.function = function;
        this.derivative = derivative;
    }

    @Override
    public MatrixFunction getFunction() {
        return function;
    }

    @Override
    public MatrixFunction getDerivative() {
        return derivative;
    }
}