/*
 * James Brundege
 * Date: 2017-04-16
 * MIT license: https://opensource.org/licenses/MIT
 */
package jmb.jcortex.strategies.weightinitializers;

import jmb.jcortex.data.SynMatrix;

/**
 * Simplified WeightInitializer for testing that sets all weights to 1.
 */
public class OnesWeightInitializer implements WeightInitializer {

    @Override
    public SynMatrix initialize(SynMatrix matrix) {
        return SynMatrix.ones(matrix.numRows(), matrix.numCols());
    }

}
