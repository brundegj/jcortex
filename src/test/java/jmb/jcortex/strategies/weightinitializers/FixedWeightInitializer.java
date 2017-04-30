/*
 * James Brundege
 * Date: 2017-04-29
 * MIT license: https://opensource.org/licenses/MIT
 */
package jmb.jcortex.strategies.weightinitializers;

import jmb.jcortex.data.SynMatrix;

/**
 * For testing only. Uses a deterministic rather than random set of weights.
 */
public class FixedWeightInitializer implements WeightInitializer {

    @Override
    public SynMatrix initialize(SynMatrix matrix) {
        for (int j = 0; j < matrix.numElements(); j++) {
            matrix.set(j, j%2==0 ? -0.5f : 0.5f);
        }
        return matrix;
    }
}
