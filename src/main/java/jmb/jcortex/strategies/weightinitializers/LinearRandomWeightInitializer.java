/*
 * James Brundege
 * Date: 2017-04-16
 * MIT license: https://opensource.org/licenses/MIT
 */
package jmb.jcortex.strategies.weightinitializers;

import jmb.jcortex.data.SynMatrix;

import java.util.Random;
import java.util.stream.IntStream;

public class LinearRandomWeightInitializer implements WeightInitializer{
    private Random random = new Random();
    private double min;
    private double max;

    public LinearRandomWeightInitializer(double min, double max) {
        this.min = min;
        this.max = max;
    }

    @Override
    public SynMatrix initialize(SynMatrix matrix) {
        IntStream.range(0, matrix.numRows()).parallel()
                .forEach(rowNum -> {
                    double[] row = matrix.getRow(rowNum);
                    for (int colNum = 0; colNum < row.length; colNum++) {
                        matrix.set(rowNum, colNum, (max-min) * random.nextDouble() + min);
                    }
                });
        return matrix;
    }

}
