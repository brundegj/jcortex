/*
 * James Brundege
 * Date: 2017-04-16
 * MIT license: https://opensource.org/licenses/MIT
 */
package jmb.jcortex.mapfunctions;

import jmb.jcortex.data.SynMatrix;

import java.util.stream.IntStream;

public interface MatrixFunction {

    SynMatrix apply(SynMatrix synMatrix);

    default SynMatrix applyToRows(SynMatrix synMatrix, RowFunction rowFunction) {
        IntStream.range(0, synMatrix.numRows()).parallel().forEach(rowIndex -> {
            double[] row = synMatrix.getRow(rowIndex);
            synMatrix.setRow(rowIndex, rowFunction.apply(row));
        });
        return synMatrix;
    }

}
