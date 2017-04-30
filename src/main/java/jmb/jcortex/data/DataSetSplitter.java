/*
 * James Brundege
 * Date: 2017-04-09
 * MIT license: https://opensource.org/licenses/MIT
 */
package jmb.jcortex.data;

import org.apache.commons.math3.util.FastMath;

public class DataSetSplitter {

    public DataSet[] split(DataSet src, double... setSizes) {
        int numRows = src.numRows();
        DataSet[] subsets = new DataSet[setSizes.length];
        int previousEnd = 0;
        for (int i = 0; i < setSizes.length; i++) {
            // if we've consumed the entire DataSet and still have another subset, there is something very wrong
            if (previousEnd >= numRows) {
                throw new IllegalArgumentException("Error, setSizes sum to > 1.0");
            }
            int numRowsInSubset = (int)FastMath.round(setSizes[i] * numRows);
            int newEnd = previousEnd + numRowsInSubset;
            // if we're over on this subset, just truncate it. This can happen due to rounding issues, so don't fail.
            if (newEnd > numRows) {
                newEnd = numRows;
            }
            subsets[i] = src.sliceRows(previousEnd, newEnd);
            previousEnd = newEnd;
        }
        return subsets;
    }

}
