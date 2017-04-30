/*
 * James Brundege
 * Date: 2017-04-09
 * MIT license: https://opensource.org/licenses/MIT
 */
package jmb.jcortex.data;

import java.util.*;
import java.util.stream.IntStream;

import static java.lang.String.format;
import static java.util.stream.Collectors.toList;

public class DataSet implements Copyable<DataSet> {
    private SynMatrix features;
    private SynMatrix labels;


    public DataSet(SynMatrix features, SynMatrix labels) {
        this.features = features;
        this.labels = labels;
    }

    public SynMatrix getFeatures() {
        return features;
    }

    public SynMatrix getLabels() {
        return labels;
    }

    @Override
    public DataSet copy() {
        return new DataSet(features.copy(), labels.copy());
    }

    public DataSet shuffleRows() {
        int numRows = features.numRows();
        List<Integer> rowNums = IntStream.range(0, features.numRows()).boxed().collect(toList());
        Collections.shuffle(rowNums);

        double[][] shuffledFeatures = new double[numRows][];
        double[][] shuffledLabels = new double[numRows][];

        for (int i = 0; i < rowNums.size(); i++) {
            int rowNum = rowNums.get(i);
            shuffledFeatures[i] = features.getRow(rowNum);

            if (labels != null) {
                shuffledLabels[i] = labels.getRow(rowNum);
            }
        }

        return new DataSet(new SynMatrix(shuffledFeatures), labels != null ? new SynMatrix(shuffledLabels) : null);
    }

    public int numRows() {
        return features.numRows();
    }

    public DataSet sliceRows(int startRowInclusive, int endRowExclusive) {
        if (startRowInclusive < 0) {
            throw new IllegalArgumentException(format("Start row is < 0. Start row: %s", startRowInclusive));
        }
        if (endRowExclusive > numRows()) {
            throw new IllegalArgumentException(format(
                    "End row is after the end of the DataSet. DataSet length: %s, end row: %s", numRows(), endRowExclusive));
        }

        SynMatrix newFeatures = features.sliceRows(startRowInclusive, endRowExclusive);
        SynMatrix newLabels = null;
        if (labels != null) {
            newLabels = labels.sliceRows(startRowInclusive, endRowExclusive);
        }
        return new DataSet(newFeatures, newLabels);
    }

}
