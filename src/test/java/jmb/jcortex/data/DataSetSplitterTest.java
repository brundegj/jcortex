/*
 * James Brundege
 * Date: 2017-04-09
 * MIT license: https://opensource.org/licenses/MIT
 */
package jmb.jcortex.data;

import org.junit.Before;
import org.junit.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

public class DataSetSplitterTest {

    private DataSetSplitter dataSetSplitter;
    private double[][] featureData = new double[][] {
            {1, 2, 3},
            {4, 5, 6},
            {7, 8, 9},
            {0, 1, 2},
            {3, 4, 5}
    };
    private double[][] labelData = new double[][]{{1}, {2}, {3}, {4}, {5}};
    private DataSet dataSet;

    @Before
    public void setup() {
        dataSetSplitter = new DataSetSplitter();
        dataSet = new DataSet(new SynMatrix(featureData), new SynMatrix(labelData));
    }

    @Test
    public void split_CreatesSubsetsByPercentage() {
        DataSet[] subsets = dataSetSplitter.split(dataSet, 0.2, 0.4, 0.4);

        assertThat(subsets).hasSize(3);

        assertThat(subsets[0].numRows()).isEqualTo(1);
        assertThat(subsets[0].getFeatures().getRow(0)).isEqualTo(featureData[0]);
        assertThat(subsets[0].getLabels().getRow(0)).isEqualTo(labelData[0]);

        assertThat(subsets[1].numRows()).isEqualTo(2);
        assertThat(subsets[1].getFeatures().getRow(0)).isEqualTo(featureData[1]);
        assertThat(subsets[1].getFeatures().getRow(1)).isEqualTo(featureData[2]);
        assertThat(subsets[1].getLabels().getRow(0)).isEqualTo(labelData[1]);
        assertThat(subsets[1].getLabels().getRow(1)).isEqualTo(labelData[2]);

        assertThat(subsets[2].numRows()).isEqualTo(2);
        assertThat(subsets[2].getFeatures().getRow(0)).isEqualTo(featureData[3]);
        assertThat(subsets[2].getFeatures().getRow(1)).isEqualTo(featureData[4]);
        assertThat(subsets[2].getLabels().getRow(0)).isEqualTo(labelData[3]);
        assertThat(subsets[2].getLabels().getRow(1)).isEqualTo(labelData[4]);
    }

    @Test
    public void split_DoesNotRequire100Percent() {
        DataSet[] subsets = dataSetSplitter.split(dataSet, 0.4);

        assertThat(subsets).hasSize(1);
        assertThat(subsets[0].numRows()).isEqualTo(2);
        assertThat(subsets[0].getFeatures().getRow(0)).isEqualTo(featureData[0]);
        assertThat(subsets[0].getFeatures().getRow(1)).isEqualTo(featureData[1]);
        assertThat(subsets[0].getLabels().getRow(0)).isEqualTo(labelData[0]);
        assertThat(subsets[0].getLabels().getRow(1)).isEqualTo(labelData[1]);
    }

    @Test
    public void split_ShortensFinalDataSet_IfTotalPercentGreaterThan1() {
        DataSet[] subsets = dataSetSplitter.split(dataSet, 0.6, 0.6);

        assertThat(subsets).hasSize(2);

        assertThat(subsets[0].numRows()).isEqualTo(3);
        assertThat(subsets[0].getFeatures().getRow(0)).isEqualTo(featureData[0]);
        assertThat(subsets[0].getFeatures().getRow(1)).isEqualTo(featureData[1]);
        assertThat(subsets[0].getFeatures().getRow(2)).isEqualTo(featureData[2]);
        assertThat(subsets[0].getLabels().getRow(0)).isEqualTo(labelData[0]);
        assertThat(subsets[0].getLabels().getRow(1)).isEqualTo(labelData[1]);
        assertThat(subsets[0].getLabels().getRow(2)).isEqualTo(labelData[2]);

        assertThat(subsets[1].numRows()).isEqualTo(2);
        assertThat(subsets[1].getFeatures().getRow(0)).isEqualTo(featureData[3]);
        assertThat(subsets[1].getFeatures().getRow(1)).isEqualTo(featureData[4]);
        assertThat(subsets[1].getLabels().getRow(0)).isEqualTo(labelData[3]);
        assertThat(subsets[1].getLabels().getRow(1)).isEqualTo(labelData[4]);
    }

    @Test
    public void split_ShortensFinalDataSet_IfIndividualPercentGreaterThan1() {
        DataSet[] subsets = dataSetSplitter.split(dataSet, 1.2);

        assertThat(subsets).hasSize(1);

        assertThat(subsets[0].numRows()).isEqualTo(5);
        assertThat(subsets[0].getFeatures().getRow(0)).isEqualTo(featureData[0]);
        assertThat(subsets[0].getFeatures().getRow(1)).isEqualTo(featureData[1]);
        assertThat(subsets[0].getFeatures().getRow(2)).isEqualTo(featureData[2]);
        assertThat(subsets[0].getFeatures().getRow(3)).isEqualTo(featureData[3]);
        assertThat(subsets[0].getFeatures().getRow(4)).isEqualTo(featureData[4]);
        assertThat(subsets[0].getLabels().getRow(0)).isEqualTo(labelData[0]);
        assertThat(subsets[0].getLabels().getRow(1)).isEqualTo(labelData[1]);
        assertThat(subsets[0].getLabels().getRow(2)).isEqualTo(labelData[2]);
        assertThat(subsets[0].getLabels().getRow(3)).isEqualTo(labelData[3]);
        assertThat(subsets[0].getLabels().getRow(4)).isEqualTo(labelData[4]);
    }

    @Test
    public void split_ThrowsException_IfWeCantReturnTheNumberOfSubsetsRequested() {
        assertThatThrownBy(() -> dataSetSplitter.split(dataSet, 0.8, 0.8, 0.8))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("setSizes sum to > 1.0");
    }

}