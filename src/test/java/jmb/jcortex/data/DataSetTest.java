/*
 * James Brundege
 * Date: 2017-04-09
 * MIT license: https://opensource.org/licenses/MIT
 */
package jmb.jcortex.data;

import org.junit.Before;
import org.junit.Test;

import java.util.Arrays;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

public class DataSetTest {

    private double[][] featureData = new double[][] {
            {1, 2, 3},
            {4, 5, 6},
            {7, 8, 9},
            {0, 1, 2},
            {3, 4, 5},
            {6, 7, 8},
            {9, 0, 1},
            {2, 3, 4},
            {5, 6, 7},
            {8, 9, 0},
    };
    private double[][] labelData = new double[][]{{1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}, {0}};
    private DataSet dataSet;

    @Before
    public void setup() {
        dataSet = new DataSet(new SynMatrix(featureData), new SynMatrix(labelData));
    }

    @Test
    public void copy_ReturnsADeepCopyWithNoSharedState() {
        DataSet copy = dataSet.copy();
        assertThat(copy).isNotSameAs(dataSet);
        assertThat(copy.getFeatures()).isNotSameAs(dataSet.getFeatures());
        assertThat(copy.getLabels()).isNotSameAs(dataSet.getLabels());
    }

    @Test
    public void sliceRows_ReturnsASubset() {
        DataSet subset = dataSet.sliceRows(1, 4);
        assertThat(subset.numRows()).isEqualTo(3);
        assertThat(subset.getFeatures().getRow(0)).isEqualTo(featureData[1]);
        assertThat(subset.getFeatures().getRow(1)).isEqualTo(featureData[2]);
        assertThat(subset.getFeatures().getRow(2)).isEqualTo(featureData[3]);
        assertThat(subset.getLabels().getRow(0)).isEqualTo(labelData[1]);
        assertThat(subset.getLabels().getRow(1)).isEqualTo(labelData[2]);
        assertThat(subset.getLabels().getRow(2)).isEqualTo(labelData[3]);
    }

    @Test
    public void sliceRows_ThrowsException_IfStartRowInvalid() {
        assertThatThrownBy(() -> dataSet.sliceRows(-1, 3))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("Start row is < 0");
    }

    @Test
    public void sliceRows_ThrowsException_IfEndRowInvalid() {
        assertThatThrownBy(() -> dataSet.sliceRows(0, 11))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("End row is after the end of the DataSet");
    }

    @Test
    public void shuffleRows_RandomizesRowOrder() {
        double[][] originalFeatures = dataSet.getFeatures().getData();
        double[][] originalLabels = dataSet.getLabels().getData();

        DataSet shuffled = dataSet.shuffleRows();
        double[][] shuffledFeatures = shuffled.getFeatures().getData();
        double[][] shuffledLabels = shuffled.getLabels().getData();

        assertThat(shuffled.numRows()).isEqualTo(dataSet.numRows());
        assertThat(shuffledFeatures).isNotEqualTo(originalFeatures);
        assertThat(Arrays.asList(shuffledFeatures)).containsExactlyInAnyOrder(originalFeatures);
        assertThat(shuffledLabels).isNotEqualTo(originalLabels);
        assertThat(Arrays.asList(shuffledLabels)).containsExactlyInAnyOrder(originalLabels);
    }

}