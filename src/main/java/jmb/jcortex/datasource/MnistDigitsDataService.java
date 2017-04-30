package jmb.jcortex.datasource;

import jmb.jcortex.data.DataSet;
import jmb.jcortex.data.SynMatrix;
import org.apache.commons.io.IOUtils;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;

import static java.lang.String.format;


/**
 * James Brundege
 * Date: 2017-02-01
 */
public class MnistDigitsDataService {

    private static final String dataFile = "/MNIST_Digits/train-images.idx3-ubyte";
    private static final String labelsFile = "/MNIST_Digits/train-labels.idx1-ubyte";

    /**
     * Load labeled MNIST images from IDX format files. MNIST images courtesy of Yann Lecun's site:
     * http://yann.lecun.com/exdb/mnist/
     */
    public DataSet loadDataFile() {
        try (InputStream imagesInputStream = this.getClass().getResourceAsStream(dataFile);
             InputStream labelsInputStream = this.getClass().getResourceAsStream(labelsFile)
        ) {
            byte[] images = IOUtils.toByteArray(imagesInputStream);
            ByteBuffer imagesBuffer = ByteBuffer.wrap(images);

            imagesBuffer.getInt();    // throw away 1st 4 bytes: magic number
            int numImages = imagesBuffer.getInt();
            int numRows = imagesBuffer.getInt();
            int numCols = imagesBuffer.getInt();
            int numPixels = numRows * numCols;

            double[][] data = new double[numImages][numPixels];
            for (int i = 0; i < numImages; i++) {
                for (int j = 0; j < numPixels; j++) {
                    double byteVal = Byte.toUnsignedInt(imagesBuffer.get());
                    data[i][j] = byteVal/255;    // normalize to 0-1 range
                }
            }

            SynMatrix features = new SynMatrix(data);

            byte[] labelBytes = IOUtils.toByteArray(labelsInputStream);
            ByteBuffer labelsBuffer = ByteBuffer.wrap(labelBytes);
            labelsBuffer.getInt();    // throw away 1st 4 bytes: magic number
            int numLabels = labelsBuffer.getInt();

            if (numImages != numLabels) {
                throw new RuntimeException(format(
                        "Unequal number of images and labels. Images: %s, Labels: %s", numImages, numLabels));
            }

            double[] labelArray = new double[numLabels];
            for (int i = 0; i < labelArray.length; i++) {
                double label = (double) Byte.toUnsignedInt(labelsBuffer.get());
                if (label < 0.0 || label > 9.0) {
                    throw new RuntimeException(format("Bad label value: %s", label));
                }
                labelArray[i] = label;
            }
            SynMatrix labels = convertIntegerDataToClasses(labelArray);

            return new DataSet(features, labels);
        } catch (IOException ex) {
            throw new RuntimeException(ex);
        }
    }

    public SynMatrix convertIntegerDataToClasses(double[] labelArray) {
        double[][] classifiedData = new double[labelArray.length][10];

        for (int i = 0; i < labelArray.length; i++) {
            long value = Math.round(labelArray[i]);
            if (value == 10) value = 0;
            int index = (int) (value);
            for (int j = 0; j < 10; j++) {
                if (j == index) {
                    classifiedData[i][j] = 1;
                } else {
                    classifiedData[i][j] = 0;
                }
            }
        }

        return new SynMatrix(classifiedData);
    }

}
