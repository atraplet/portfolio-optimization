package com.ustermetrics.portfoliooptimization;

import com.ustermetrics.ecos4j.Model;
import com.ustermetrics.ecos4j.Parameters;
import lombok.val;
import org.ejml.data.DMatrixSparseCSC;
import org.ejml.dense.row.factory.DecompositionFactory_DDRM;
import org.ejml.ops.DConvertMatrixStruct;
import org.ejml.simple.SimpleMatrix;

import java.util.Arrays;

import static com.ustermetrics.ecos4j.Status.OPTIMAL;

public class Ecos4j {

    /**
     * Solve a Markowitz portfolio optimization problem with <a href="https://github.com/atraplet/ecos4j">ecos4j</a>.
     */
    public static void main(String[] args) {
        // Define portfolio optimization problem
        val mu = new SimpleMatrix(new double[]{0.05, 0.09, 0.07, 0.06});
        val sigma = new SimpleMatrix(
                4, 4, true,
                0.0016, 0.0006, 0.0008, -0.0004,
                0.0006, 0.0225, 0.0015, -0.0015,
                0.0008, 0.0015, 0.0025, -0.001,
                -0.0004, -0.0015, -0.001, 0.01
        );
        val sigmaLimit = 0.06;

        // Problem dimension
        val n = mu.getNumRows();

        // Compute Cholesky decomposition of sigma
        val chol = DecompositionFactory_DDRM.chol(n, true);
        if (!chol.decompose(sigma.copy().getMatrix()))
            throw new IllegalArgumentException("Cholesky decomposition failed");
        val upTriMat = SimpleMatrix.wrap(chol.getT(null)).transpose();

        // Define second-order cone program
        val cMat = mu.negative()
                .concatRows(new SimpleMatrix(1, 1));
        System.out.println("\ncMat");
        cMat.print();

        val aMat = SimpleMatrix.ones(1, n)
                .concatColumns(new SimpleMatrix(1, 1));
        System.out.println("\naMat");
        aMat.print();

        val bMat = SimpleMatrix.ones(1, 1);
        System.out.println("\nbMat");
        bMat.print();

        val gMatPosOrt = SimpleMatrix.identity(n)
                .negative()
                .concatColumns(new SimpleMatrix(n, 1))
                .concatRows(new SimpleMatrix(1, n).concatColumns(SimpleMatrix.ones(1, 1)));
        val gMatSoc = new SimpleMatrix(1, n)
                .concatColumns(SimpleMatrix.filled(1, 1, -1.0))
                .concatRows(upTriMat.negative().concatColumns(new SimpleMatrix(n, 1)));
        val gMat = gMatPosOrt.concatRows(gMatSoc);
        System.out.println("\ngMat");
        gMat.print();

        val hMat = new SimpleMatrix(2 * n + 2, 1);
        hMat.set(n, 0, sigmaLimit);
        System.out.println("\nhMat");
        hMat.print();

        // ecos4j needs sparse aMat and gMat
        val tol = 1e-8;
        val aSpMat = DConvertMatrixStruct.convert(aMat.getDDRM(), (DMatrixSparseCSC) null, tol);
        System.out.println("\naSpMat");
        aSpMat.print();

        val gSpMat = DConvertMatrixStruct.convert(gMat.getDDRM(), (DMatrixSparseCSC) null, tol);
        System.out.println("\ngSpMat");
        gSpMat.print();

        // Create model
        try (val model = new Model()) {

            // Set up model
            model.setup(n + 1, new long[]{n + 1}, 0, getNzValues(gSpMat), getColIdx(gSpMat),
                    getNzRows(gSpMat), cMat.getDDRM().data, hMat.getDDRM().data, getNzValues(aSpMat),
                    getColIdx(aSpMat), getNzRows(aSpMat), bMat.getDDRM().data);

            // Create and set parameters
            val parameters = Parameters.builder()
                    .verbose(true)
                    .build();
            model.setParameters(parameters);

            // Optimize model
            val status = model.optimize();
            if (status != OPTIMAL) {
                throw new IllegalStateException("Optimization failed");
            }

            // Get solution
            val xMat = new SimpleMatrix(model.x())
                    .extractMatrix(0, n, 0, 1);
            System.out.println("xMat");
            xMat.print();
        }
    }

    private static double[] getNzValues(DMatrixSparseCSC matrix) {
        if (!matrix.isIndicesSorted()) matrix.sortIndices(null);
        if (matrix.nz_values.length == matrix.nz_length) {
            return matrix.nz_values;
        } else {
            return Arrays.copyOfRange(matrix.nz_values, 0, matrix.nz_length);
        }
    }

    private static long[] getNzRows(DMatrixSparseCSC matrix) {
        if (!matrix.isIndicesSorted()) matrix.sortIndices(null);
        if (matrix.nz_rows.length == matrix.nz_length) {
            return toLongArray(matrix.nz_rows);
        } else {
            return toLongArray(Arrays.copyOfRange(matrix.nz_rows, 0, matrix.nz_length));
        }
    }

    private static long[] getColIdx(DMatrixSparseCSC matrix) {
        return toLongArray(matrix.col_idx);
    }

    private static long[] toLongArray(int[] arr) {
        return Arrays.stream(arr).asLongStream().toArray();
    }

}
