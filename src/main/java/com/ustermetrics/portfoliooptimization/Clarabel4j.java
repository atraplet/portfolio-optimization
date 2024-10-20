package com.ustermetrics.portfoliooptimization;

import com.ustermetrics.clarabel4j.*;
import lombok.val;
import org.ejml.data.DMatrixSparseCSC;
import org.ejml.dense.row.factory.DecompositionFactory_DDRM;
import org.ejml.ops.DConvertMatrixStruct;
import org.ejml.simple.SimpleMatrix;

import java.util.Arrays;
import java.util.List;

import static com.ustermetrics.clarabel4j.Status.SOLVED;

public class Clarabel4j {

    /**
     * Solve a Markowitz portfolio optimization problem with
     * <a href="https://github.com/atraplet/clarabel4j">clarabel4j</a>.
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
        val n = mu.getNumRows() + 1;

        // Compute Cholesky decomposition of sigma
        val chol = DecompositionFactory_DDRM.chol(n - 1, true);
        if (!chol.decompose(sigma.copy().getMatrix()))
            throw new IllegalArgumentException("Cholesky decomposition failed");
        val upTriMat = SimpleMatrix.wrap(chol.getT(null)).transpose();

        // Define second-order cone program
        val qMat = mu.negative()
                .concatRows(new SimpleMatrix(1, 1));
        System.out.println("\nqMat");
        qMat.print();

        val aMatZeroCone = SimpleMatrix.ones(1, n - 1)
                .concatColumns(new SimpleMatrix(1, 1));
        val aMatNonNeg = SimpleMatrix.identity(n - 1)
                .negative()
                .concatColumns(new SimpleMatrix(n - 1, 1))
                .concatRows(new SimpleMatrix(1, n - 1).concatColumns(SimpleMatrix.ones(1, 1)));
        val aMatSoc = new SimpleMatrix(1, n - 1)
                .concatColumns(SimpleMatrix.filled(1, 1, -1.0))
                .concatRows(upTriMat.negative().concatColumns(new SimpleMatrix(n - 1, 1)));
        val aMat = aMatZeroCone.concatRows(aMatNonNeg).concatRows(aMatSoc);
        System.out.println("\naMat");
        aMat.print();

        val bMat = new SimpleMatrix(2 * (n - 1) + 3, 1);
        bMat.set(0, 0, 1.);
        bMat.set(n, 0, sigmaLimit);
        System.out.println("\nbMat");
        bMat.print();

        // clarabel4j needs sparse aMat
        val tol = 1e-8;
        val aSpMat = DConvertMatrixStruct.convert(aMat.getDDRM(), (DMatrixSparseCSC) null, tol);
        System.out.println("\naSpMat");
        aSpMat.print();

        // Create model
        try (val model = new Model()) {
            // Create and set parameters
            val parameters = Parameters.builder()
                    .verbose(true)
                    .build();
            model.setParameters(parameters);

            // Set up model
            model.setup(qMat.getDDRM().data, convert(aSpMat), bMat.getDDRM().data,
                    List.of(new ZeroCone(1), new NonnegativeCone(n), new SecondOrderCone(n)));

            // Optimize model
            val status = model.optimize();
            if (status != SOLVED) {
                throw new IllegalStateException("Optimization failed");
            }

            // Get solution
            val xMat = new SimpleMatrix(model.x())
                    .extractMatrix(0, n - 1, 0, 1);
            System.out.println("xMat");
            xMat.print();
        }
    }

    private static Matrix convert(DMatrixSparseCSC matrix) {
        return new Matrix(matrix.getNumRows(), matrix.getNumCols(), getColIdx(matrix), getNzRows(matrix),
                getNzValues(matrix));
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
