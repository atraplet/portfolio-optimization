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

public class Main {

    /**
     * Solve a Markowitz portfolio optimization problem with <a href="https://github.com/atraplet/ecos4j">ecos4j</a>.
     */
    public static void main(String[] args) {
        // Define portfolio optimization problem
        val mu = new SimpleMatrix(new double[]{0.05, 0.06, 0.08, 0.06});
        val sigma = new SimpleMatrix(4, 4, true,
                0.0225, 0.003, 0.015, 0.0225,
                0.003, 0.04, 0.035, 0.024,
                0.015, 0.035, 0.0625, 0.06,
                0.0225, 0.024, 0.06, 0.09);
        val sigmaLimit = 0.2;

        // Problem dimension
        val n = mu.getNumRows();

        // Compute Cholesky decomposition of sigma
        val chol = DecompositionFactory_DDRM.chol(n, true);
        if (!chol.decompose(sigma.getMatrix()))
            throw new RuntimeException("Cholesky decomposition failed");
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
            model.setup(n + 1, new long[]{n + 1}, 0, gSpMat.nz_values, toLongArray(gSpMat.col_idx),
                    toLongArray(gSpMat.nz_rows), cMat.getDDRM().data, hMat.getDDRM().data, aSpMat.nz_values,
                    toLongArray(aSpMat.col_idx), toLongArray(aSpMat.nz_rows), bMat.getDDRM().data);

            // Create and set parameters
            val parameters = Parameters.builder()
                    .verbose(true)
                    .build();
            model.setParameters(parameters);

            // Optimize model
            val status = model.optimize();
            if (status != OPTIMAL) {
                throw new RuntimeException("Optimization failed");
            }

            // Get solution
            val xMat = new SimpleMatrix(model.x());
            System.out.println("xMat");
            xMat.print();
        }
    }

    private static long[] toLongArray(int[] arr) {
        return Arrays.stream(arr).asLongStream().toArray();
    }

}
