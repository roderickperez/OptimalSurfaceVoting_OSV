package osv;

import java.io.File;
import java.util.Arrays;
import edu.mines.jtk.dsp.LocalOrientFilter;
import edu.mines.jtk.io.ArrayInputStream;
import edu.mines.jtk.io.ArrayOutputStream;

/**
 * Minimal command-line runner that applies 2D OSV to a single seismic slice.
 *
 * Usage:
 *   java -cp <classpath> osv.OSV2DRunner <inputDat> <outputDir> <n1> <n2>
 *
 * Input/output .dat files are big-endian float32.
 */
public class OSV2DRunner {
  public static void main(String[] args) throws Exception {
    if (args.length < 4) {
      System.err.println("Usage: osv.OSV2DRunner <inputDat> <outputDir> <n1> <n2>");
      System.exit(1);
    }

    String inputDat = args[0];
    String outputDir = args[1];
    int n1 = Integer.parseInt(args[2]);
    int n2 = Integer.parseInt(args[3]);

    File outDir = new File(outputDir);
    if (!outDir.exists() && !outDir.mkdirs()) {
      throw new RuntimeException("Failed to create output directory: " + outputDir);
    }

    float[][] gx = new float[n2][n1];
    ArrayInputStream ais = new ArrayInputStream(inputDat);
    ais.readFloats(gx);
    ais.close();

    sanitizeInPlace(gx, 0.0f);

    // 1) Compute linearity attribute.
    float[][] el = new float[n2][n1];
    float[][] u1 = new float[n2][n1];
    float[][] u2 = new float[n2][n1];
    LocalOrientFilter lof = new LocalOrientFilter(4.0, 1.0);
    lof.applyForNormalLinear(gx, u1, u2, el);
    sanitizeInPlace(el, 1.0f);

    // Real seismic sections are more stable with a discontinuity-like score.
    float[][] fa = invert01(normalize01(el));
    gammaInPlace(fa, 2.0f);

    // 2) Approximate dip scan.
    FaultOrientScanner2 fos = new FaultOrientScanner2(8.0);
    float[][][] scan = bestScan(fos, fa);
    float[][] ft = combineScores(fa, normalize01(scan[0]));
    float[][] pt = scan[1];
    sanitizeInPlace(ft, 0.0f);
    sanitizeInPlace(pt, 80.0f);

    OptimalPathVoter opv = new OptimalPathVoter(15, 30);
    opv.setStrainMax(0.25);
    opv.setPathSmoothing(2.0);

    float[][] voteScore = ft;
    float fm = chooseThreshold(opv, voteScore, pt);
    if (countSeeds(opv, 4, fm, voteScore, pt) == 0) {
      voteScore = fa;
      fm = chooseThreshold(opv, voteScore, pt);
    }
    if (countSeeds(opv, 4, fm, voteScore, pt) == 0) {
      fill(pt, 80.0f);
      voteScore = fa;
      fm = chooseThreshold(opv, voteScore, pt);
    }

    int seedCount = countSeeds(opv, 4, fm, voteScore, pt);
    if (seedCount == 0) {
      throw new IllegalStateException("Unable to pick OSV seeds from the selected section.");
    }

    // 3) Optimal path voting + thinning.
    float[][][] fvw = opv.applyVoting(4, fm, voteScore, pt);
    float[][] fv = fvw[0];
    float[][] w1 = fvw[1];
    float[][] w2 = fvw[2];
    float[][] fvt = opv.thin(fv, w1, w2);

    // Attribute-guided post-filter: keeps OSV responses near fault-attribute ridges.
    float[][] fan = normalize01(fa);
    float[][] fvg = guidedScore(fv, fan, 1.5f);
    float[][] fvtg = opv.thin(fvg, w1, w2);

    sanitizeInPlace(fv, 0.0f);
    sanitizeInPlace(fvt, 0.0f);
    sanitizeInPlace(fvg, 0.0f);
    sanitizeInPlace(fvtg, 0.0f);
    normalizeInPlace(fv);
    normalizeInPlace(fvt);
    normalizeInPlace(fvg);
    normalizeInPlace(fvtg);

    write2D(new File(outDir, "el.dat").getAbsolutePath(), el);
    write2D(new File(outDir, "fa.dat").getAbsolutePath(), fa);
    write2D(new File(outDir, "ft.dat").getAbsolutePath(), ft);
    write2D(new File(outDir, "pt.dat").getAbsolutePath(), pt);
    write2D(new File(outDir, "fv.dat").getAbsolutePath(), fv);
    write2D(new File(outDir, "fvt.dat").getAbsolutePath(), fvt);
    write2D(new File(outDir, "fvg.dat").getAbsolutePath(), fvg);
    write2D(new File(outDir, "fvtg.dat").getAbsolutePath(), fvtg);

    System.out.println("OSV 2D run complete.");
    System.out.println("Seed count: " + seedCount + ", threshold=" + fm);
    System.out.println("Wrote outputs in: " + outDir.getAbsolutePath());
  }

  private static void write2D(String fileName, float[][] data) throws Exception {
    ArrayOutputStream aos = new ArrayOutputStream(fileName);
    aos.writeFloats(data);
    aos.close();
  }

  private static float[][][] bestScan(FaultOrientScanner2 fos, float[][] score) {
    double[][] ranges = new double[][] {
      {45.0, 89.0},
      {60.0, 89.0},
      {30.0, 89.0},
      {75.0, 85.0}
    };

    float[][][] best = null;
    int bestCount = -1;
    float bestMax = -1.0f;
    for (double[] range : ranges) {
      float[][][] scan = fos.scanDip(range[0], range[1], score);
      sanitizeInPlace(scan[0], 0.0f);
      sanitizeInPlace(scan[1], 80.0f);
      int count = countPositive(scan[0], 1.0e-6f);
      float maxValue = maxFinite(scan[0]);
      if (count > bestCount || (count == bestCount && maxValue > bestMax)) {
        best = scan;
        bestCount = count;
        bestMax = maxValue;
      }
    }
    return best;
  }

  private static int countSeeds(OptimalPathVoter opv, int d, float fm, float[][] score, float[][] pt) {
    return opv.pickSeeds(d, fm, score, pt).length;
  }

  private static float chooseThreshold(OptimalPathVoter opv, float[][] score, float[][] pt) {
    float[] candidates = new float[] {0.8f, 0.6f, 0.4f, 0.25f, 0.1f, 0.03f, 0.0f};
    for (float candidate : candidates) {
      if (countSeeds(opv, 4, candidate, score, pt) > 0) {
        return candidate;
      }
    }
    return 0.0f;
  }

  private static int countPositive(float[][] x, float threshold) {
    int count = 0;
    for (float[] row : x) {
      for (float value : row) {
        if (Float.isFinite(value) && value > threshold) {
          ++count;
        }
      }
    }
    return count;
  }

  private static float maxFinite(float[][] x) {
    float maxValue = -Float.MAX_VALUE;
    for (float[] row : x) {
      for (float value : row) {
        if (Float.isFinite(value) && value > maxValue) {
          maxValue = value;
        }
      }
    }
    return maxValue;
  }

  private static void sanitizeInPlace(float[][] x, float fillValue) {
    for (float[] row : x) {
      for (int i = 0; i < row.length; ++i) {
        if (!Float.isFinite(row[i])) {
          row[i] = fillValue;
        }
      }
    }
  }

  private static float[][] normalize01(float[][] x) {
    int n2 = x.length;
    int n1 = x[0].length;
    float xmin = Float.MAX_VALUE;
    float xmax = -Float.MAX_VALUE;
    for (float[] row : x) {
      for (float value : row) {
        if (!Float.isFinite(value)) {
          continue;
        }
        if (value < xmin) {
          xmin = value;
        }
        if (value > xmax) {
          xmax = value;
        }
      }
    }
    if (xmin == Float.MAX_VALUE || xmax == -Float.MAX_VALUE) {
      float[][] zeros = new float[n2][n1];
      return zeros;
    }
    float den = xmax - xmin;
    if (den < 1.0e-6f) {
      den = 1.0e-6f;
    }
    float[][] y = new float[n2][n1];
    for (int i2 = 0; i2 < n2; ++i2) {
      for (int i1 = 0; i1 < n1; ++i1) {
        float value = x[i2][i1];
        y[i2][i1] = Float.isFinite(value) ? (value - xmin) / den : 0.0f;
      }
    }
    return y;
  }

  private static float[][] invert01(float[][] x) {
    int n2 = x.length;
    int n1 = x[0].length;
    float[][] y = new float[n2][n1];
    for (int i2 = 0; i2 < n2; ++i2) {
      for (int i1 = 0; i1 < n1; ++i1) {
        y[i2][i1] = 1.0f - x[i2][i1];
      }
    }
    return y;
  }

  private static float[][] combineScores(float[][] a, float[][] b) {
    int n2 = a.length;
    int n1 = a[0].length;
    float[][] y = new float[n2][n1];
    for (int i2 = 0; i2 < n2; ++i2) {
      for (int i1 = 0; i1 < n1; ++i1) {
        y[i2][i1] = a[i2][i1] * b[i2][i1];
      }
    }
    normalizeInPlace(y);
    return y;
  }

  private static float[][] guidedScore(float[][] score, float[][] attr, float power) {
    int n2 = score.length;
    int n1 = score[0].length;
    float[][] y = new float[n2][n1];
    for (int i2 = 0; i2 < n2; ++i2) {
      for (int i1 = 0; i1 < n1; ++i1) {
        float a = attr[i2][i1];
        if (a < 0.0f) {
          a = 0.0f;
        }
        y[i2][i1] = score[i2][i1] * (float)Math.pow(a, power);
      }
    }
    normalizeInPlace(y);
    return y;
  }

  private static void gammaInPlace(float[][] x, float power) {
    for (float[] row : x) {
      for (int i = 0; i < row.length; ++i) {
        float value = row[i];
        if (value < 0.0f) {
          value = 0.0f;
        }
        row[i] = (float)Math.pow(value, power);
      }
    }
  }

  private static void normalizeInPlace(float[][] x) {
    float[][] normalized = normalize01(x);
    for (int i2 = 0; i2 < x.length; ++i2) {
      System.arraycopy(normalized[i2], 0, x[i2], 0, x[i2].length);
    }
  }

  private static void fill(float[][] x, float value) {
    for (float[] row : x) {
      Arrays.fill(row, value);
    }
  }
}
