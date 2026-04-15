import java.io.File;
import java.util.Arrays;
import edu.mines.jtk.dsp.LocalOrientFilter;
import edu.mines.jtk.io.ArrayInputStream;
import edu.mines.jtk.io.ArrayOutputStream;
import osv.FaultOrientScanner2;
import osv.OptimalPathVoter;

/**
 * Minimal command-line runner that applies 2D OSV to a single seismic slice.
 *
 * Usage:
 *   java --class-path <classpath> tools/OSV2DRunner.java <inputDat> <outputDir> <n1> <n2> [faultScoreDat]
 *
 * Input/output .dat files are big-endian float32.
 */
public class OSV2DRunner {
  public static void main(String[] args) throws Exception {
    if (args.length < 4) {
      System.err.println("Usage: OSV2DRunner <inputDat> <outputDir> <n1> <n2> [faultScoreDat]");
      System.exit(1);
    }

    String inputDat = args[0];
    String outputDir = args[1];
    int n1 = Integer.parseInt(args[2]);
    int n2 = Integer.parseInt(args[3]);
    String faultScoreDat = args.length > 4 ? args[4] : null;
    boolean useExternalFaultScore = faultScoreDat != null;

    File outDir = new File(outputDir);
    if (!outDir.exists() && !outDir.mkdirs()) {
      throw new RuntimeException("Failed to create output directory: " + outputDir);
    }

    float[][] gx = new float[n2][n1];
    ArrayInputStream ais = new ArrayInputStream(inputDat);
    ais.readFloats(gx);
    ais.close();

    sanitizeInPlace(gx, 0.0f);

    float[][] el = new float[n2][n1];
    float[][] u1 = new float[n2][n1];
    float[][] u2 = new float[n2][n1];
    LocalOrientFilter lof = new LocalOrientFilter(4.0, 1.0);
    lof.applyForNormalLinear(gx, u1, u2, el);
    sanitizeInPlace(el, 1.0f);

    float[][] faLocal = invert01(normalize01(el));
    gammaInPlace(faLocal, 2.0f);

    float[][] fa;
    float[][] faSeed;
    if (useExternalFaultScore) {
      fa = read2D(faultScoreDat, n1, n2);
      sanitizeInPlace(fa, 0.0f);
      clipInPlace(fa, 0.0f, 1.0f);
      normalizeInPlace(fa);
      // Use seismic-based linearity for orientation scanning.
      // FaultSeg probability is spatially diffuse; the directional ridge
      // detector (FaultOrientScanner2) needs crisp edge-like structure at
      // fault locations.  The seismic linearity (faLocal) provides exactly
      // that: low linearity values right where horizontal reflectors are cut
      // by faults, forming narrow ridges that the dip scanner can track
      // reliably.  FaultSeg probability is used only as the voting score.
      faSeed = faLocal;
    } else {
      fa = faLocal;
      faSeed = faLocal;
    }

    FaultOrientScanner2 fos = new FaultOrientScanner2(8.0);
    float[][][] scan = bestScan(fos, faSeed);
    float[][][] scanThin = fos.thin(scan);
    // Combined score: FaultSeg (or local linearity) × seismic orientation
    // confidence.  Using fa here (not faSeed) ensures the external FaultSeg
    // probability drives the score even though faSeed was used for the scan.
    float[][] ft = combineScores(fa, normalize01(scanThin[0]));
    float[][] pt = scanThin[1];
    sanitizeInPlace(ft, 0.0f);
    sanitizeInPlace(pt, 80.0f);

    OptimalPathVoter opv = new OptimalPathVoter(15, 30);
    opv.setStrainMax(0.25);
    opv.setPathSmoothing(2.0);

    float[][] fan = normalize01(fa);
    // Gentler gating for external fault score: fa already encodes fault
    // likelihood, so a low floor and mild power preserve more signal while
    // still suppressing near-zero background.
    float prePower = useExternalFaultScore ? 1.0f : 1.5f;
    float preFloor = useExternalFaultScore ? 0.20f : 0.55f;
    float[][] scoreBase = guidedScore(ft, fan, prePower, preFloor);
    float[][] score = scoreBase;
    int minSeeds = useExternalFaultScore ? Math.max(48, (n1 * n2) / 3000) : Math.max(24, (n1 * n2) / 2500);
    float fm = chooseThreshold(opv, score, pt, minSeeds);
    if (countSeeds(opv, 4, fm, score, pt) == 0) {
      score = ft;
      fm = chooseThreshold(opv, score, pt, minSeeds);
    }
    if (countSeeds(opv, 4, fm, score, pt) == 0) {
      score = fa;
      fm = chooseThreshold(opv, score, pt, minSeeds);
    }
    if (!useExternalFaultScore && countSeeds(opv, 4, fm, score, pt) == 0) {
      score = faSeed;
      fm = chooseThreshold(opv, score, pt, minSeeds);
    }
    if (countSeeds(opv, 4, fm, score, pt) == 0) {
      fill(pt, 80.0f);
      score = faSeed;
      fm = chooseThreshold(opv, score, pt, minSeeds);
    }

    int seedCount = countSeeds(opv, 4, fm, score, pt);
    if (seedCount == 0) {
      throw new IllegalStateException("Unable to pick OSV seeds from the selected section.");
    }

    float[][][] fvw = opv.applyVoting(4, fm, score, pt);
    float[][] fv = fvw[0];
    float[][] w1 = fvw[1];
    float[][] w2 = fvw[2];
    float[][] fvt = opv.thin(fv, w1, w2);

    // Attribute-guided post-filter: gates OSV voting score by FaultSeg
    // probability to suppress seismic-linearity artefacts that lack
    // FaultSeg confirmation.  Mild power/floor to retain as much valid
    // fault signal as possible.
    float postPower = useExternalFaultScore ? 1.0f : 1.8f;
    float postFloor = useExternalFaultScore ? 0.15f : 0.6f;
    float[][] fvg = guidedScore(fv, fan, postPower, postFloor);
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

  private static float[][] read2D(String fileName, int n1, int n2) throws Exception {
    float[][] data = new float[n2][n1];
    ArrayInputStream ais = new ArrayInputStream(fileName);
    ais.readFloats(data);
    ais.close();
    return data;
  }

  private static float[][] normalize01(float[][] x) {
    int n2 = x.length;
    int n1 = x[0].length;
    float xmin = Float.MAX_VALUE;
    float xmax = -Float.MAX_VALUE;
    for (int i2 = 0; i2 < n2; ++i2) {
      for (int i1 = 0; i1 < n1; ++i1) {
        float v = x[i2][i1];
        if (!Float.isFinite(v)) continue;
        if (v < xmin) xmin = v;
        if (v > xmax) xmax = v;
      }
    }
    if (xmin == Float.MAX_VALUE || xmax == -Float.MAX_VALUE) {
      return new float[n2][n1];
    }
    float den = xmax - xmin;
    if (den < 1e-6f) den = 1e-6f;

    float[][] y = new float[n2][n1];
    for (int i2 = 0; i2 < n2; ++i2) {
      for (int i1 = 0; i1 < n1; ++i1) {
        float v = x[i2][i1];
        y[i2][i1] = Float.isFinite(v) ? (v - xmin) / den : 0.0f;
      }
    }
    return y;
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

  private static float chooseThreshold(OptimalPathVoter opv, float[][] score, float[][] pt, int minSeeds) {
    float[] candidates = new float[] {0.8f, 0.6f, 0.4f, 0.25f, 0.1f, 0.03f, 0.0f};
    float firstPositive = 0.0f;
    for (float candidate : candidates) {
      int seeds = countSeeds(opv, 4, candidate, score, pt);
      if (seeds > 0 && firstPositive == 0.0f) {
        firstPositive = candidate;
      }
      if (seeds >= minSeeds) {
        return candidate;
      }
    }
    if (firstPositive > 0.0f) {
      return firstPositive;
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

  private static void gammaInPlace(float[][] x, float power) {
    for (float[] row : x) {
      for (int i = 0; i < row.length; ++i) {
        float value = row[i];
        if (value < 0.0f) value = 0.0f;
        row[i] = (float)Math.pow(value, power);
      }
    }
  }

  private static void clipInPlace(float[][] x, float minValue, float maxValue) {
    for (float[] row : x) {
      for (int i = 0; i < row.length; ++i) {
        if (row[i] < minValue) row[i] = minValue;
        if (row[i] > maxValue) row[i] = maxValue;
      }
    }
  }

  private static void normalizeInPlace(float[][] x) {
    float[][] y = normalize01(x);
    for (int i2 = 0; i2 < x.length; ++i2) {
      System.arraycopy(y[i2], 0, x[i2], 0, x[i2].length);
    }
  }

  private static void fill(float[][] x, float value) {
    for (float[] row : x) {
      Arrays.fill(row, value);
    }
  }

  private static float[][] seedBlend(float[][] faultScore, float[][] localScore, float localWeight) {
    int n2 = faultScore.length;
    int n1 = faultScore[0].length;
    float[][] y = new float[n2][n1];
    for (int i2 = 0; i2 < n2; ++i2) {
      for (int i1 = 0; i1 < n1; ++i1) {
        float a = faultScore[i2][i1];
        float b = localScore[i2][i1] * localWeight;
        y[i2][i1] = Math.max(a, b);
      }
    }
    normalizeInPlace(y);
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

  private static float[][] guidedScore(float[][] score, float[][] attr, float power, float floorQuantile) {
    int n2 = score.length;
    int n1 = score[0].length;
    float floor = quantile(attr, floorQuantile);
    if (!Float.isFinite(floor)) floor = 0.0f;
    float den = 1.0f - floor;
    if (den < 1.0e-6f) den = 1.0f;
    float[][] y = new float[n2][n1];
    for (int i2 = 0; i2 < n2; ++i2) {
      for (int i1 = 0; i1 < n1; ++i1) {
        float a = attr[i2][i1];
        if (a < 0.0f) a = 0.0f;
        float gate = (a - floor) / den;
        if (gate < 0.0f) gate = 0.0f;
        if (gate > 1.0f) gate = 1.0f;
        y[i2][i1] = score[i2][i1] * gate * (float)Math.pow(a, power);
      }
    }
    normalizeInPlace(y);
    return y;
  }

  private static float quantile(float[][] x, float q) {
    if (q <= 0.0f) return minFinite(x);
    if (q >= 1.0f) return maxFinite(x);

    int n = x.length * x[0].length;
    float[] v = new float[n];
    int k = 0;
    for (float[] row : x) {
      for (float value : row) {
        if (Float.isFinite(value)) {
          v[k++] = value;
        }
      }
    }
    if (k == 0) return 0.0f;
    Arrays.sort(v, 0, k);
    float p = q * (k - 1);
    int i0 = (int)Math.floor(p);
    int i1 = Math.min(k - 1, i0 + 1);
    float w = p - i0;
    return (1.0f - w) * v[i0] + w * v[i1];
  }

  private static float minFinite(float[][] x) {
    float minValue = Float.MAX_VALUE;
    for (float[] row : x) {
      for (float value : row) {
        if (Float.isFinite(value) && value < minValue) {
          minValue = value;
        }
      }
    }
    return minValue == Float.MAX_VALUE ? 0.0f : minValue;
  }
}
