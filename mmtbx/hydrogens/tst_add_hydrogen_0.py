from __future__ import absolute_import, division, print_function
import time

from mmtbx.hydrogens.tst_add_hydrogen import compare_models

# ------------------------------------------------------------------------------

def run():
  test_000()
  test_001()
  test_002()
  # test_003()
  # test_004()
  # test_005()
  # test_006()
  # test_007()
  # test_008()

# ------------------------------------------------------------------------------

def test_000():
  '''
  Added a torsion to NH2_CTERM
  '''
  compare_models(pdb_str = pdb_str_000)

# ------------------------------------------------------------------------------

def test_001():
  '''
  A nucleotide with missing for some reason and a missing dihedral
  '''
  compare_models(pdb_str = pdb_str_001)

# ------------------------------------------------------------------------------

def test_002():
  '''
  Carbohydrates need more dihedrals
  '''
  compare_models(pdb_str = pdb_str_002)

# ------------------------------------------------------------------------------

pdb_str_000 = """
CRYST1   56.445   72.085  123.593  90.00  90.00  90.00 P 21 21 21
SCALE1      0.017716  0.000000  0.000000        0.00000
SCALE2      0.000000  0.013873  0.000000        0.00000
SCALE3      0.000000  0.000000  0.008091        0.00000
HETATM    1  N   DLE E  12     -46.710 -11.701  15.468  1.00 34.07           N
HETATM    2  CA  DLE E  12     -47.200 -11.850  14.105  1.00 39.64           C
HETATM    3  C   DLE E  12     -48.679 -11.482  14.009  1.00 41.13           C
HETATM    4  O   DLE E  12     -49.335 -11.798  13.021  1.00 42.80           O
HETATM    5  CB  DLE E  12     -46.390 -10.976  13.144  1.00 43.98           C
HETATM    6  CG  DLE E  12     -44.950 -11.415  12.889  1.00 46.65           C
HETATM    7  CD1 DLE E  12     -44.919 -12.765  12.197  1.00 46.88           C
HETATM    8  CD2 DLE E  12     -44.198 -10.363  12.085  1.00 48.25           C
HETATM   10  HA  DLE E  12     -47.098 -12.777  13.837  1.00 39.64           H
HETATM   11  HB2 DLE E  12     -46.356 -10.078  13.509  1.00 43.98           H
HETATM   12  HB3 DLE E  12     -46.844 -10.969  12.287  1.00 43.98           H
HETATM   13  HG  DLE E  12     -44.493 -11.511  13.739  1.00 46.65           H
HETATM   14 HD11 DLE E  12     -43.996 -13.021  12.046  1.00 46.88           H
HETATM   15 HD12 DLE E  12     -45.386 -12.696  11.349  1.00 46.88           H
HETATM   16 HD13 DLE E  12     -45.356 -13.420  12.763  1.00 46.88           H
HETATM   17 HD21 DLE E  12     -44.646 -10.236  11.234  1.00 48.25           H
HETATM   18 HD22 DLE E  12     -43.289 -10.668  11.938  1.00 48.25           H
HETATM   19 HD23 DLE E  12     -44.192  -9.530  12.582  1.00 48.25           H
HETATM   20  N   NH2 E 202     -49.193 -10.809  15.033  1.00 40.90           N
HETATM   21  HN1 NH2 E 202     -48.680 -10.596  15.728  1.00 40.90           H
HETATM   22  HN2 NH2 E 202     -50.051 -10.572  15.024  1.00 40.90           H
END
"""

pdb_str_001 = '''
CRYST1   60.683   61.851   76.893  90.00  90.00  90.00 P 21 21 21
SCALE1      0.016479  0.000000  0.000000        0.00000
SCALE2      0.000000  0.016168  0.000000        0.00000
SCALE3      0.000000  0.000000  0.013005        0.00000
HETATM    1  C1' ADP A1311      -7.459  14.326  10.821  0.60 14.02           C
HETATM    2  C2  ADP A1311      -5.449  12.545   7.224  0.60 15.85           C
HETATM    3  C2' ADP A1311      -7.594  15.611  11.604  0.60 14.04           C
HETATM    4  C3' ADP A1311      -8.097  15.126  12.927  0.60 13.75           C
HETATM    5  C4  ADP A1311      -6.903  13.885   8.453  0.60 14.26           C
HETATM    6  C4' ADP A1311      -8.964  13.965  12.513  0.60 14.88           C
HETATM    7  C5  ADP A1311      -7.414  14.430   7.198  0.60 14.79           C
HETATM    8  C5' ADP A1311     -10.403  14.410  12.351  0.60 12.89           C
HETATM    9  C6  ADP A1311      -6.877  13.954   5.942  0.60 15.37           C
HETATM   10  C8  ADP A1311      -8.458  15.356   8.818  0.60 13.69           C
HETATM   11  N1  ADP A1311      -5.907  13.025   6.029  0.60 16.08           N
HETATM   12  N3  ADP A1311      -5.925  12.949   8.425  0.60 15.93           N
HETATM   13  N6  ADP A1311      -7.358  14.471   4.786  0.60 16.26           N
HETATM   14  N7  ADP A1311      -8.348  15.322   7.484  0.60 15.22           N
HETATM   15  N9  ADP A1311      -7.603  14.521   9.383  0.60 14.10           N
HETATM   16  O1A ADP A1311     -12.593  15.064  10.083  0.60  6.78           O
HETATM   17  O1B ADP A1311     -11.480  19.808  11.509  0.30 13.50           O
HETATM   18  O2' ADP A1311      -6.345  16.239  11.737  0.60 12.16           O
HETATM   19  O2A ADP A1311     -12.829  15.834  12.325  0.60  8.51           O
HETATM   20  O2B ADP A1311     -10.576  17.778  12.692  0.30 17.96           O
HETATM   21  O3' ADP A1311      -7.034  14.659  13.737  0.60 15.01           O
HETATM   22  O3A ADP A1311     -11.739  17.443  10.617  0.60 12.19           O
HETATM   23  O3B ADP A1311     -13.078  18.233  12.535  0.20 17.22           O
HETATM   24  O4' ADP A1311      -8.522  13.493  11.245  0.60 15.01           O
HETATM   25  O5' ADP A1311     -10.557  15.534  11.511  0.60 12.05           O
HETATM   26  PA  ADP A1311     -12.004  15.966  11.098  0.60 10.28           P
HETATM   27  PB  ADP A1311     -11.709  18.390  11.914  0.30 15.89           P
HETATM   28  H2  ADP A1311      -4.669  11.794   7.209  0.60 15.85           H
HETATM   29  H1' ADP A1311      -6.491  13.859  11.051  0.60 14.02           H
HETATM   30  H2' ADP A1311      -8.331  16.303  11.184  0.60 14.04           H
HETATM   31  H3' ADP A1311      -8.694  15.901  13.428  0.60 13.75           H
HETATM   32  H4' ADP A1311      -8.905  13.169  13.269  0.60 14.88           H
HETATM   33  H8  ADP A1311      -9.149  16.001   9.344  0.60 13.69           H
HETATM   34 H5'1 ADP A1311     -10.814  14.645  13.335  0.60 12.89           H
HETATM   35 H5'2 ADP A1311     -10.986  13.581  11.944  0.60 12.89           H
HETATM   36 HN61 ADP A1311      -8.087  15.169   4.810  0.60 16.26           H
HETATM   37 HN62 ADP A1311      -6.991  14.154   3.900  0.60 16.26           H
HETATM   38 HO2' ADP A1311      -6.433  17.013  12.310  0.60 12.16           H
HETATM   39 HO3' ADP A1311      -6.507  15.409  14.043  0.60 15.01           H
'''

pdb_str_002 = '''
CRYST1   16.163   16.054   17.729  90.00  90.00  90.00 P 1
SCALE1      0.061870  0.000000  0.000000        0.00000
SCALE2      0.000000  0.062290  0.000000        0.00000
SCALE3      0.000000  0.000000  0.056405        0.00000
HETATM    1  C1  NAG A   1       7.207   7.892   8.696  1.00 20.00      A    C
HETATM    2  C2  NAG A   1       8.726   7.903   8.675  1.00 20.00      A    C
HETATM    3  C3  NAG A   1       9.299   7.873  10.037  1.00 20.00      A    C
HETATM    4  C4  NAG A   1       8.735   8.963  10.912  1.00 20.00      A    C
HETATM    5  C5  NAG A   1       7.211   8.952  10.928  1.00 20.00      A    C
HETATM    6  C6  NAG A   1       6.722  10.156  11.692  1.00 20.00      A    C
HETATM    7  C7  NAG A   1       9.920   6.866   6.642  1.00 20.00      A    C
HETATM    8  C8  NAG A   1      10.391   5.633   5.851  1.00 20.00      A    C
HETATM    9  N2  NAG A   1       9.210   6.694   7.918  1.00 20.00      A    N
HETATM   10  O1  NAG A   1       6.748   8.064   7.430  1.00 20.00      A    O
HETATM   11  O3  NAG A   1      10.730   8.045   9.943  1.00 20.00      A    O
HETATM   12  O4  NAG A   1       9.211   8.778  12.243  1.00 20.00      A    O
HETATM   13  O5  NAG A   1       6.644   8.983   9.574  1.00 20.00      A    O
HETATM   14  O6  NAG A   1       5.328  10.234  11.597  1.00 20.00      A    O
HETATM   15  O7  NAG A   1      10.109   7.947   6.214  1.00 20.00      A    O
HETATM   16  H1  NAG A   1       6.928   7.042   9.072  1.00 20.00      A    H
HETATM   17  H2  NAG A   1       9.004   8.727   8.246  1.00 20.00      A    H
HETATM   18  H3  NAG A   1       9.042   7.013  10.404  1.00 20.00      A    H
HETATM   19  H4  NAG A   1       9.054   9.784  10.505  1.00 20.00      A    H
HETATM   20  H5  NAG A   1       6.893   8.131  11.334  1.00 20.00      A    H
HETATM   21  H61 NAG A   1       7.154  10.946  11.330  1.00 20.00      A    H
HETATM   22  H62 NAG A   1       7.018  10.077  12.612  1.00 20.00      A    H
HETATM   23  H81 NAG A   1      10.095   4.828   6.305  1.00 20.00      A    H
HETATM   24  H82 NAG A   1      11.359   5.638   5.795  1.00 20.00      A    H
HETATM   25  H83 NAG A   1      10.013   5.661   4.958  1.00 20.00      A    H
HETATM   26  HN2 NAG A   1       9.068   5.908   8.238  1.00 20.00      A    H
HETATM   27  HO1 NAG A   1       6.819   8.882   7.211  1.00 20.00      A    H
HETATM   28  HO3 NAG A   1      10.919   8.822  10.230  1.00 20.00      A    H
HETATM   29  HO4 NAG A   1       9.549   8.001  12.310  1.00 20.00      A    H
HETATM   30  HO6 NAG A   1       5.126  10.415  10.791  1.00 20.00      A    H
END
'''

if (__name__ == "__main__"):
  t0 = time.time()
  run()
  print("OK. Time: %8.3f"%(time.time()-t0))