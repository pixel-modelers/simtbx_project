from __future__ import absolute_import, division, print_function
import time
import mmtbx.model
import iotbx.pdb
from mmtbx.hydrogens import reduce_hydrogen
from mmtbx.hydrogens.tst_add_hydrogen import compare_models
from libtbx.utils import null_out
#from libtbx.test_utils import approx_equal

# ------------------------------------------------------------------------------

def run():
  test_000()
  test_001()
  test_002()
  test_003()
  test_004()

# ------------------------------------------------------------------------------

def test_000():
  '''
    Make sure reduce does not crash for single_atom_residue models
  '''
  pdb_inp = iotbx.pdb.input(lines=pdb_str_000.split("\n"), source_info=None)
  # initial model (has no H atoms)
  model_initial = mmtbx.model.manager(model_input = pdb_inp, log = null_out())
  number_h_expected = model_initial.get_hd_selection().count(True)
  assert(number_h_expected == 0)
  # place H atoms
  reduce_add_h_obj = reduce_hydrogen.place_hydrogens(model = model_initial)
  reduce_add_h_obj.run()
  # We don't expect H atoms to be placed
  # (not enough restraints for single atom residues)
  model_h_added = reduce_add_h_obj.get_model()
  number_h_placed = model_h_added.get_hd_selection().count(True)
  assert(number_h_placed == 0)

# ------------------------------------------------------------------------------

def test_001():
  '''
    Check keyword n_terminal_charge:
    NH3 on resseq 1, first residue in chain, or no NH3 at all
  '''
  pdb_inp = iotbx.pdb.input(lines=pdb_str_001.split("\n"), source_info=None)
  # initial model
  model_initial = mmtbx.model.manager(model_input = pdb_inp, log = null_out())
  #
  # place H atoms: NH3 at resseq 1 only
  reduce_add_h_obj = reduce_hydrogen.place_hydrogens(model = model_initial)
  reduce_add_h_obj.run()
  model_h_added = reduce_add_h_obj.get_model()
  #
  hd_sel_h_added = model_h_added.get_hd_selection()
  ph_h_added     = model_h_added.get_hierarchy()
  h_atoms_added = ph_h_added.select(hd_sel_h_added).atoms()
  h_names_added = list(h_atoms_added.extract_name())
  assert(h_names_added.count(' H1 ')==1)
  assert(h_names_added.count(' H2 ')==1)
  assert(h_names_added.count(' H3 ')==1)
  #
  # place H atoms: NH3 at first residue in chain
  reduce_add_h_obj = reduce_hydrogen.place_hydrogens(
    model = model_initial,
    n_terminal_charge = 'first_in_chain')
  reduce_add_h_obj.run()
  model_h_added = reduce_add_h_obj.get_model()
  #
  hd_sel_h_added = model_h_added.get_hd_selection()
  ph_h_added     = model_h_added.get_hierarchy()
  h_atoms_added = ph_h_added.select(hd_sel_h_added).atoms()
  h_names_added = list(h_atoms_added.extract_name())
  assert(h_names_added.count(' H1 ')==3)
  assert(h_names_added.count(' H2 ')==3)
  assert(h_names_added.count(' H3 ')==3)
  #
  # place H atoms: no NH3
  reduce_add_h_obj = reduce_hydrogen.place_hydrogens(
    model = model_initial,
    n_terminal_charge = 'no_charge')
  reduce_add_h_obj.run()
  model_h_added = reduce_add_h_obj.get_model()
  #
  hd_sel_h_added = model_h_added.get_hd_selection()
  ph_h_added     = model_h_added.get_hierarchy()
  h_atoms_added = ph_h_added.select(hd_sel_h_added).atoms()
  h_names_added = list(h_atoms_added.extract_name())
  assert(h_names_added.count(' H1 ')==0)
  assert(h_names_added.count(' H2 ')==0)
  assert(h_names_added.count(' H3 ')==0)

# ------------------------------------------------------------------------------

def test_002():
  '''
    SIN forms covalent link to GLU 1. Make sure the default NH3  at the
    N-terminal becomes a single peptide H in this particular scenario.
  '''
  compare_models(pdb_str = pdb_str_002)

# ------------------------------------------------------------------------------

def test_003():
  '''
    Carbohydrate forms covlalent link to ASN. Make sure valence is correct for
    the NAG and protein links. Both parts are in double conformation.
  '''
  compare_models(pdb_str = pdb_str_003)

# ------------------------------------------------------------------------------

def test_004():
  '''
    Carbohydrate forms covlalent link to ASN. Make sure valence is correct for
    the NAG and protein links.
  '''
  compare_models(pdb_str = pdb_str_004)

# ------------------------------------------------------------------------------

pdb_str_000 = """
REMARK Make sure reduce does not crash for single_atom_residue models
CRYST1   22.029   33.502   24.035  90.00  90.00  90.00 P 1
ATOM      6  P     A A  10     -62.272  56.445  13.820  1.00 15.00           P
ATOM      7  P     G A  11     -63.673  51.410  11.026  1.00 15.00           P
ATOM      8  P     U A  12     -62.888  45.926   9.711  1.00 15.00           P
ATOM      9  P     U A  13     -60.326  41.305  11.244  1.00 15.00           P
ATOM     10  P     U A  14     -57.909  36.481  13.207  1.00 15.00           P
ATOM     11  P     G A  15     -62.106  32.943  15.800  1.00 15.00           P
ATOM     12  P     A A  16     -65.446  37.240  15.291  1.00 15.00           P
ATOM     13  P     U A  17     -66.286  42.354  18.232  1.00 15.00           P
ATOM     14  P     C A  18     -64.629  46.517  21.258  1.00 15.00           P
ATOM     15  P     A A  19     -60.460  50.019  23.746  1.00 15.00           P
ATOM     16  P     U A  20     -54.257  51.133  23.481  1.00 15.00           P
"""

pdb_str_001 = """
REMARK Make sure NH3 is applied correctly depending on keyword
CRYST1   30.200   47.800   61.300  90.00  90.00  90.00 P 21 21 21
ATOM      1  N   TYR A   5       8.831  48.837  54.788  1.00  9.67           N
ATOM      2  CA  TYR A   5       7.706  49.436  54.084  1.00  9.24           C
ATOM      3  C   TYR A   5       6.456  48.981  54.822  1.00 10.02           C
ATOM      4  O   TYR A   5       6.310  47.784  55.139  1.00 10.62           O
ATOM      5  CB  TYR A   5       7.599  48.942  52.633  1.00  9.83           C
ATOM      6  CG  TYR A   5       8.692  49.472  51.736  1.00 12.79           C
ATOM      7  CD1 TYR A   5       9.959  48.898  51.781  1.00 13.60           C
ATOM      8  CD2 TYR A   5       8.415  50.537  50.880  1.00 12.13           C
ATOM      9  CE1 TYR A   5      10.961  49.400  50.960  1.00 14.77           C
ATOM     10  CE2 TYR A   5       9.426  51.032  50.065  1.00 14.21           C
ATOM     11  CZ  TYR A   5      10.685  50.467  50.116  1.00 14.05           C
ATOM     12  OH  TYR A   5      11.708  50.978  49.318  1.00 17.48           O
ATOM     13  H1  TYR A   5       9.284  48.311  54.231  1.00  9.67           H
ATOM     14  H2  TYR A   5       9.367  49.479  55.092  1.00  9.67           H
ATOM     15  H3  TYR A   5       8.530  48.354  55.472  1.00  9.67           H
ATOM     16  HA  TYR A   5       7.805  50.400  54.049  1.00  9.24           H
ATOM     17  HB2 TYR A   5       7.653  47.974  52.627  1.00  9.83           H
ATOM     18  HB3 TYR A   5       6.748  49.229  52.266  1.00  9.83           H
ATOM     19  HD1 TYR A   5      10.133  48.187  52.354  1.00 13.60           H
ATOM     20  HD2 TYR A   5       7.564  50.912  50.855  1.00 12.13           H
ATOM     21  HE1 TYR A   5      11.811  49.024  50.976  1.00 14.77           H
ATOM     22  HE2 TYR A   5       9.255  51.741  49.488  1.00 14.21           H
ATOM     23  HH  TYR A   5      12.418  50.550  49.452  1.00 17.48           H
TER
ATOM     24  N   GLU B   1      14.684  52.510  58.829  1.00 14.47           N
ATOM     25  CA  GLU B   1      13.950  53.593  58.225  1.00 15.28           C
ATOM     26  C   GLU B   1      12.566  53.009  57.937  1.00 14.74           C
ATOM     27  O   GLU B   1      12.459  51.916  57.371  1.00 14.27           O
ATOM     28  CB  GLU B   1      14.667  53.949  56.967  1.00 18.89           C
ATOM     29  CG  GLU B   1      13.973  54.945  56.083  1.00 27.57           C
ATOM     30  CD  GLU B   1      14.729  55.326  54.802  1.00 32.66           C
ATOM     31  OE1 GLU B   1      15.802  54.776  54.508  1.00 34.50           O
ATOM     32  OE2 GLU B   1      14.224  56.201  54.090  1.00 36.48           O
ATOM     33  H1  GLU B   1      15.398  52.322  58.332  1.00 14.47           H
ATOM     34  H2  GLU B   1      14.945  52.748  59.646  1.00 14.47           H
ATOM     35  H3  GLU B   1      14.162  51.791  58.882  1.00 14.47           H
ATOM     36  HA  GLU B   1      13.870  54.399  58.759  1.00 15.28           H
ATOM     37  HB2 GLU B   1      15.529  54.326  57.204  1.00 18.89           H
ATOM     38  HB3 GLU B   1      14.790  53.139  56.447  1.00 18.89           H
ATOM     39  HG2 GLU B   1      13.835  55.760  56.590  1.00 27.57           H
ATOM     40  HG3 GLU B   1      13.119  54.573  55.814  1.00 27.57           H
TER
ATOM     41  N   PHE C  -3       6.984  40.342  58.778  1.00  8.80           N
ATOM     42  CA  PHE C  -3       7.384  40.247  60.166  1.00  8.32           C
ATOM     43  C   PHE C  -3       7.719  38.788  60.513  1.00  9.15           C
ATOM     44  O   PHE C  -3       8.710  38.555  61.201  1.00  9.31           O
ATOM     45  CB  PHE C  -3       6.284  40.754  61.091  1.00  8.94           C
ATOM     46  CG  PHE C  -3       6.705  40.642  62.560  1.00  9.84           C
ATOM     47  CD1 PHE C  -3       7.825  41.288  63.022  1.00 10.62           C
ATOM     48  CD2 PHE C  -3       5.989  39.828  63.426  1.00 12.63           C
ATOM     49  CE1 PHE C  -3       8.229  41.132  64.328  1.00 11.20           C
ATOM     50  CE2 PHE C  -3       6.398  39.679  64.737  1.00 13.74           C
ATOM     51  CZ  PHE C  -3       7.527  40.326  65.197  1.00 12.55           C
ATOM     52  H1  PHE C  -3       6.155  40.662  58.729  1.00  8.80           H
ATOM     53  H2  PHE C  -3       7.538  40.888  58.346  1.00  8.80           H
ATOM     54  H3  PHE C  -3       7.013  39.534  58.405  1.00  8.80           H
ATOM     55  HA  PHE C  -3       8.167  40.801  60.311  1.00  8.32           H
ATOM     56  HB2 PHE C  -3       6.101  41.686  60.894  1.00  8.94           H
ATOM     57  HB3 PHE C  -3       5.483  40.224  60.960  1.00  8.94           H
ATOM     58  HD1 PHE C  -3       8.313  41.834  62.449  1.00 10.62           H
ATOM     59  HD2 PHE C  -3       5.232  39.381  63.123  1.00 12.63           H
ATOM     60  HE1 PHE C  -3       8.988  41.578  64.629  1.00 11.20           H
ATOM     61  HE2 PHE C  -3       5.909  39.138  65.314  1.00 13.74           H
ATOM     62  HZ  PHE C  -3       7.808  40.220  66.077  1.00 12.55           H
TER
END
"""

pdb_str_002 = """
REMARK Make sure linking and valence are correct (keep one H of NH3 terminal)
CRYST1   33.386   33.386   65.691  90.00  90.00 120.00 H 3 2
SCALE1      0.029953  0.017293  0.000000        0.00000
SCALE2      0.000000  0.034586  0.000000        0.00000
SCALE3      0.000000  0.000000  0.015223        0.00000
ATOM      1  N   GLU A   1      10.117  25.200   2.571  1.00 -1.00           N
ANISOU    1  N   GLU A   1     3000   2835   1127    208   -142   -228       N
ATOM      2  CA  GLU A   1       9.338  24.612   3.642  1.00 -1.00           C
ANISOU    2  CA  GLU A   1     2595   2691   1161    236   -238   -217       C
ATOM      3  C   GLU A   1      10.179  23.909   4.698  1.00 -1.00           C
ANISOU    3  C   GLU A   1     2462   2375    957    271   -124   -310       C
ATOM      4  O   GLU A   1       9.927  24.071   5.895  1.00 -1.00           O
ANISOU    4  O   GLU A   1     2439   2116    850    143   -268   -383       O
ATOM      5  CB  GLU A   1       8.308  23.601   3.105  1.00 -1.00           C
ANISOU    5  CB  GLU A   1     2772   2664   1227    243   -283   -303       C
ATOM      6  CG  GLU A   1       7.502  22.878   4.179  1.00 -1.00           C
ANISOU    6  CG  GLU A   1     2867   3180   1472     61   -157   -268       C
ATOM      7  CD  GLU A   1       6.505  23.776   4.881  1.00 -1.00           C
ANISOU    7  CD  GLU A   1     3055   3241   1610    122   -165   -363       C
ATOM      8  OE1 GLU A   1       6.017  24.742   4.259  1.00 -1.00           O
ANISOU    8  OE1 GLU A   1     3535   3430   1924    346   -221   -295       O
ATOM      9  OE2 GLU A   1       6.197  23.520   6.058  1.00 -1.00           O
ANISOU    9  OE2 GLU A   1     3239   3344   1625     40   -119   -371       O
ATOM     10  H1  GLU A   1       9.876  24.837   1.795  1.00 18.32           H
ATOM     13  HA  GLU A   1       8.880  25.353   4.068  1.00 16.97           H
ATOM     14  HB2 GLU A   1       7.679  24.074   2.537  1.00 17.54           H
ATOM     15  HB3 GLU A   1       8.778  22.927   2.590  1.00 17.54           H
ATOM     16  HG2 GLU A   1       8.111  22.529   4.848  1.00 19.79           H
ATOM     17  HG3 GLU A   1       7.010  22.151   3.767  1.00 19.79           H
TER
HETATM   18  C1  SIN A   0      11.143  26.020   2.776  1.00 -1.00           C
ANISOU   18  C1  SIN A   0     3119   2972   1277    124   -163   -240       C
HETATM   19  C2  SIN A   0      11.845  26.557   1.542  1.00 -1.00           C
ANISOU   19  C2  SIN A   0     3459   3368   1388    102    -39   -151       C
HETATM   20  C3  SIN A   0      12.465  27.926   1.781  1.00 -1.00           C
ANISOU   20  C3  SIN A   0     3640   3427   1555     45     -9   -143       C
HETATM   21  C4  SIN A   0      11.485  29.081   1.820  1.00 -1.00           C
ANISOU   21  C4  SIN A   0     3756   3638   1693    179      8   -113       C
HETATM   22  O1  SIN A   0      11.536  26.296   3.914  1.00 -1.00           O
ANISOU   22  O1  SIN A   0     3338   3021   1204     43   -125   -241       O
HETATM   23  O3  SIN A   0      11.966  30.230   1.941  1.00 -1.00           O
ANISOU   23  O3  SIN A   0     3949   3621   1603    176    -78    -76       O
HETATM   24  O4  SIN A   0      10.281  28.844   2.062  1.00 -1.00           O
ANISOU   24  O4  SIN A   0     3699   3930   2011    285      2   -324       O
HETATM   25  H21 SIN A   0      11.127  26.627   0.725  1.00 21.62           H
HETATM   26  H22 SIN A   0      12.626  25.858   1.243  1.00 21.62           H
HETATM   27  H31 SIN A   0      13.002  27.899   2.729  1.00 22.69           H
HETATM   28  H32 SIN A   0      13.190  28.117   0.990  1.00 22.69           H
END
"""


pdb_str_003 = """
REMARK scenario for linking, valence and a double conformation
CRYST1   39.670   48.940   71.610  88.74  97.15 108.44 P 1
SCALE1      0.025208  0.008405  0.003318        0.00000
SCALE2      0.000000  0.021539  0.000398        0.00000
SCALE3      0.000000  0.000000  0.014076        0.00000
ATOM      1  N  AASN A  74      12.741   6.714   3.033  0.88  5.87           N
ATOM      2  CA AASN A  74      13.110   7.769   2.091  0.88  6.12           C
ATOM      3  C  AASN A  74      11.884   8.130   1.219  0.88  5.86           C
ATOM      4  O  AASN A  74      10.858   7.490   1.271  0.88  6.44           O
ATOM      5  CB AASN A  74      14.293   7.285   1.218  0.88  7.27           C
ATOM      6  CG AASN A  74      15.066   8.432   0.603  0.88  8.82           C
ATOM      7  OD1AASN A  74      15.487   9.375   1.229  0.88 11.36           O
ATOM      8  ND2AASN A  74      15.252   8.317  -0.703  0.88  8.84           N
ATOM      9  H  AASN A  74      13.382   6.467   3.550  0.88  5.87           H
ATOM     10  HA AASN A  74      13.396   8.571   2.556  0.88  6.12           H
ATOM     11  HB2AASN A  74      14.904   6.771   1.769  0.88  7.27           H
ATOM     12  HB3AASN A  74      13.951   6.732   0.498  0.88  7.27           H
ATOM     13 HD21AASN A  74      14.863   7.686  -1.139  0.88  8.84           H
ATOM     15  N  BASN A  74      12.696   6.684   3.030  0.12  5.92           N
ATOM     16  CA BASN A  74      13.040   7.695   2.058  0.12  6.02           C
ATOM     17  C  BASN A  74      11.859   8.100   1.181  0.12  5.93           C
ATOM     18  O  BASN A  74      10.802   7.435   1.187  0.12  7.07           O
ATOM     19  CB BASN A  74      14.200   7.209   1.214  0.12  5.18           C
ATOM     20  CG BASN A  74      15.493   7.906   1.579  0.12  3.11           C
ATOM     21  OD1BASN A  74      15.680   8.391   2.668  0.12  3.08           O
ATOM     22  ND2BASN A  74      16.411   7.915   0.608  0.12  3.22           N
ATOM     23  H  BASN A  74      13.365   6.396   3.487  0.12  5.92           H
ATOM     24  HA BASN A  74      13.309   8.501   2.527  0.12  6.02           H
ATOM     25  HB2BASN A  74      14.012   7.388   0.279  0.12  5.18           H
ATOM     26  HB3BASN A  74      14.319   6.256   1.352  0.12  5.18           H
ATOM     27 HD21BASN A  74      16.242   7.490  -0.121  0.12  3.22           H
TER
HETATM   29  C1 ANAG C   1      16.113   9.249  -1.424  0.74  8.24           C
HETATM   30  C2 ANAG C   1      16.970   8.487  -2.451  0.74  7.99           C
HETATM   31  C3 ANAG C   1      17.836   9.477  -3.177  0.74  8.76           C
HETATM   32  C4 ANAG C   1      17.107  10.640  -3.663  0.74 10.13           C
HETATM   33  C5 ANAG C   1      16.092  11.231  -2.643  0.74  8.62           C
HETATM   34  C6 ANAG C   1      15.151  12.287  -3.168  0.74  9.76           C
HETATM   35  C7 ANAG C   1      17.696   6.180  -1.895  0.74  8.89           C
HETATM   36  C8 ANAG C   1      18.571   5.397  -0.988  0.74  9.89           C
HETATM   37  N2 ANAG C   1      17.767   7.513  -1.768  0.74  8.57           N
HETATM   38  O3 ANAG C   1      18.515   8.763  -4.246  0.74 10.65           O
HETATM   39  O4 ANAG C   1      18.042  11.683  -3.981  0.74 13.35           O
HETATM   40  O5 ANAG C   1      15.310  10.151  -2.140  0.74  7.25           O
HETATM   41  O6 ANAG C   1      14.360  11.822  -4.218  0.74  8.68           O
HETATM   42  O7 ANAG C   1      16.861   5.728  -2.714  0.74 10.29           O
HETATM   43  H1 ANAG C   1      16.684   9.730  -0.805  0.74  8.24           H
HETATM   44  H2 ANAG C   1      16.401   8.030  -3.090  0.74  7.99           H
HETATM   45  H3 ANAG C   1      18.456   9.829  -2.519  0.74  8.76           H
HETATM   46  H4 ANAG C   1      16.631  10.272  -4.424  0.74 10.13           H
HETATM   47  H5 ANAG C   1      16.602  11.677  -1.949  0.74  8.62           H
HETATM   48  H61ANAG C   1      15.688  13.047  -3.442  0.74  9.76           H
HETATM   49  H62ANAG C   1      14.606  12.589  -2.424  0.74  9.76           H
HETATM   50  H81ANAG C   1      19.036   6.002  -0.389  0.74  9.89           H
HETATM   51  H82ANAG C   1      19.216   4.900  -1.515  0.74  9.89           H
HETATM   52  H83ANAG C   1      18.028   4.781  -0.471  0.74  9.89           H
HETATM   53  HN2ANAG C   1      18.352   7.824  -1.219  0.74  8.57           H
HETATM   55  HO3ANAG C   1      18.959   8.129  -3.895  0.74 10.65           H
HETATM   57  HO6ANAG C   1      14.872  11.499  -4.815  0.74  8.68           H
HETATM   58  C1 BNAG C   1      17.701   8.609   0.704  0.26  6.07           C
HETATM   59  C2 BNAG C   1      18.523   8.170  -0.427  0.26  8.31           C
HETATM   60  C3 BNAG C   1      19.761   8.820  -0.183  0.26  6.55           C
HETATM   61  C4 BNAG C   1      19.463  10.425  -0.498  0.26 10.04           C
HETATM   62  C5 BNAG C   1      18.496  10.802   0.555  0.26  8.70           C
HETATM   63  C6 BNAG C   1      17.911  12.176   0.535  0.26 13.23           C
HETATM   64  C7 BNAG C   1      18.720   5.989  -1.367  0.26 10.97           C
HETATM   65  C8 BNAG C   1      19.249   4.641  -1.192  0.26 12.42           C
HETATM   66  N2 BNAG C   1      18.962   6.846  -0.336  0.26  9.44           N
HETATM   67  O3 BNAG C   1      20.699   8.721  -1.435  0.26 14.71           O
HETATM   68  O4 BNAG C   1      20.552  11.276  -0.389  0.26 12.04           O
HETATM   69  O5 BNAG C   1      17.320   9.980   0.458  0.26  7.16           O
HETATM   70  O6 BNAG C   1      17.693  12.418  -0.804  0.26 19.92           O
HETATM   71  O7 BNAG C   1      18.181   6.220  -2.430  0.26 11.69           O
HETATM   72  H1 BNAG C   1      18.211   8.549   1.527  0.26  6.07           H
HETATM   73  H2 BNAG C   1      18.012   8.327  -1.237  0.26  8.31           H
HETATM   74  H3 BNAG C   1      20.062   8.479   0.674  0.26  6.55           H
HETATM   75  H4 BNAG C   1      19.176  10.456  -1.424  0.26 10.04           H
HETATM   76  H5 BNAG C   1      19.021  10.704   1.364  0.26  8.70           H
HETATM   77  H61BNAG C   1      18.525  12.807   0.942  0.26 13.23           H
HETATM   78  H62BNAG C   1      17.099  12.201   1.065  0.26 13.23           H
HETATM   79  H81BNAG C   1      19.656   4.567  -0.315  0.26 12.42           H
HETATM   80  H82BNAG C   1      19.915   4.466  -1.875  0.26 12.42           H
HETATM   81  H83BNAG C   1      18.526   4.000  -1.271  0.26 12.42           H
HETATM   82  HN2BNAG C   1      19.380   6.582   0.368  0.26  9.44           H
HETATM   84  HO3BNAG C   1      20.316   9.121  -2.080  0.26 14.71           H
HETATM   86  HO6BNAG C   1      18.423  12.288  -1.220  0.26 19.92           H
HETATM   87  C1 ANAG C   2      17.995  12.199  -5.260  0.74 19.86           C
HETATM   88  C2 ANAG C   2      18.677  13.574  -5.253  0.74 23.29           C
HETATM   89  C3 ANAG C   2      18.830  14.151  -6.591  0.74 29.68           C
HETATM   90  C4 ANAG C   2      19.318  13.037  -7.482  0.74 30.27           C
HETATM   91  C5 ANAG C   2      18.591  11.736  -7.460  0.74 29.19           C
HETATM   92  C6 ANAG C   2      19.149  10.584  -8.227  0.74 33.79           C
HETATM   93  C7 ANAG C   2      18.471  14.805  -3.050  0.74 27.91           C
HETATM   94  C8 ANAG C   2      17.772  15.886  -2.243  0.74 37.24           C
HETATM   95  N2 ANAG C   2      18.165  14.566  -4.332  0.74 22.31           N
HETATM   96  O3 ANAG C   2      19.864  15.164  -6.531  0.74 40.55           O
HETATM   97  O4 ANAG C   2      19.327  13.494  -8.818  0.74 46.76           O
HETATM   98  O5 ANAG C   2      18.644  11.227  -6.086  0.74 21.90           O
HETATM   99  O6 ANAG C   2      20.567  10.602  -8.018  0.74 41.31           O
HETATM  100  O7 ANAG C   2      19.396  14.260  -2.622  0.74 24.10           O
HETATM  101  H1 ANAG C   2      18.016  11.864  -4.350  0.74 19.86           H
HETATM  102  H2 ANAG C   2      19.547  13.331  -4.899  0.74 23.29           H
HETATM  103  H3 ANAG C   2      17.981  14.517  -6.883  0.74 29.68           H
HETATM  104  H4 ANAG C   2      20.179  12.882  -7.064  0.74 30.27           H
HETATM  105  H5 ANAG C   2      17.721  11.946  -7.834  0.74 29.19           H
HETATM  106  H61ANAG C   2      18.738   9.762  -7.916  0.74 33.79           H
HETATM  107  H62ANAG C   2      18.910  10.675  -9.163  0.74 33.79           H
HETATM  108  H81ANAG C   2      17.079  16.294  -2.785  0.74 37.24           H
HETATM  109  H82ANAG C   2      18.419  16.560  -1.981  0.74 37.24           H
HETATM  110  H83ANAG C   2      17.375  15.490  -1.451  0.74 37.24           H
HETATM  111  HN2ANAG C   2      17.567  15.083  -4.671  0.74 22.31           H
HETATM  113  HO3ANAG C   2      20.590  14.783  -6.307  0.74 40.55           H
HETATM  114  HO4ANAG C   2      19.521  14.322  -8.826  0.74 46.76           H
HETATM  115  HO6ANAG C   2      20.857  11.373  -8.226  0.74 41.31           H
HETATM  116  C1 BNAG C   2      20.710  12.370  -1.239  0.26 14.76           C
HETATM  117  C2 BNAG C   2      21.943  13.194  -0.806  0.26 19.56           C
HETATM  118  C3 BNAG C   2      22.215  14.268  -1.785  0.26 24.38           C
HETATM  119  C4 BNAG C   2      22.458  13.627  -3.140  0.26 21.94           C
HETATM  120  C5 BNAG C   2      21.254  12.760  -3.461  0.26 20.16           C
HETATM  121  C6 BNAG C   2      21.355  12.027  -4.829  0.26 23.66           C
HETATM  122  C7 BNAG C   2      22.490  13.708   1.511  0.26 32.27           C
HETATM  123  C8 BNAG C   2      21.906  14.269   2.792  0.26 36.29           C
HETATM  124  N2 BNAG C   2      21.644  13.753   0.496  0.26 26.94           N
HETATM  125  O3 BNAG C   2      23.271  15.027  -1.295  0.26 27.97           O
HETATM  126  O4 BNAG C   2      22.468  14.619  -4.109  0.26 26.97           O
HETATM  127  O5 BNAG C   2      20.976  11.868  -2.415  0.26 11.40           O
HETATM  128  O6 BNAG C   2      22.466  11.207  -5.128  0.26 29.08           O
HETATM  129  O7 BNAG C   2      23.600  13.383   1.407  0.26 21.94           O
HETATM  130  H1 BNAG C   2      19.928  12.942  -1.289  0.26 14.76           H
HETATM  131  H2 BNAG C   2      22.730  12.628  -0.768  0.26 19.56           H
HETATM  132  H3 BNAG C   2      21.443  14.847  -1.886  0.26 24.38           H
HETATM  133  H4 BNAG C   2      23.296  13.144  -3.071  0.26 21.94           H
HETATM  134  H5 BNAG C   2      20.480  13.340  -3.539  0.26 20.16           H
HETATM  135  H61BNAG C   2      21.284  12.729  -5.495  0.26 23.66           H
HETATM  136  H62BNAG C   2      20.541  11.503  -4.893  0.26 23.66           H
HETATM  137  H81BNAG C   2      20.980  14.517   2.641  0.26 36.29           H
HETATM  138  H82BNAG C   2      22.413  15.051   3.060  0.26 36.29           H
HETATM  139  H83BNAG C   2      21.953  13.595   3.488  0.26 36.29           H
HETATM  140  HN2BNAG C   2      20.879  14.130   0.606  0.26 26.94           H
HETATM  142  HO3BNAG C   2      23.951  14.520  -1.236  0.26 27.97           H
HETATM  143  HO4BNAG C   2      22.798  15.328  -3.776  0.26 26.97           H
HETATM  144  HO6BNAG C   2      23.173  11.663  -5.010  0.26 29.08           H
END
"""

pdb_str_004 = """
REMARK Linking + valence scenario for a carbohydrate
CRYST1   39.670   48.940   71.610  88.74  97.15 108.44 P 1
SCALE1      0.025208  0.008405  0.003318        0.00000
SCALE2      0.000000  0.021539  0.000398        0.00000
SCALE3      0.000000  0.000000  0.014076        0.00000
ATOM      1  N   ASN B  74      -3.026  34.995  30.688  1.00  9.05           N
ATOM      2  CA  ASN B  74      -2.493  35.978  31.595  1.00  9.88           C
ATOM      3  C   ASN B  74      -1.547  35.327  32.616  1.00  9.26           C
ATOM      4  O   ASN B  74      -1.420  34.120  32.689  1.00  9.22           O
ATOM      5  CB  ASN B  74      -3.624  36.727  32.324  1.00 12.24           C
ATOM      6  CG  ASN B  74      -3.216  38.094  32.845  1.00 14.21           C
ATOM      7  OD1 ASN B  74      -2.472  38.834  32.203  1.00 18.21           O
ATOM      8  ND2 ASN B  74      -3.692  38.409  34.028  1.00 16.96           N
ATOM      9  H   ASN B  74      -2.549  34.876  29.982  1.00  9.05           H
ATOM     10  HA  ASN B  74      -1.988  36.631  31.085  1.00  9.88           H
ATOM     11  HB2 ASN B  74      -3.912  36.195  33.082  1.00 12.24           H
ATOM     12  HB3 ASN B  74      -4.362  36.854  31.708  1.00 12.24           H
ATOM     13 HD21 ASN B  74      -4.198  37.847  34.438  1.00 16.96           H
TER
HETATM   15  C1  NAG D   1      -3.389  39.731  34.716  1.00 15.06           C
HETATM   16  C2  NAG D   1      -4.468  40.105  35.528  1.00 16.43           C
HETATM   17  C3  NAG D   1      -4.201  41.525  36.083  1.00 15.29           C
HETATM   18  C4  NAG D   1      -2.754  41.564  36.747  1.00 15.66           C
HETATM   19  C5  NAG D   1      -1.708  40.987  35.892  1.00 17.17           C
HETATM   20  C6  NAG D   1      -0.490  40.596  36.940  1.00 25.12           C
HETATM   21  C7  NAG D   1      -6.706  39.271  34.932  1.00 17.46           C
HETATM   22  C8  NAG D   1      -7.943  39.400  34.022  1.00 20.06           C
HETATM   23  N2  NAG D   1      -5.748  40.154  34.750  1.00 16.04           N
HETATM   24  O3  NAG D   1      -5.243  41.846  36.992  1.00 17.10           O
HETATM   25  O4  NAG D   1      -2.597  43.022  36.916  1.00 18.40           O
HETATM   26  O5  NAG D   1      -2.177  39.700  35.427  1.00 15.81           O
HETATM   27  O6  NAG D   1       0.621  40.445  35.919  1.00 26.15           O
HETATM   28  O7  NAG D   1      -6.670  38.382  35.716  1.00 17.36           O
HETATM   29  H1  NAG D   1      -3.305  40.387  34.006  1.00 15.06           H
HETATM   30  H2  NAG D   1      -4.545  39.445  36.235  1.00 16.43           H
HETATM   31  H3  NAG D   1      -4.201  42.160  35.350  1.00 15.29           H
HETATM   32  H4  NAG D   1      -2.726  41.025  37.553  1.00 15.66           H
HETATM   33  H5  NAG D   1      -1.456  41.561  35.151  1.00 17.17           H
HETATM   34  H61 NAG D   1      -0.689  39.788  37.439  1.00 25.12           H
HETATM   35  H62 NAG D   1      -0.340  41.295  37.596  1.00 25.12           H
HETATM   36  H81 NAG D   1      -7.800  40.119  33.386  1.00 20.06           H
HETATM   37  H82 NAG D   1      -8.722  39.596  34.566  1.00 20.06           H
HETATM   38  H83 NAG D   1      -8.079  38.566  33.546  1.00 20.06           H
HETATM   39  HN2 NAG D   1      -5.852  40.780  34.170  1.00 16.04           H
HETATM   41  HO3 NAG D   1      -5.213  41.286  37.631  1.00 17.10           H
HETATM   43  HO6 NAG D   1       0.364  39.894  35.325  1.00 26.15           H
HETATM   44  C1  NAG D   2      -2.391  43.486  38.203  1.00 21.11           C
HETATM   45  C2  NAG D   2      -1.775  44.823  37.997  1.00 26.59           C
HETATM   46  C3  NAG D   2      -1.585  45.513  39.294  1.00 26.26           C
HETATM   47  C4  NAG D   2      -2.959  45.457  39.969  1.00 29.92           C
HETATM   48  C5  NAG D   2      -3.462  44.035  40.104  1.00 30.15           C
HETATM   49  C6  NAG D   2      -4.756  43.848  40.745  1.00 35.86           C
HETATM   50  C7  NAG D   2      -0.234  44.953  35.978  1.00 30.77           C
HETATM   51  C8  NAG D   2       1.206  44.728  35.397  1.00 33.11           C
HETATM   52  N2  NAG D   2      -0.459  44.736  37.311  1.00 26.48           N
HETATM   53  O3  NAG D   2      -1.072  46.826  39.192  1.00 28.25           O
HETATM   54  O4  NAG D   2      -2.849  45.859  41.286  1.00 32.37           O
HETATM   55  O5  NAG D   2      -3.648  43.578  38.774  1.00 23.11           O
HETATM   56  O6  NAG D   2      -5.481  44.858  40.253  1.00 30.27           O
HETATM   57  O7  NAG D   2      -1.176  45.443  35.293  1.00 29.59           O
HETATM   58  H1  NAG D   2      -1.822  42.981  38.805  1.00 21.11           H
HETATM   59  H2  NAG D   2      -2.392  45.317  37.434  1.00 26.59           H
HETATM   60  H3  NAG D   2      -0.902  45.033  39.787  1.00 26.26           H
HETATM   61  H4  NAG D   2      -3.496  46.020  39.389  1.00 29.92           H
HETATM   62  H5  NAG D   2      -2.821  43.547  40.644  1.00 30.15           H
HETATM   63  H61 NAG D   2      -4.662  43.876  41.710  1.00 35.86           H
HETATM   64  H62 NAG D   2      -5.117  42.974  40.529  1.00 35.86           H
HETATM   65  H81 NAG D   2       1.786  44.392  36.098  1.00 33.11           H
HETATM   66  H82 NAG D   2       1.552  45.570  35.063  1.00 33.11           H
HETATM   67  H83 NAG D   2       1.163  44.083  34.673  1.00 33.11           H
HETATM   68  HN2 NAG D   2       0.216  44.531  37.802  1.00 26.48           H
HETATM   70  HO3 NAG D   2      -1.634  47.292  38.756  1.00 28.25           H
HETATM   71  HO4 NAG D   2      -2.248  46.457  41.343  1.00 32.37           H
HETATM   72  HO6 NAG D   2      -5.060  45.582  40.396  1.00 30.27           H
END
"""

# ------------------------------------------------------------------------------

if (__name__ == "__main__"):
  t0 = time.time()
  run()
  print("OK. Time: %8.3f"%(time.time()-t0))
