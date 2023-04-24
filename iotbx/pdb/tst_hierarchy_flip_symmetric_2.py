from __future__ import division
import sys
from libtbx import easy_run

pdbs = {
  '6exy_ARG_168_H.pdb' : '''
CRYST1   37.314   58.352   63.867  90.00  90.00  90.00 P 21 21 21
ATOM      1  N   ARG A 168      22.726 -15.648  12.491  1.00 14.41           N
ATOM      2  CA  ARG A 168      22.811 -14.357  13.143  1.00 15.09           C
ATOM      3  C   ARG A 168      21.569 -13.549  12.811  1.00 13.95           C
ATOM      4  O   ARG A 168      20.857 -13.814  11.839  1.00 14.06           O
ATOM      5  CB  ARG A 168      24.057 -13.582  12.689  1.00 17.54           C
ATOM      6  CG  ARG A 168      25.343 -14.366  12.775  1.00 19.91           C
ATOM      7  CD  ARG A 168      25.613 -14.857  14.174  1.00 20.97           C
ATOM      8  NE  ARG A 168      26.737 -15.790  14.190  1.00 21.81           N
ATOM      9  CZ  ARG A 168      26.658 -17.059  14.578  1.00 21.76           C
ATOM     10  NH1 ARG A 168      27.742 -17.813  14.539  1.00 21.87           N
ATOM     11  NH2 ARG A 168      25.521 -17.587  15.017  1.00 21.76           N
ATOM     12  H   ARG A 168      22.668 -15.599  11.475  1.00 14.15           H
ATOM     13  HA  ARG A 168      22.844 -14.490  14.224  1.00 15.42           H
ATOM     14  HB2 ARG A 168      23.915 -13.295  11.650  1.00 17.96           H
ATOM     15  HB3 ARG A 168      24.174 -12.692  13.305  1.00 17.77           H
ATOM     16  HD2 ARG A 168      25.863 -14.006  14.804  1.00 21.25           H
ATOM     17  HD3 ARG A 168      24.731 -15.354  14.564  1.00 21.33           H
ATOM     18  HE  ARG A 168      27.476 -15.591  13.512  1.00 21.85           H
ATOM     19  HG2 ARG A 168      25.288 -15.225  12.102  1.00 20.44           H
ATOM     20  HG3 ARG A 168      26.173 -13.708  12.500  1.00 20.06           H
ATOM     21 HH11 ARG A 168      28.617 -17.413  14.207  1.00 22.11           H
ATOM     22 HH12 ARG A 168      27.696 -18.788  14.835  1.00 21.99           H
ATOM     23 HH21 ARG A 168      25.514 -18.567  15.308  1.00 21.74           H
ATOM     24 HH22 ARG A 168      24.658 -17.044  15.059  1.00 21.55           H
TER''',
  '6exy_ARG_168_D.pdb' : '''
CRYST1   37.314   58.352   63.867  90.00  90.00  90.00 P 21 21 21
ATOM      1  N   ARG A 168      22.726 -15.648  12.491  1.00 14.41           N
ATOM      2  CA  ARG A 168      22.811 -14.357  13.143  1.00 15.09           C
ATOM      3  C   ARG A 168      21.569 -13.549  12.811  1.00 13.95           C
ATOM      4  O   ARG A 168      20.857 -13.814  11.839  1.00 14.06           O
ATOM      5  CB  ARG A 168      24.057 -13.582  12.689  1.00 17.54           C
ATOM      6  CG  ARG A 168      25.343 -14.366  12.775  1.00 19.91           C
ATOM      7  CD  ARG A 168      25.613 -14.857  14.174  1.00 20.97           C
ATOM      8  NE  ARG A 168      26.737 -15.790  14.190  1.00 21.81           N
ATOM      9  CZ  ARG A 168      26.658 -17.059  14.578  1.00 21.76           C
ATOM     10  NH1 ARG A 168      27.742 -17.813  14.539  1.00 21.87           N
ATOM     11  NH2 ARG A 168      25.521 -17.587  15.017  1.00 21.76           N
ATOM     12  D   ARG A 168      22.668 -15.599  11.475  1.00 14.15           D
ATOM     13  DA  ARG A 168      22.844 -14.490  14.224  1.00 15.42           D
ATOM     14  DB2 ARG A 168      23.915 -13.295  11.650  1.00 17.96           D
ATOM     15  DB3 ARG A 168      24.174 -12.692  13.305  1.00 17.77           D
ATOM     16  DD2 ARG A 168      25.863 -14.006  14.804  1.00 21.25           D
ATOM     17  DD3 ARG A 168      24.731 -15.354  14.564  1.00 21.33           D
ATOM     18  DE  ARG A 168      27.476 -15.591  13.512  1.00 21.85           D
ATOM     19  DG2 ARG A 168      25.288 -15.225  12.102  1.00 20.44           D
ATOM     20  DG3 ARG A 168      26.173 -13.708  12.500  1.00 20.06           D
ATOM     21 DH11 ARG A 168      28.617 -17.413  14.207  1.00 22.11           D
ATOM     22 DH12 ARG A 168      27.696 -18.788  14.835  1.00 21.99           D
ATOM     23 DH21 ARG A 168      25.514 -18.567  15.308  1.00 21.74           D
ATOM     24 DH22 ARG A 168      24.658 -17.044  15.059  1.00 21.55           D
TER''',
  '6exy_ARG_168_ac_H.pdb': '''
CRYST1   37.314   58.352   63.867  90.00  90.00  90.00 P 21 21 21
SCALE1      0.026800  0.000000  0.000000        0.00000
SCALE2      0.000000  0.017137  0.000000        0.00000
SCALE3      0.000000  0.000000  0.015658        0.00000
ATOM      1  N  AARG A 168      22.726 -15.648  12.491  0.46 14.41           N
ATOM      2  CA AARG A 168      22.811 -14.357  13.143  0.46 15.09           C
ATOM      3  C  AARG A 168      21.569 -13.549  12.811  0.46 13.95           C
ATOM      4  O  AARG A 168      20.857 -13.814  11.839  0.46 14.06           O
ATOM      5  CB AARG A 168      24.057 -13.582  12.689  0.46 17.54           C
ATOM      6  CG AARG A 168      25.343 -14.366  12.775  0.46 19.91           C
ATOM      7  CD AARG A 168      25.613 -14.857  14.174  0.46 20.97           C
ATOM      8  NE AARG A 168      26.737 -15.790  14.190  0.46 21.81           N
ATOM      9  CZ AARG A 168      26.658 -17.059  14.578  0.46 21.76           C
ATOM     10  NH1AARG A 168      27.742 -17.813  14.539  0.46 21.87           N
ATOM     11  NH2AARG A 168      25.521 -17.587  15.017  0.46 21.76           N
ATOM     12  H  AARG A 168      22.668 -15.599  11.475  0.46 14.15           H
ATOM     13  HA AARG A 168      22.844 -14.490  14.224  0.46 15.42           H
ATOM     14  HB2AARG A 168      23.915 -13.295  11.650  0.46 17.96           H
ATOM     15  HB3AARG A 168      24.174 -12.692  13.305  0.46 17.77           H
ATOM     16  HD2AARG A 168      25.863 -14.006  14.804  0.46 21.25           H
ATOM     17  HD3AARG A 168      24.731 -15.354  14.564  0.46 21.33           H
ATOM     18  HE AARG A 168      27.476 -15.591  13.512  0.46 21.85           H
ATOM     19  HG2AARG A 168      25.288 -15.225  12.102  0.46 20.44           H
ATOM     20  HG3AARG A 168      26.173 -13.708  12.500  0.46 20.06           H
ATOM     21 HH11AARG A 168      28.617 -17.413  14.207  0.46 22.11           H
ATOM     22 HH12AARG A 168      27.696 -18.788  14.835  0.46 21.99           H
ATOM     23 HH21AARG A 168      25.514 -18.567  15.308  0.46 21.74           H
ATOM     24 HH22AARG A 168      24.658 -17.044  15.059  0.46 21.55           H
ATOM     25  N  BARG A 168      22.711 -15.649  12.474  0.54 14.14           N
ATOM     26  CA BARG A 168      22.916 -14.337  13.062  0.54 14.70           C
ATOM     27  C  BARG A 168      21.754 -13.431  12.679  0.54 13.29           C
ATOM     28  O  BARG A 168      21.184 -13.538  11.588  0.54 13.90           O
ATOM     29  CB BARG A 168      24.218 -13.716  12.533  0.54 17.49           C
ATOM     30  CG BARG A 168      25.439 -14.599  12.700  0.54 20.99           C
ATOM     31  CD BARG A 168      26.144 -14.335  14.005  0.54 22.91           C
ATOM     32  NE BARG A 168      27.298 -15.226  14.175  0.54 24.26           N
ATOM     33  CZ BARG A 168      28.570 -14.887  13.964  0.54 24.87           C
ATOM     34  NH1BARG A 168      29.519 -15.802  14.139  0.54 24.92           N
ATOM     35  NH2BARG A 168      28.905 -13.657  13.596  0.54 24.77           N
ATOM     36  H  BARG A 168      22.563 -15.624  11.467  0.54 13.85           H
ATOM     37  HA BARG A 168      22.959 -14.410  14.146  0.54 15.03           H
ATOM     38  HB2BARG A 168      24.096 -13.533  11.468  0.54 17.99           H
ATOM     39  HB3BARG A 168      24.418 -12.777  13.046  0.54 17.29           H
ATOM     40  HD2BARG A 168      26.474 -13.296  14.007  0.54 23.34           H
ATOM     41  HD3BARG A 168      25.450 -14.509  14.828  0.54 22.84           H
ATOM     42  HE BARG A 168      27.141 -16.106  14.675  0.54 24.26           H
ATOM     43  HG2BARG A 168      25.143 -15.649  12.671  0.54 21.23           H
ATOM     44  HG3BARG A 168      26.149 -14.391  11.896  0.54 21.33           H
ATOM     45 HH11BARG A 168      29.265 -16.742  14.428  0.54 24.77           H
ATOM     46 HH12BARG A 168      30.500 -15.563  13.994  0.54 25.10           H
ATOM     47 HH21BARG A 168      29.887 -13.425  13.448  0.54 24.89           H
ATOM     48 HH22BARG A 168      28.193 -12.942  13.455  0.54 24.59           H
TER''',
  '6exy_ARG_168_ac_D.pdb': '''
CRYST1   37.314   58.352   63.867  90.00  90.00  90.00 P 21 21 21
ATOM      1  N  AARG A 168      22.726 -15.648  12.491  0.46 14.41           N
ATOM      2  CA AARG A 168      22.811 -14.357  13.143  0.46 15.09           C
ATOM      3  C  AARG A 168      21.569 -13.549  12.811  0.46 13.95           C
ATOM      4  O  AARG A 168      20.857 -13.814  11.839  0.46 14.06           O
ATOM      5  CB AARG A 168      24.057 -13.582  12.689  0.46 17.54           C
ATOM      6  CG AARG A 168      25.343 -14.366  12.775  0.46 19.91           C
ATOM      7  CD AARG A 168      25.613 -14.857  14.174  0.46 20.97           C
ATOM      8  NE AARG A 168      26.737 -15.790  14.190  0.46 21.81           N
ATOM      9  CZ AARG A 168      26.658 -17.059  14.578  0.46 21.76           C
ATOM     10  NH1AARG A 168      27.742 -17.813  14.539  0.46 21.87           N
ATOM     11  NH2AARG A 168      25.521 -17.587  15.017  0.46 21.76           N
ATOM     12  D  AARG A 168      22.668 -15.599  11.475  0.46 14.15           D
ATOM     13  DA AARG A 168      22.844 -14.490  14.224  0.46 15.42           D
ATOM     14  DB2AARG A 168      23.915 -13.295  11.650  0.46 17.96           D
ATOM     15  DB3AARG A 168      24.174 -12.692  13.305  0.46 17.77           D
ATOM     16  DD2AARG A 168      25.863 -14.006  14.804  0.46 21.25           D
ATOM     17  DD3AARG A 168      24.731 -15.354  14.564  0.46 21.33           D
ATOM     18  DE AARG A 168      27.476 -15.591  13.512  0.46 21.85           D
ATOM     19  DG2AARG A 168      25.288 -15.225  12.102  0.46 20.44           D
ATOM     20  DG3AARG A 168      26.173 -13.708  12.500  0.46 20.06           D
ATOM     21 DH11AARG A 168      28.617 -17.413  14.207  0.46 22.11           D
ATOM     22 DH12AARG A 168      27.696 -18.788  14.835  0.46 21.99           D
ATOM     23 DH21AARG A 168      25.514 -18.567  15.308  0.46 21.74           D
ATOM     24 DH22AARG A 168      24.658 -17.044  15.059  0.46 21.55           D
ATOM     25  N  BARG A 168      22.711 -15.649  12.474  0.54 14.14           N
ATOM     26  CA BARG A 168      22.916 -14.337  13.062  0.54 14.70           C
ATOM     27  C  BARG A 168      21.754 -13.431  12.679  0.54 13.29           C
ATOM     28  O  BARG A 168      21.184 -13.538  11.588  0.54 13.90           O
ATOM     29  CB BARG A 168      24.218 -13.716  12.533  0.54 17.49           C
ATOM     30  CG BARG A 168      25.439 -14.599  12.700  0.54 20.99           C
ATOM     31  CD BARG A 168      26.144 -14.335  14.005  0.54 22.91           C
ATOM     32  NE BARG A 168      27.298 -15.226  14.175  0.54 24.26           N
ATOM     33  CZ BARG A 168      28.570 -14.887  13.964  0.54 24.87           C
ATOM     34  NH1BARG A 168      29.519 -15.802  14.139  0.54 24.92           N
ATOM     35  NH2BARG A 168      28.905 -13.657  13.596  0.54 24.77           N
ATOM     36  D  BARG A 168      22.563 -15.624  11.467  0.54 13.85           D
ATOM     37  DA BARG A 168      22.959 -14.410  14.146  0.54 15.03           D
ATOM     38  DB2BARG A 168      24.096 -13.533  11.468  0.54 17.99           D
ATOM     39  DB3BARG A 168      24.418 -12.777  13.046  0.54 17.29           D
ATOM     40  DD2BARG A 168      26.474 -13.296  14.007  0.54 23.34           D
ATOM     41  DD3BARG A 168      25.450 -14.509  14.828  0.54 22.84           D
ATOM     42  DE BARG A 168      27.141 -16.106  14.675  0.54 24.26           D
ATOM     43  DG2BARG A 168      25.143 -15.649  12.671  0.54 21.23           D
ATOM     44  DG3BARG A 168      26.149 -14.391  11.896  0.54 21.33           D
ATOM     45 DH11BARG A 168      29.265 -16.742  14.428  0.54 24.77           D
ATOM     46 DH12BARG A 168      30.500 -15.563  13.994  0.54 25.10           D
ATOM     47 DH21BARG A 168      29.887 -13.425  13.448  0.54 24.89           D
ATOM     48 DH22BARG A 168      28.193 -12.942  13.455  0.54 24.59           D
TER''',
  '6exy_ARG_168_HD.pdb': '''
CRYST1   37.314   58.352   63.867  90.00  90.00  90.00 P 21 21 21
ATOM      1  N   ARG A 168      22.726 -15.648  12.491  1.00 14.41           N
ATOM      2  CA  ARG A 168      22.811 -14.357  13.143  1.00 15.09           C
ATOM      3  C   ARG A 168      21.569 -13.549  12.811  1.00 13.95           C
ATOM      4  O   ARG A 168      20.857 -13.814  11.839  1.00 14.06           O
ATOM      5  CB  ARG A 168      24.057 -13.582  12.689  1.00 17.54           C
ATOM      6  CG  ARG A 168      25.343 -14.366  12.775  1.00 19.91           C
ATOM      7  CD  ARG A 168      25.613 -14.857  14.174  1.00 20.97           C
ATOM      8  NE  ARG A 168      26.737 -15.790  14.190  1.00 21.81           N
ATOM      9  CZ  ARG A 168      26.658 -17.059  14.578  1.00 21.76           C
ATOM     10  NH1 ARG A 168      27.742 -17.813  14.539  1.00 21.87           N
ATOM     11  NH2 ARG A 168      25.521 -17.587  15.017  1.00 21.76           N

ATOM     12  H  AARG A 168      22.668 -15.599  11.475  0.50 14.15           H
ATOM     13  HA AARG A 168      22.844 -14.490  14.224  0.50 15.42           H
ATOM     14  HB2AARG A 168      23.915 -13.295  11.650  0.50 17.96           H
ATOM     15  HB3AARG A 168      24.174 -12.692  13.305  0.50 17.77           H
ATOM     16  HD2AARG A 168      25.863 -14.006  14.804  0.50 21.25           H
ATOM     17  HD3AARG A 168      24.731 -15.354  14.564  0.50 21.33           H
ATOM     18  HE AARG A 168      27.476 -15.591  13.512  0.50 21.85           H
ATOM     19  HG2AARG A 168      25.288 -15.225  12.102  0.50 20.44           H
ATOM     20  HG3AARG A 168      26.173 -13.708  12.500  0.50 20.06           H
ATOM     21 HH11AARG A 168      28.617 -17.413  14.207  0.50 22.11           H
ATOM     22 HH12AARG A 168      27.696 -18.788  14.835  0.50 21.99           H
ATOM     23 HH21AARG A 168      25.514 -18.567  15.308  0.50 21.74           H
ATOM     24 HH22AARG A 168      24.658 -17.044  15.059  0.50 21.55           H

ATOM     12  D  BARG A 168      22.668 -15.599  11.475  0.50 14.15           D
ATOM     13  DA BARG A 168      22.844 -14.490  14.224  0.50 15.42           D
ATOM     14  DB2BARG A 168      23.915 -13.295  11.650  0.50 17.96           D
ATOM     15  DB3BARG A 168      24.174 -12.692  13.305  0.50 17.77           D
ATOM     16  DD2BARG A 168      25.863 -14.006  14.804  0.50 21.25           D
ATOM     17  DD3BARG A 168      24.731 -15.354  14.564  0.50 21.33           D
ATOM     18  DE BARG A 168      27.476 -15.591  13.512  0.50 21.85           D
ATOM     19  DG2BARG A 168      25.288 -15.225  12.102  0.50 20.44           D
ATOM     20  DG3BARG A 168      26.173 -13.708  12.500  0.50 20.06           D
ATOM     21 DH11BARG A 168      28.617 -17.413  14.207  0.50 22.11           D
ATOM     22 DH12BARG A 168      27.696 -18.788  14.835  0.50 21.99           D
ATOM     23 DH21BARG A 168      25.514 -18.567  15.308  0.50 21.74           D
ATOM     24 DH22BARG A 168      24.658 -17.044  15.059  0.50 21.55           D
TER
END''',
  '6exy_ARG_168.pdb' : '''
CRYST1   37.314   58.352   63.867  90.00  90.00  90.00 P 21 21 21
SCALE1      0.026800  0.000000  0.000000        0.00000
SCALE2      0.000000  0.017137  0.000000        0.00000
SCALE3      0.000000  0.000000  0.015658        0.00000
ATOM      1  N  AARG A 168      22.726 -15.648  12.491  0.46 14.41           N
ANISOU    1  N  AARG A 168     2956    966   1555    581    -86    136       N
ATOM      2  CA AARG A 168      22.811 -14.357  13.143  0.46 15.09           C
ANISOU    2  CA AARG A 168     3060   1053   1622    589   -181    175       C
ATOM      3  C  AARG A 168      21.569 -13.549  12.811  0.46 13.95           C
ANISOU    3  C  AARG A 168     2841   1018   1442    621   -138    182       C
ATOM      4  O  AARG A 168      20.857 -13.814  11.839  0.46 14.06           O
ANISOU    4  O  AARG A 168     2826   1004   1514    573   -118    163       O
ATOM      5  CB AARG A 168      24.057 -13.582  12.689  0.46 17.54           C
ANISOU    5  CB AARG A 168     3330   1445   1891    513   -228    128       C
ATOM      6  CG AARG A 168      25.343 -14.366  12.775  0.46 19.91           C
ANISOU    6  CG AARG A 168     3575   1848   2140    467   -301     33       C
ATOM      7  CD AARG A 168      25.613 -14.857  14.174  0.46 20.97           C
ANISOU    7  CD AARG A 168     3477   2162   2327    406   -383    -59       C
ATOM      8  NE AARG A 168      26.737 -15.790  14.190  0.46 21.81           N
ANISOU    8  NE AARG A 168     3407   2382   2496    366   -487   -107       N
ATOM      9  CZ AARG A 168      26.658 -17.059  14.578  0.46 21.76           C
ANISOU    9  CZ AARG A 168     3277   2448   2542    575   -532    -57       C
ATOM     10  NH1AARG A 168      27.742 -17.813  14.539  0.46 21.87           N
ANISOU   10  NH1AARG A 168     3066   2581   2663    814   -403     67       N
ATOM     11  NH2AARG A 168      25.521 -17.587  15.017  0.46 21.76           N
ANISOU   11  NH2AARG A 168     3326   2429   2513    552   -594   -236       N
ATOM     12  D  AARG A 168      22.668 -15.599  11.475  0.46 14.15           D
ANISOU   12  D  AARG A 168     2885    950   1542    582   -141    105       D
ATOM     13  DA AARG A 168      22.844 -14.490  14.224  0.46 15.42           D
ANISOU   13  DA AARG A 168     3152   1054   1653    583    -81    106       D
ATOM     14  DB2AARG A 168      23.915 -13.295  11.650  0.46 17.96           D
ANISOU   14  DB2AARG A 168     3476   1436   1912    501   -173    212       D
ATOM     15  DB3AARG A 168      24.174 -12.692  13.305  0.46 17.77           D
ANISOU   15  DB3AARG A 168     3355   1464   1931    387   -291    131       D
ATOM     16  DD2AARG A 168      25.863 -14.006  14.804  0.46 21.25           D
ANISOU   16  DD2AARG A 168     3522   2186   2366    390   -353    -79       D
ATOM     17  DD3AARG A 168      24.731 -15.354  14.564  0.46 21.33           D
ANISOU   17  DD3AARG A 168     3561   2231   2311    424   -438    -36       D
ATOM     18  DE AARG A 168      27.476 -15.591  13.512  0.46 21.85           D
ANISOU   18  DE AARG A 168     3347   2457   2499    283   -457    -83       D
ATOM     19  DG2AARG A 168      25.288 -15.225  12.102  0.46 20.44           D
ANISOU   19  DG2AARG A 168     3738   1887   2143    380   -335     48       D
ATOM     20  DG3AARG A 168      26.173 -13.708  12.500  0.46 20.06           D
ANISOU   20  DG3AARG A 168     3622   1863   2138    488   -337     59       D
ATOM     21 DH11AARG A 168      28.617 -17.413  14.207  0.46 22.11           D
ANISOU   21 DH11AARG A 168     3047   2635   2719    804   -373     54       D
ATOM     22 DH12AARG A 168      27.696 -18.788  14.835  0.46 21.99           D
ANISOU   22 DH12AARG A 168     3066   2597   2691    903   -413    136       D
ATOM     23 DH21AARG A 168      25.514 -18.567  15.308  0.46 21.74           D
ANISOU   23 DH21AARG A 168     3330   2402   2528    485   -659   -218       D
ATOM     24 DH22AARG A 168      24.658 -17.044  15.059  0.46 21.55           D
ANISOU   24 DH22AARG A 168     3338   2400   2449    616   -597   -337       D
ATOM     25  N  BARG A 168      22.711 -15.649  12.474  0.54 14.14           N
ANISOU   25  N  BARG A 168     2909    890   1574    431    -88    194       N
ATOM     26  CA BARG A 168      22.916 -14.337  13.062  0.54 14.70           C
ANISOU   26  CA BARG A 168     3012    930   1644    272   -155    322       C
ATOM     27  C  BARG A 168      21.754 -13.431  12.679  0.54 13.29           C
ANISOU   27  C  BARG A 168     2672    878   1499    320   -282    274       C
ATOM     28  O  BARG A 168      21.184 -13.538  11.588  0.54 13.90           O
ANISOU   28  O  BARG A 168     2857    906   1520    224   -204    196       O
ATOM     29  CB BARG A 168      24.218 -13.716  12.533  0.54 17.49           C
ANISOU   29  CB BARG A 168     3378   1367   1900     45     39    378       C
ATOM     30  CG BARG A 168      25.439 -14.599  12.700  0.54 20.99           C
ANISOU   30  CG BARG A 168     3923   1867   2185    -86     70    392       C
ATOM     31  CD BARG A 168      26.144 -14.335  14.005  0.54 22.91           C
ANISOU   31  CD BARG A 168     4161   2216   2329    -68    -22    374       C
ATOM     32  NE BARG A 168      27.298 -15.226  14.175  0.54 24.26           N
ANISOU   32  NE BARG A 168     4227   2551   2439      4      7    450       N
ATOM     33  CZ BARG A 168      28.570 -14.887  13.964  0.54 24.87           C
ANISOU   33  CZ BARG A 168     4262   2732   2456     26     43    553       C
ATOM     34  NH1BARG A 168      29.519 -15.802  14.139  0.54 24.92           N
ANISOU   34  NH1BARG A 168     4235   2750   2482    165     50    505       N
ATOM     35  NH2BARG A 168      28.905 -13.657  13.596  0.54 24.77           N
ANISOU   35  NH2BARG A 168     4229   2838   2344    -54     88    715       N
ATOM     36  D  BARG A 168      22.563 -15.624  11.467  0.54 13.85           D
ANISOU   36  D  BARG A 168     2828    866   1568    428   -157    162       D
ATOM     37  DA BARG A 168      22.959 -14.410  14.146  0.54 15.03           D
ANISOU   37  DA BARG A 168     3128    919   1663    262    -42    263       D
ATOM     38  DB2BARG A 168      24.096 -13.533  11.468  0.54 17.99           D
ANISOU   38  DB2BARG A 168     3546   1336   1954     37     29    428       D
ATOM     39  DB3BARG A 168      24.418 -12.777  13.046  0.54 17.29           D
ANISOU   39  DB3BARG A 168     3306   1368   1894   -108    103    377       D
ATOM     40  DD2BARG A 168      26.474 -13.296  14.007  0.54 23.34           D
ANISOU   40  DD2BARG A 168     4263   2253   2351    -81    -55    351       D
ATOM     41  DD3BARG A 168      25.450 -14.509  14.828  0.54 22.84           D
ANISOU   41  DD3BARG A 168     4113   2203   2364   -115    -47    340       D
ATOM     42  DE BARG A 168      27.141 -16.106  14.675  0.54 24.26           D
ANISOU   42  DE BARG A 168     4250   2553   2413      8     10    478       D
ATOM     43  DG2BARG A 168      25.143 -15.649  12.671  0.54 21.23           D
ANISOU   43  DG2BARG A 168     3960   1882   2225   -198     76    419       D
ATOM     44  DG3BARG A 168      26.149 -14.391  11.896  0.54 21.33           D
ANISOU   44  DG3BARG A 168     3970   1942   2193   -123    193    405       D
ATOM     45 DH11BARG A 168      29.265 -16.742  14.428  0.54 24.77           D
ANISOU   45 DH11BARG A 168     4246   2702   2463    241     54    430       D
ATOM     46 DH12BARG A 168      30.500 -15.563  13.994  0.54 25.10           D
ANISOU   46 DH12BARG A 168     4250   2794   2491    119     60    519       D
ATOM     47 DH21BARG A 168      29.887 -13.425  13.448  0.54 24.89           D
ANISOU   47 DH21BARG A 168     4260   2888   2309    -56    137    739       D
ATOM     48 DH22BARG A 168      28.193 -12.942  13.455  0.54 24.59           D
ANISOU   48 DH22BARG A 168     4200   2854   2289    -90    166    783       D
TER
END''',
  '6exy_ARG_168_iso_hydrogen.pdb' : '''
CRYST1   37.314   58.352   63.867  90.00  90.00  90.00 P 21 21 21
SCALE1      0.026800  0.000000  0.000000        0.00000
SCALE2      0.000000  0.017137  0.000000        0.00000
SCALE3      0.000000  0.000000  0.015658        0.00000
ATOM      1  N  AARG A 168      22.726 -15.648  12.491  0.46 14.41           N
ATOM      2  CA AARG A 168      22.811 -14.357  13.143  0.46 15.09           C
ATOM      3  C  AARG A 168      21.569 -13.549  12.811  0.46 13.95           C
ATOM      4  O  AARG A 168      20.857 -13.814  11.839  0.46 14.06           O
ATOM      5  CB AARG A 168      24.057 -13.582  12.689  0.46 17.54           C
ATOM      6  CG AARG A 168      25.343 -14.366  12.775  0.46 19.91           C
ATOM      7  CD AARG A 168      25.613 -14.857  14.174  0.46 20.97           C
ATOM      8  NE AARG A 168      26.737 -15.790  14.190  0.46 21.81           N
ATOM      9  CZ AARG A 168      26.658 -17.059  14.578  0.46 21.76           C
ATOM     10  NH1AARG A 168      27.742 -17.813  14.539  0.46 21.87           N
ATOM     11  NH2AARG A 168      25.521 -17.587  15.017  0.46 21.76           N
ATOM     12  H  AARG A 168      22.668 -15.599  11.475  0.46 14.15           H
ATOM     13  HA AARG A 168      22.844 -14.490  14.224  0.46 15.42           H
ATOM     14  HB2AARG A 168      23.915 -13.295  11.650  0.46 17.96           H
ATOM     15  HB3AARG A 168      24.174 -12.692  13.305  0.46 17.77           H
ATOM     16  HD2AARG A 168      25.863 -14.006  14.804  0.46 21.25           H
ATOM     17  HD3AARG A 168      24.731 -15.354  14.564  0.46 21.33           H
ATOM     18  HE AARG A 168      27.476 -15.591  13.512  0.46 21.85           H
ATOM     19  HG2AARG A 168      25.288 -15.225  12.102  0.46 20.44           H
ATOM     20  HG3AARG A 168      26.173 -13.708  12.500  0.46 20.06           H
ATOM     21 HH11AARG A 168      28.617 -17.413  14.207  0.46 22.11           H
ATOM     22 HH12AARG A 168      27.696 -18.788  14.835  0.46 21.99           H
ATOM     23 HH21AARG A 168      25.514 -18.567  15.308  0.46 21.74           H
ATOM     24 HH22AARG A 168      24.658 -17.044  15.059  0.46 21.55           H
ATOM     25  N  BARG A 168      22.711 -15.649  12.474  0.54 14.14           N
ATOM     26  CA BARG A 168      22.916 -14.337  13.062  0.54 14.70           C
ATOM     27  C  BARG A 168      21.754 -13.431  12.679  0.54 13.29           C
ATOM     28  O  BARG A 168      21.184 -13.538  11.588  0.54 13.90           O
ATOM     29  CB BARG A 168      24.218 -13.716  12.533  0.54 17.49           C
ATOM     30  CG BARG A 168      25.439 -14.599  12.700  0.54 20.99           C
ATOM     31  CD BARG A 168      26.144 -14.335  14.005  0.54 22.91           C
ATOM     32  NE BARG A 168      27.298 -15.226  14.175  0.54 24.26           N
ATOM     33  CZ BARG A 168      28.570 -14.887  13.964  0.54 24.87           C
ATOM     34  NH1BARG A 168      29.519 -15.802  14.139  0.54 24.92           N
ATOM     35  NH2BARG A 168      28.905 -13.657  13.596  0.54 24.77           N
ATOM     36  H  BARG A 168      22.563 -15.624  11.467  0.54 13.85           H
ATOM     37  HA BARG A 168      22.959 -14.410  14.146  0.54 15.03           H
ATOM     38  HB2BARG A 168      24.096 -13.533  11.468  0.54 17.99           H
ATOM     39  HB3BARG A 168      24.418 -12.777  13.046  0.54 17.29           H
ATOM     40  HD2BARG A 168      26.474 -13.296  14.007  0.54 23.34           H
ATOM     41  HD3BARG A 168      25.450 -14.509  14.828  0.54 22.84           H
ATOM     42  HE BARG A 168      27.141 -16.106  14.675  0.54 24.26           H
ATOM     43  HG2BARG A 168      25.143 -15.649  12.671  0.54 21.23           H
ATOM     44  HG3BARG A 168      26.149 -14.391  11.896  0.54 21.33           H
ATOM     45 HH11BARG A 168      29.265 -16.742  14.428  0.54 24.77           H
ATOM     46 HH12BARG A 168      30.500 -15.563  13.994  0.54 25.10           H
ATOM     47 HH21BARG A 168      29.887 -13.425  13.448  0.54 24.89           H
ATOM     48 HH22BARG A 168      28.193 -12.942  13.455  0.54 24.59           H
TER
END''',
}

def main(only_i=None):
  try: only_i=int(only_i)
  except Exception: only_i=None
  for file_name, lines in pdbs.items():
    print(file_name)
    f=open(file_name, 'w')
    f.write(lines)
    del f
    for i in range(2):
      cmd = 'phenix.pdb_interpretation %s flip_symmetric=%s' % (file_name, i)
      print(cmd)
      rc = easy_run.go(cmd)
      for line in rc.stdout_lines:
        if line.find('target:')>-1:
          print(line)
          tmp=line.split()
          target=float(tmp[1])
          assert target<2000

if __name__ == '__main__':
  main(*tuple(sys.argv[1:]))
