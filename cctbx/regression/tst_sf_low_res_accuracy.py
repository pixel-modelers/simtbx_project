import iotbx.pdb
from scitbx.array_family import flex
import time

pdb_str="""
CRYST1  454.007  426.033  459.047  90.00  90.00  90.00 P 1
SCALE1      0.002203  0.000000  0.000000        0.00000
SCALE2      0.000000  0.002347  0.000000        0.00000
SCALE3      0.000000  0.000000  0.002178        0.00000
ATOM      1  CA  VAL A1358     148.938 119.717 100.000  1.00 33.88           C
ATOM      2  CA  PRO A1359     149.108 116.282 101.953  1.00 31.91           C
ATOM      3  CA  PRO A1360     146.078 114.542 103.797  1.00 26.33           C
ATOM      4  CA  PRO A1361     145.621 112.862 107.337  1.00 18.42           C
ATOM      5  CA  THR A1362     145.081 109.243 108.683  1.00 21.37           C
ATOM      6  CA  ASP A1363     143.651 106.972 111.575  1.00 20.53           C
ATOM      7  CA  LEU A1364     140.301 108.604 112.732  1.00 13.10           C
ATOM      8  CA  ARG A1365     138.856 107.558 116.217  1.00 13.90           C
ATOM      9  CA  PHE A1366     136.295 108.604 118.951  1.00 12.59           C
ATOM     10  CA  THR A1367     136.338 108.844 122.851  1.00 18.27           C
ATOM     11  CA  ASN A1368     134.748 110.767 125.861  1.00 35.66           C
ATOM     12  CA  ILE A1369     131.093 110.630 124.665  1.00 20.50           C
ATOM     13  CA  GLY A1370     128.787 112.946 126.681  1.00 31.22           C
ATOM     14  CA  PRO A1371     125.231 114.133 125.725  1.00 30.22           C
ATOM     15  CA  ASP A1372     126.459 117.360 124.041  1.00 23.75           C
ATOM     16  CA  THR A1373     130.112 116.268 123.662  1.00 17.75           C
ATOM     17  CA  MET A1374     132.554 113.816 121.995  1.00 13.70           C
ATOM     18  CA  ARG A1375     136.395 113.642 121.605  1.00 12.83           C
ATOM     19  CA  VAL A1376     137.530 112.924 118.003  1.00 11.64           C
ATOM     20  CA  THR A1377     141.175 111.755 117.305  1.00 13.55           C
ATOM     21  CA  TRP A1378     143.436 111.275 114.163  1.00 10.25           C
ATOM     22  CA  ALA A1379     147.077 110.870 112.883  1.00 20.66           C
ATOM     23  CA  PRO A1380     148.832 113.776 110.968  1.00 25.69           C
ATOM     24  CA  PRO A1381     150.776 113.842 107.607  1.00 34.10           C
ATOM     25  CA  ASP A1385     154.007 122.439 109.035  1.00 55.68           C
ATOM     26  CA  LEU A1386     150.309 122.713 108.083  1.00 39.19           C
ATOM     27  CA  THR A1387     148.167 125.829 108.846  1.00 33.58           C
ATOM     28  CA  ASN A1388     144.910 123.873 109.428  1.00 18.33           C
ATOM     29  CA  PHE A1389     142.989 120.679 110.062  1.00 13.50           C
ATOM     30  CA  LEU A1390     139.443 121.388 108.928  1.00 14.05           C
ATOM     31  CA  VAL A1391     137.092 119.270 111.076  1.00 11.67           C
ATOM     32  CA  ARG A1392     133.729 119.258 109.231  1.00 12.02           C
ATOM     33  CA  TYR A1393     130.897 117.502 111.152  1.00 10.39           C
ATOM     34  CA  SER A1394     127.111 117.090 110.574  1.00 16.06           C
ATOM     35  CA  PRO A1395     124.173 114.979 111.936  1.00 17.95           C
ATOM     36  CA  VAL A1396     123.808 111.698 109.904  1.00 24.54           C
ATOM     37  CA  LYS A1397     120.134 112.689 109.250  1.00 44.54           C
ATOM     38  CA  ASN A1398     121.139 116.103 107.784  1.00 43.24           C
ATOM     39  CA  GLU A1399     124.609 115.685 106.151  1.00 54.32           C
ATOM     40  CA  GLU A1400     124.190 119.276 104.701  1.00 42.39           C
ATOM     41  CA  VAL A1402     128.032 120.991 108.262  1.00 34.73           C
ATOM     42  CA  ALA A1403     129.573 122.671 111.271  1.00 31.08           C
ATOM     43  CA  GLU A1404     133.269 123.451 110.546  1.00 18.73           C
ATOM     44  CA  LEU A1405     136.125 123.710 113.086  1.00 23.21           C
ATOM     45  CA  SER A1406     139.558 125.155 112.173  1.00 19.59           C
ATOM     46  CA  ILE A1407     141.752 123.692 114.833  1.00 22.08           C
ATOM     47  CA  SER A1408     145.018 126.033 115.249  1.00 33.50           C
ATOM     48  CA  PRO A1409     148.634 124.336 115.685  1.00 35.29           C
ATOM     49  CA  SER A1410     148.476 120.979 118.025  1.00 37.93           C
ATOM     50  CA  ASP A1411     145.131 118.710 118.597  1.00 27.98           C
ATOM     51  CA  ASN A1412     145.573 115.280 117.118  1.00 21.72           C
ATOM     52  CA  ALA A1413     142.225 115.275 119.063  1.00 16.22           C
ATOM     53  CA  VAL A1414     139.281 117.816 119.201  1.00 15.68           C
ATOM     54  CA  VAL A1415     136.371 117.803 121.698  1.00 15.31           C
ATOM     55  CA  LEU A1416     133.225 118.528 119.662  1.00 16.01           C
ATOM     56  CA  THR A1417     130.718 120.452 121.897  1.00 20.05           C
ATOM     57  CA  ASN A1418     127.070 121.739 121.572  1.00 32.11           C
ATOM     58  CA  LEU A1419     125.988 118.404 119.986  1.00 15.97           C
ATOM     59  CA  LEU A1420     122.395 117.086 119.982  1.00 22.66           C
ATOM     60  CA  PRO A1421     121.812 114.423 122.733  1.00 20.32           C
ATOM     61  CA  GLY A1422     121.263 110.823 121.509  1.00 23.77           C
ATOM     62  CA  THR A1423     122.224 111.850 117.885  1.00 18.46           C
ATOM     63  CA  GLU A1424     124.604 110.218 115.318  1.00 14.18           C
ATOM     64  CA  TYR A1425     127.176 112.502 113.569  1.00 13.02           C
ATOM     65  CA  VAL A1426     129.513 112.297 110.551  1.00 12.30           C
ATOM     66  CA  VAL A1427     132.997 113.789 111.143  1.00 12.16           C
ATOM     67  CA  SER A1428     135.627 114.576 108.470  1.00 11.24           C
ATOM     68  CA  VAL A1429     139.197 116.027 108.576  1.00 11.54           C
ATOM     69  CA  SER A1430     141.130 117.805 105.734  1.00 13.82           C
ATOM     70  CA  SER A1431     144.703 119.108 106.173  1.00 18.85           C
ATOM     71  CA  VAL A1432     145.115 122.769 105.027  1.00 24.56           C
ATOM     72  CA  TYR A1433     148.261 124.777 104.098  1.00 45.81           C
ATOM     73  CA  HIS A1436     144.719 125.120 100.347  1.00 55.81           C
ATOM     74  CA  GLU A1437     142.703 122.002 101.363  1.00 29.48           C
ATOM     75  CA  SER A1438     143.992 118.417 101.042  1.00 22.89           C
ATOM     76  CA  THR A1439     141.620 115.400 100.644  1.00 28.41           C
ATOM     77  CA  PRO A1440     139.399 114.638 103.739  1.00 17.14           C
ATOM     78  CA  LEU A1441     139.506 111.600 106.101  1.00 14.44           C
ATOM     79  CA  ARG A1442     135.936 110.522 107.346  1.00 14.35           C
ATOM     80  CA  GLY A1443     134.058 108.636 110.191  1.00 15.43           C
ATOM     81  CA  ARG A1444     130.819 108.445 112.368  1.00 11.66           C
ATOM     82  CA  GLN A1445     129.772 108.415 116.114  1.00 11.50           C
ATOM     83  CA  LYS A1446     126.623 108.803 118.415  1.00 16.19           C
ATOM     84  CA  THR A1447     126.219 111.062 121.551  1.00 17.99           C
ATOM     85  CA  GLY A1448     124.870 110.091 125.040  1.00 29.95           C
ATOM     86  CA  LEU A1449     121.592 110.926 126.887  1.00 22.66           C
ATOM     87  CA  ASP A1450     121.190 114.131 128.936  1.00 29.34           C
ATOM     88  CA  SER A1451     119.589 114.286 132.415  1.00 22.82           C
ATOM     89  CA  PRO A1452     116.259 115.773 133.432  1.00 20.71           C
ATOM     90  CA  THR A1453     116.918 119.260 134.966  1.00 21.30           C
ATOM     91  CA  GLY A1454     115.025 121.690 137.311  1.00 40.30           C
ATOM     92  CA  ILE A1455     113.515 119.461 140.065  1.00 15.59           C
ATOM     93  CA  ASP A1456     110.666 121.305 141.864  1.00 20.21           C
ATOM     94  CA  PHE A1457     108.039 120.443 144.565  1.00 13.19           C
ATOM     95  CA  SER A1458     104.364 121.636 144.401  1.00 31.40           C
ATOM     96  CA  ASP A1459     100.972 120.385 145.865  1.00 30.41           C
ATOM     97  CA  ILE A1460     102.658 120.059 149.305  1.00 24.94           C
ATOM     98  CA  THR A1461     100.000 118.617 151.677  1.00 48.83           C
ATOM     99  CA  ASN A1463     100.070 113.301 152.758  1.00 22.51           C
ATOM    100  CA  SER A1464     101.256 114.201 149.190  1.00 18.68           C
ATOM    101  CA  PHE A1465     103.502 116.463 147.133  1.00 13.23           C
ATOM    102  CA  THR A1466     103.878 116.832 143.319  1.00 15.46           C
ATOM    103  CA  VAL A1467     107.394 116.566 141.859  1.00 13.69           C
ATOM    104  CA  HIS A1468     108.182 118.418 138.594  1.00 14.58           C
ATOM    105  CA  TRP A1469     111.247 118.176 136.299  1.00 10.82           C
ATOM    106  CA  ILE A1470     112.365 119.716 132.973  1.00 19.51           C
ATOM    107  CA  ALA A1471     112.613 117.148 130.133  1.00 23.06           C
ATOM    108  CA  PRO A1472     116.007 116.133 128.576  1.00 26.02           C
ATOM    109  CA  ARG A1473     116.707 117.205 124.930  1.00 24.80           C
ATOM    110  CA  ALA A1474     117.405 113.548 123.992  1.00 30.93           C
ATOM    111  CA  THR A1475     114.556 111.456 122.577  1.00 27.61           C
ATOM    112  CA  ILE A1476     113.479 109.292 125.565  1.00 20.47           C
ATOM    113  CA  THR A1477     111.104 106.367 126.245  1.00 20.70           C
ATOM    114  CA  GLY A1478     110.236 107.515 129.830  1.00 17.71           C
ATOM    115  CA  TYR A1479     111.339 108.336 133.412  1.00 11.13           C
ATOM    116  CA  ARG A1480     112.030 106.377 136.642  1.00 11.29           C
ATOM    117  CA  ILE A1481     111.354 108.176 139.957  1.00 11.59           C
ATOM    118  CA  ARG A1482     112.549 106.852 143.378  1.00 11.91           C
ATOM    119  CA  HIS A1483     111.280 108.193 146.740  1.00 12.39           C
ATOM    120  CA  HIS A1484     111.933 107.485 150.478  1.00 13.07           C
ATOM    121  CA  PRO A1485     111.962 109.317 153.882  1.00 16.85           C
ATOM    122  CA  GLU A1486     115.450 110.852 154.527  1.00 35.55           C
ATOM    123  CA  HIS A1487     116.161 108.541 157.533  1.00 37.44           C
ATOM    124  CA  PHE A1488     114.775 105.425 155.653  1.00 55.77           C
ATOM    125  CA  GLY A1490     114.925 100.391 153.173  1.00 51.28           C
ATOM    126  CA  ARG A1491     114.526 100.000 149.373  1.00 35.07           C
ATOM    127  CA  PRO A1492     113.029 103.299 147.995  1.00 34.67           C
ATOM    128  CA  ARG A1493     109.580 103.211 146.323  1.00 19.34           C
ATOM    129  CA  GLU A1494     110.097 103.197 142.505  1.00 17.33           C
ATOM    130  CA  ASP A1495     107.545 104.782 140.129  1.00 18.92           C
ATOM    131  CA  ARG A1496     107.622 104.735 136.272  1.00 14.77           C
ATOM    132  CA  VAL A1497     106.408 107.703 134.219  1.00 18.52           C
ATOM    133  CA  PRO A1498     105.929 107.937 130.379  1.00 22.40           C
ATOM    134  CA  HIS A1499     108.129 110.340 128.324  1.00 27.21           C
ATOM    135  CA  SER A1500     104.897 112.436 127.939  1.00 31.96           C
ATOM    136  CA  ARG A1501     104.944 113.394 131.694  1.00 25.65           C
ATOM    137  CA  ASN A1502     107.309 115.925 133.305  1.00 18.08           C
ATOM    138  CA  SER A1503     105.596 115.470 136.749  1.00 13.26           C
ATOM    139  CA  ILE A1504     104.291 113.003 139.407  1.00 11.86           C
ATOM    140  CA  THR A1505     102.098 113.449 142.526  1.00 12.54           C
ATOM    141  CA  LEU A1506     103.719 111.288 145.225  1.00 13.17           C
ATOM    142  CA  THR A1507     100.924 110.090 147.581  1.00 17.45           C
ATOM    143  CA  ASN A1508     100.502 108.068 150.815  1.00 21.22           C
ATOM    144  CA  LEU A1509     103.276 110.082 152.508  1.00 15.59           C
ATOM    145  CA  THR A1510     103.316 110.677 156.334  1.00 27.65           C
ATOM    146  CA  PRO A1511     102.558 114.210 157.787  1.00 40.50           C
ATOM    147  CA  GLY A1512     105.530 116.313 159.047  1.00 44.65           C
ATOM    148  CA  THR A1513     108.047 113.929 157.313  1.00 22.55           C
ATOM    149  CA  GLU A1514     110.992 114.686 154.966  1.00 21.10           C
ATOM    150  CA  TYR A1515     111.497 112.635 151.738  1.00 12.67           C
ATOM    151  CA  VAL A1516     114.448 112.279 149.296  1.00 13.40           C
ATOM    152  CA  VAL A1517     113.412 111.940 145.595  1.00 11.78           C
ATOM    153  CA  SER A1518     115.583 110.904 142.590  1.00 12.34           C
ATOM    154  CA  ILE A1519     114.725 110.895 138.832  1.00 11.14           C
ATOM    155  CA  VAL A1520     116.321 109.121 135.793  1.00 12.44           C
ATOM    156  CA  ALA A1521     115.462 109.388 132.060  1.00 14.96           C
ATOM    157  CA  LEU A1522     115.507 106.239 129.850  1.00 16.54           C
ATOM    158  CA  ASN A1523     116.032 105.607 126.094  1.00 25.63           C
ATOM    159  CA  GLY A1524     115.153 101.870 125.768  1.00 29.20           C
ATOM    160  CA  ARG A1525     118.103 100.567 127.906  1.00 25.61           C
ATOM    161  CA  GLU A1526     120.342 103.685 128.085  1.00 26.12           C
ATOM    162  CA  GLU A1527     119.837 105.577 131.422  1.00 19.28           C
ATOM    163  CA  SER A1528     120.671 109.264 132.149  1.00 21.68           C
ATOM    164  CA  PRO A1529     122.567 110.558 135.203  1.00 26.78           C
ATOM    165  CA  LEU A1530     120.377 111.168 138.317  1.00 17.50           C
ATOM    166  CA  LEU A1531     118.417 114.337 139.198  1.00 14.81           C
ATOM    167  CA  ILE A1532     117.866 114.501 143.053  1.00 15.20           C
ATOM    168  CA  GLY A1533     115.890 116.681 145.574  1.00 19.48           C
ATOM    169  CA  GLN A1534     114.319 116.753 149.115  1.00 13.66           C
ATOM    170  CA  GLN A1535     110.796 117.747 150.352  1.00 14.79           C
ATOM    171  CA  SER A1536     108.762 117.751 153.634  1.00 21.12           C
ATOM    172  CA  THR A1537     104.992 117.146 154.169  1.00 27.76           C
TER
END
"""

def rf(a,b):
  a=abs(a).data()
  b=abs(b).data()
  sc = flex.sum(a*b)/flex.sum(b*b)
  b = b*sc
  return flex.sum(flex.abs(a-b))/flex.sum(a+b)*2*100.

def run():
  p = iotbx.pdb.input(source_info=None, lines=pdb_str)
  h = p.construct_hierarchy()
  x = h.extract_xray_structure(crystal_symmetry=p.crystal_symmetry())
  for d_min in [8,9,10,15,20,25,30,35,40,45,
                50,55,60,65,70,75,80,85,90,95,100,200]:
    t0=time.time()
    fc_f = x.structure_factors(d_min=d_min, algorithm="fft").f_calc()
    t = time.time()-t0
    fc_d = x.structure_factors(d_min=d_min, algorithm="direct").f_calc()
    r = rf(fc_f, fc_d)
    assert r<0.6, r

if (__name__ == "__main__"):
  run()
