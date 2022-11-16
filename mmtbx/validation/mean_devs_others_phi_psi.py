from __future__ import division
# Phi/Psi corrections in Angstrom and Radials
mean_devs = {
  (-180, -180) : (0.029359, 1.476773), # 25
  (-180, -170) : (0.035818, 1.235930), # 18
  (-180, 130) : (0.045066, 1.160502), # 11
  (-180, 140) : (0.067266, 1.534169), # 24
  (-180, 150) : (0.062265, 1.505901), # 43
  (-180, 160) : (0.049417, 1.541162), # 139
  (-180, 170) : (0.047744, 1.423330), # 181
  (-180, 180) : (0.045340, 1.270865), # 43
  (-170, -180) : (0.013552, 0.663749), # 255
  (-170, -170) : (0.016430, -0.276961), # 153
  (-170, -160) : (0.034168, -0.845927), # 29
  (-170, -150) : (0.045719, -0.742083), # 17
  (-170, 60) : (0.038443, 0.568721), # 15
  (-170, 70) : (0.017132, 0.283521), # 23
  (-170, 80) : (0.017301, 0.252702), # 30
  (-170, 90) : (0.011954, 1.185261), # 49
  (-170, 100) : (0.031514, 0.981755), # 59
  (-170, 110) : (0.038783, 0.943201), # 74
  (-170, 120) : (0.043244, 1.288484), # 109
  (-170, 130) : (0.048511, 1.322186), # 222
  (-170, 140) : (0.050626, 1.251322), # 451
  (-170, 150) : (0.043644, 1.319651), # 915
  (-170, 160) : (0.032810, 1.529320), # 1970
  (-170, 170) : (0.031009, 1.414359), # 1823
  (-170, 180) : (0.017812, 1.073868), # 440
  (-160, -180) : (0.018593, -1.132160), # 645
  (-160, -170) : (0.035792, -1.288926), # 518
  (-160, -160) : (0.041219, -1.113640), # 132
  (-160, -150) : (0.053706, -0.960896), # 41
  (-160, -140) : (0.068756, -1.303776), # 16
  (-160, -80) : (0.055265, -1.695600), # 10
  (-160, -70) : (0.042122, -1.555272), # 13
  (-160, -60) : (0.024494, -2.087194), # 18
  (-160, -40) : (0.064355, -2.318616), # 10
  (-160, -30) : (0.059846, -2.471725), # 16
  (-160, -20) : (0.035414, 2.841441), # 13
  (-160, -10) : (0.027161, 1.602463), # 17
  (-160, 0) : (0.025357, -2.522811), # 17
  (-160, 10) : (0.013473, -2.103185), # 23
  (-160, 20) : (0.021422, -1.934568), # 33
  (-160, 30) : (0.019913, -1.658571), # 28
  (-160, 40) : (0.031633, -1.478178), # 39
  (-160, 50) : (0.022385, -0.880022), # 57
  (-160, 60) : (0.020230, -0.654770), # 108
  (-160, 70) : (0.018573, -0.126868), # 187
  (-160, 80) : (0.020687, 0.214319), # 198
  (-160, 90) : (0.010017, 0.477550), # 225
  (-160, 100) : (0.016380, 0.670242), # 340
  (-160, 110) : (0.028885, 0.822513), # 406
  (-160, 120) : (0.037765, 0.949206), # 652
  (-160, 130) : (0.041760, 1.125116), # 1338
  (-160, 140) : (0.040232, 1.136394), # 2695
  (-160, 150) : (0.022044, 1.345638), # 4224
  (-160, 160) : (0.011000, 1.912644), # 7290
  (-160, 170) : (0.007784, 1.364515), # 4916
  (-160, 180) : (0.008390, -0.620457), # 1074
  (-150, -180) : (0.030401, -1.347161), # 534
  (-150, -170) : (0.049183, -1.437106), # 578
  (-150, -160) : (0.058121, -1.388857), # 183
  (-150, -150) : (0.063012, -1.257634), # 54
  (-150, -140) : (0.079264, -1.237406), # 22
  (-150, -130) : (0.090213, -1.357946), # 15
  (-150, -120) : (0.059541, -1.275802), # 12
  (-150, -100) : (0.082962, -1.697643), # 17
  (-150, -90) : (0.070645, -1.417964), # 13
  (-150, -80) : (0.062311, -1.521939), # 25
  (-150, -70) : (0.048739, -1.516001), # 43
  (-150, -60) : (0.030051, -1.833990), # 44
  (-150, -50) : (0.026509, -1.964720), # 20
  (-150, -40) : (0.027501, -2.319423), # 23
  (-150, -30) : (0.069941, -2.167868), # 36
  (-150, -20) : (0.048803, -2.372901), # 63
  (-150, -10) : (0.050697, -2.213126), # 63
  (-150, 0) : (0.039934, -2.002779), # 84
  (-150, 10) : (0.035736, -1.885229), # 148
  (-150, 20) : (0.042778, -1.814675), # 164
  (-150, 30) : (0.035853, -1.725718), # 137
  (-150, 40) : (0.033221, -1.548638), # 153
  (-150, 50) : (0.023262, -1.282154), # 178
  (-150, 60) : (0.019388, -0.808829), # 310
  (-150, 70) : (0.018503, -0.349989), # 455
  (-150, 80) : (0.017682, -0.090983), # 461
  (-150, 90) : (0.012165, -0.001844), # 465
  (-150, 100) : (0.021599, 0.499922), # 661
  (-150, 110) : (0.031978, 0.688204), # 1031
  (-150, 120) : (0.034048, 0.882210), # 1597
  (-150, 130) : (0.034397, 1.083682), # 3062
  (-150, 140) : (0.034556, 1.180010), # 4729
  (-150, 150) : (0.015386, 1.672921), # 6810
  (-150, 160) : (0.007453, 2.922074), # 8879
  (-150, 170) : (0.004453, -1.594025), # 4449
  (-150, 180) : (0.018273, -1.357383), # 939
  (-140, -180) : (0.026733, -1.344778), # 336
  (-140, -170) : (0.049192, -1.408728), # 382
  (-140, -160) : (0.070258, -1.365403), # 152
  (-140, -150) : (0.081336, -1.320733), # 52
  (-140, -140) : (0.060939, -0.973270), # 33
  (-140, -130) : (0.063330, -1.135906), # 31
  (-140, -120) : (0.092688, -1.511143), # 23
  (-140, -110) : (0.073118, -1.384692), # 35
  (-140, -100) : (0.079993, -1.493600), # 27
  (-140, -90) : (0.081257, -1.625083), # 26
  (-140, -80) : (0.071074, -1.484381), # 43
  (-140, -70) : (0.058939, -1.569096), # 61
  (-140, -60) : (0.032011, -1.683671), # 75
  (-140, -50) : (0.011902, -2.249436), # 71
  (-140, -40) : (0.020503, -2.312424), # 67
  (-140, -30) : (0.031974, -2.533448), # 104
  (-140, -20) : (0.043630, -2.325594), # 140
  (-140, -10) : (0.039559, -2.208457), # 150
  (-140, 0) : (0.038258, -2.135567), # 300
  (-140, 10) : (0.037241, -1.945830), # 450
  (-140, 20) : (0.035165, -1.845564), # 481
  (-140, 30) : (0.028458, -1.719100), # 420
  (-140, 40) : (0.025312, -1.415504), # 356
  (-140, 50) : (0.015159, -0.868033), # 357
  (-140, 60) : (0.014568, -0.458121), # 661
  (-140, 70) : (0.017779, -0.177935), # 785
  (-140, 80) : (0.018740, 0.155126), # 766
  (-140, 90) : (0.017015, 0.275624), # 722
  (-140, 100) : (0.022604, 0.521698), # 951
  (-140, 110) : (0.029876, 0.712264), # 1542
  (-140, 120) : (0.033560, 0.978111), # 2956
  (-140, 130) : (0.038610, 1.194038), # 5232
  (-140, 140) : (0.040762, 1.401885), # 7133
  (-140, 150) : (0.032926, 1.676519), # 8926
  (-140, 160) : (0.016194, 1.926364), # 7792
  (-140, 170) : (0.005327, 1.183707), # 2984
  (-140, 180) : (0.012322, -1.147036), # 597
  (-130, -180) : (0.027144, -1.188930), # 290
  (-130, -170) : (0.046168, -1.395956), # 297
  (-130, -160) : (0.057829, -1.254000), # 147
  (-130, -150) : (0.068939, -1.248611), # 64
  (-130, -140) : (0.080096, -1.288661), # 44
  (-130, -130) : (0.086934, -1.337816), # 32
  (-130, -120) : (0.077344, -1.386542), # 30
  (-130, -110) : (0.070724, -1.377859), # 45
  (-130, -100) : (0.102420, -1.534836), # 23
  (-130, -90) : (0.065018, -1.564328), # 50
  (-130, -80) : (0.043701, -1.289497), # 72
  (-130, -70) : (0.051782, -1.454851), # 65
  (-130, -60) : (0.032680, -1.765848), # 172
  (-130, -50) : (0.024601, -2.041449), # 169
  (-130, -40) : (0.013127, -3.005401), # 221
  (-130, -30) : (0.017654, -2.838203), # 279
  (-130, -20) : (0.028297, -2.530074), # 421
  (-130, -10) : (0.027097, -2.404278), # 716
  (-130, 0) : (0.033617, -2.103720), # 860
  (-130, 10) : (0.032675, -1.904949), # 1001
  (-130, 20) : (0.034274, -1.811790), # 1124
  (-130, 30) : (0.029780, -1.579367), # 965
  (-130, 40) : (0.026018, -1.465237), # 717
  (-130, 50) : (0.017650, -1.190658), # 551
  (-130, 60) : (0.013249, -0.441372), # 1011
  (-130, 70) : (0.015969, -0.060612), # 1468
  (-130, 80) : (0.020676, 0.289201), # 1231
  (-130, 90) : (0.023671, 0.495791), # 1026
  (-130, 100) : (0.026625, 0.518105), # 1422
  (-130, 110) : (0.033398, 0.785376), # 2571
  (-130, 120) : (0.035998, 1.045762), # 4892
  (-130, 130) : (0.041093, 1.232892), # 7748
  (-130, 140) : (0.043593, 1.444585), # 9395
  (-130, 150) : (0.040187, 1.587967), # 9780
  (-130, 160) : (0.028391, 1.590165), # 6174
  (-130, 170) : (0.016528, 1.174480), # 2144
  (-130, 180) : (0.013131, -0.608203), # 457
  (-120, -180) : (0.025568, -0.992685), # 260
  (-120, -170) : (0.041080, -1.272759), # 268
  (-120, -160) : (0.059554, -1.189238), # 145
  (-120, -150) : (0.060360, -1.082290), # 73
  (-120, -140) : (0.075450, -1.318204), # 47
  (-120, -130) : (0.086287, -1.434168), # 44
  (-120, -120) : (0.091843, -1.398153), # 41
  (-120, -110) : (0.086805, -1.372080), # 53
  (-120, -100) : (0.085118, -1.345392), # 46
  (-120, -90) : (0.090895, -1.619430), # 64
  (-120, -80) : (0.059439, -1.291628), # 49
  (-120, -70) : (0.049880, -1.622469), # 126
  (-120, -60) : (0.037692, -1.683699), # 267
  (-120, -50) : (0.014831, -1.938358), # 354
  (-120, -40) : (0.009740, -2.912930), # 468
  (-120, -30) : (0.014668, -2.927558), # 794
  (-120, -20) : (0.018980, -2.630340), # 1143
  (-120, -10) : (0.022722, -2.307778), # 1553
  (-120, 0) : (0.026977, -2.015164), # 1780
  (-120, 10) : (0.028222, -1.797231), # 2121
  (-120, 20) : (0.032517, -1.655163), # 2247
  (-120, 30) : (0.031270, -1.604729), # 1374
  (-120, 40) : (0.031585, -1.401299), # 635
  (-120, 50) : (0.018368, -1.308172), # 275
  (-120, 60) : (0.016673, -0.912092), # 307
  (-120, 70) : (0.015642, 0.152469), # 509
  (-120, 80) : (0.024415, 0.386441), # 710
  (-120, 90) : (0.025986, 0.469195), # 1207
  (-120, 100) : (0.031755, 0.562014), # 2089
  (-120, 110) : (0.036264, 0.766240), # 3651
  (-120, 120) : (0.039157, 1.041287), # 6738
  (-120, 130) : (0.042759, 1.246071), # 9797
  (-120, 140) : (0.045733, 1.424395), # 10024
  (-120, 150) : (0.042513, 1.528635), # 7969
  (-120, 160) : (0.034398, 1.440162), # 4440
  (-120, 170) : (0.020347, 1.039837), # 1598
  (-120, 180) : (0.014157, -0.534090), # 380
  (-110, -180) : (0.026520, -1.189893), # 254
  (-110, -170) : (0.042209, -1.264611), # 240
  (-110, -160) : (0.065581, -1.318008), # 115
  (-110, -150) : (0.086132, -1.321534), # 62
  (-110, -140) : (0.069599, -1.228096), # 65
  (-110, -130) : (0.098566, -1.360986), # 41
  (-110, -120) : (0.102567, -1.354365), # 42
  (-110, -110) : (0.093524, -1.491954), # 47
  (-110, -100) : (0.088469, -1.466373), # 53
  (-110, -90) : (0.057870, -1.354444), # 60
  (-110, -80) : (0.059834, -1.566256), # 57
  (-110, -70) : (0.045144, -1.461193), # 123
  (-110, -60) : (0.026064, -1.736404), # 271
  (-110, -50) : (0.013649, -2.000743), # 475
  (-110, -40) : (0.007622, 2.830772), # 885
  (-110, -30) : (0.012127, 3.109683), # 1401
  (-110, -20) : (0.014492, -2.809614), # 1956
  (-110, -10) : (0.016365, -2.356916), # 2564
  (-110, 0) : (0.020366, -2.021124), # 3331
  (-110, 10) : (0.028734, -1.727880), # 4472
  (-110, 20) : (0.035262, -1.689068), # 3611
  (-110, 30) : (0.036958, -1.609932), # 1441
  (-110, 40) : (0.034301, -1.502252), # 369
  (-110, 50) : (0.030110, -1.367023), # 131
  (-110, 60) : (0.027642, -0.864299), # 91
  (-110, 70) : (0.022644, -0.146205), # 115
  (-110, 80) : (0.025724, 0.117881), # 249
  (-110, 90) : (0.027829, 0.326249), # 892
  (-110, 100) : (0.033345, 0.553325), # 2462
  (-110, 110) : (0.037742, 0.752580), # 5044
  (-110, 120) : (0.039476, 1.026366), # 8211
  (-110, 130) : (0.042773, 1.240424), # 10441
  (-110, 140) : (0.045349, 1.391509), # 9073
  (-110, 150) : (0.041169, 1.484279), # 6303
  (-110, 160) : (0.031592, 1.360033), # 3353
  (-110, 170) : (0.017299, 0.789013), # 1403
  (-110, 180) : (0.015043, -0.688675), # 387
  (-100, -180) : (0.046188, -1.502687), # 318
  (-100, -170) : (0.049597, -1.387920), # 292
  (-100, -160) : (0.062849, -1.321211), # 128
  (-100, -150) : (0.066216, -1.200503), # 70
  (-100, -140) : (0.082153, -1.190036), # 46
  (-100, -130) : (0.097580, -1.357186), # 45
  (-100, -120) : (0.086525, -1.389480), # 39
  (-100, -110) : (0.085578, -1.370908), # 39
  (-100, -100) : (0.087615, -1.398867), # 41
  (-100, -90) : (0.064727, -1.408143), # 44
  (-100, -80) : (0.060746, -1.438851), # 78
  (-100, -70) : (0.043812, -1.643059), # 124
  (-100, -60) : (0.022974, -1.701794), # 299
  (-100, -50) : (0.007590, -1.896365), # 792
  (-100, -40) : (0.006287, 2.937790), # 1557
  (-100, -30) : (0.009495, 2.811395), # 2265
  (-100, -20) : (0.012223, -2.644722), # 2925
  (-100, -10) : (0.013269, -2.456152), # 4190
  (-100, 0) : (0.018267, -2.006644), # 7002
  (-100, 10) : (0.031949, -1.773938), # 7483
  (-100, 20) : (0.040416, -1.735029), # 3231
  (-100, 30) : (0.045210, -1.681464), # 838
  (-100, 40) : (0.031181, -1.501510), # 245
  (-100, 50) : (0.026895, -1.117853), # 145
  (-100, 60) : (0.035489, -1.081908), # 211
  (-100, 70) : (0.031513, -0.783464), # 277
  (-100, 80) : (0.029310, -0.527512), # 411
  (-100, 90) : (0.030931, 0.113822), # 943
  (-100, 100) : (0.032657, 0.448904), # 2595
  (-100, 110) : (0.036250, 0.714378), # 5490
  (-100, 120) : (0.037672, 1.010882), # 7884
  (-100, 130) : (0.041623, 1.218791), # 8938
  (-100, 140) : (0.041418, 1.373632), # 7428
  (-100, 150) : (0.036679, 1.416140), # 5099
  (-100, 160) : (0.026858, 1.255872), # 3052
  (-100, 170) : (0.017827, 0.724410), # 1586
  (-100, 180) : (0.025678, -1.170661), # 437
  (-90, -180) : (0.061451, -1.648075), # 453
  (-90, -170) : (0.062347, -1.541634), # 462
  (-90, -160) : (0.066322, -1.296499), # 109
  (-90, -150) : (0.094409, -1.351361), # 48
  (-90, -140) : (0.068399, -0.834079), # 20
  (-90, -130) : (0.064351, -1.014566), # 19
  (-90, -120) : (0.058942, -1.190495), # 13
  (-90, -110) : (0.086245, -1.183197), # 13
  (-90, -100) : (0.058817, -1.210231), # 15
  (-90, -90) : (0.067824, -1.457738), # 38
  (-90, -80) : (0.051174, -1.289209), # 64
  (-90, -70) : (0.053574, -1.574032), # 99
  (-90, -60) : (0.024561, -1.616899), # 371
  (-90, -50) : (0.004678, -1.691570), # 1162
  (-90, -40) : (0.005125, -2.603259), # 2445
  (-90, -30) : (0.007203, -3.027935), # 3600
  (-90, -20) : (0.011843, -2.676921), # 4654
  (-90, -10) : (0.011418, -2.587192), # 8298
  (-90, 0) : (0.023666, -2.011294), # 10623
  (-90, 10) : (0.038532, -1.832239), # 5194
  (-90, 20) : (0.049092, -1.815058), # 993
  (-90, 30) : (0.040662, -1.670169), # 220
  (-90, 40) : (0.048847, -1.654950), # 181
  (-90, 50) : (0.038431, -1.479638), # 389
  (-90, 60) : (0.036788, -1.241824), # 1025
  (-90, 70) : (0.032461, -0.994181), # 1644
  (-90, 80) : (0.026324, -0.689133), # 1468
  (-90, 90) : (0.024148, -0.040341), # 1621
  (-90, 100) : (0.027793, 0.350432), # 2860
  (-90, 110) : (0.034967, 0.628860), # 5151
  (-90, 120) : (0.035107, 0.946896), # 6716
  (-90, 130) : (0.037164, 1.183564), # 7201
  (-90, 140) : (0.037288, 1.333551), # 6335
  (-90, 150) : (0.030190, 1.399885), # 4971
  (-90, 160) : (0.021721, 1.180289), # 3435
  (-90, 170) : (0.012613, 0.433872), # 2042
  (-90, 180) : (0.047994, -1.569113), # 578
  (-80, -180) : (0.073557, -1.661085), # 572
  (-80, -170) : (0.074117, -1.579096), # 413
  (-80, -160) : (0.049065, -1.159707), # 41
  (-80, -150) : (0.063982, -1.138736), # 13
  (-80, -80) : (0.047572, -1.194599), # 26
  (-80, -70) : (0.050701, -1.500202), # 79
  (-80, -60) : (0.022857, -1.550404), # 410
  (-80, -50) : (0.009973, -1.724942), # 2311
  (-80, -40) : (0.012995, -1.860143), # 6355
  (-80, -30) : (0.024040, -1.968582), # 8321
  (-80, -20) : (0.018019, -2.367014), # 9133
  (-80, -10) : (0.019980, -2.242025), # 12314
  (-80, 0) : (0.041239, -1.952565), # 7069
  (-80, 10) : (0.064735, -1.887881), # 1259
  (-80, 20) : (0.070443, -1.849384), # 110
  (-80, 30) : (0.031641, -1.892952), # 38
  (-80, 40) : (0.043587, -1.669022), # 65
  (-80, 50) : (0.032810, -1.530482), # 214
  (-80, 60) : (0.033228, -1.281523), # 764
  (-80, 70) : (0.029568, -1.122369), # 1428
  (-80, 80) : (0.022114, -0.766257), # 1450
  (-80, 90) : (0.018966, -0.103329), # 1258
  (-80, 100) : (0.025474, 0.352597), # 1779
  (-80, 110) : (0.033083, 0.625727), # 3396
  (-80, 120) : (0.033502, 0.870713), # 5433
  (-80, 130) : (0.031441, 1.143835), # 7241
  (-80, 140) : (0.032546, 1.342653), # 7609
  (-80, 150) : (0.026688, 1.375591), # 6860
  (-80, 160) : (0.020509, 1.203033), # 5318
  (-80, 170) : (0.010455, 0.374881), # 2937
  (-80, 180) : (0.046093, -1.595636), # 794
  (-70, -180) : (0.041928, -1.351027), # 141
  (-70, -170) : (0.033867, -1.300923), # 38
  (-70, -70) : (0.063628, -1.508406), # 44
  (-70, -60) : (0.029449, -1.706806), # 690
  (-70, -50) : (0.010334, -1.612509), # 15438
  (-70, -40) : (0.020037, -1.825406), # 75331
  (-70, -30) : (0.035730, -1.944946), # 33685
  (-70, -20) : (0.031032, -2.126191), # 20912
  (-70, -10) : (0.041844, -2.027345), # 11427
  (-70, 0) : (0.072279, -1.952250), # 1573
  (-70, 10) : (0.086122, -1.902999), # 59
  (-70, 50) : (0.031471, -1.561751), # 12
  (-70, 60) : (0.038597, -1.831048), # 22
  (-70, 70) : (0.029886, -1.166584), # 40
  (-70, 80) : (0.018822, -0.283415), # 71
  (-70, 90) : (0.021960, -0.019937), # 117
  (-70, 100) : (0.024553, 0.226328), # 294
  (-70, 110) : (0.031866, 0.448648), # 1232
  (-70, 120) : (0.030067, 0.770202), # 4001
  (-70, 130) : (0.026996, 1.128070), # 8396
  (-70, 140) : (0.027894, 1.335239), # 11207
  (-70, 150) : (0.023622, 1.404602), # 10454
  (-70, 160) : (0.021387, 1.257077), # 7016
  (-70, 170) : (0.014532, 0.762253), # 2503
  (-70, 180) : (0.020579, -1.197229), # 325
  (-60, -70) : (0.046026, -1.354239), # 23
  (-60, -60) : (0.024393, -1.686354), # 1319
  (-60, -50) : (0.005510, -1.529582), # 70535
  (-60, -40) : (0.014210, -1.995169), # 145956
  (-60, -30) : (0.024544, -2.131879), # 36188
  (-60, -20) : (0.030012, -2.217526), # 13772
  (-60, -10) : (0.043608, -2.078433), # 1296
  (-60, 0) : (0.071569, -2.061247), # 16
  (-60, 100) : (0.027563, -0.332715), # 11
  (-60, 110) : (0.029168, 0.298416), # 262
  (-60, 120) : (0.026701, 0.667432), # 2529
  (-60, 130) : (0.023504, 1.077095), # 8987
  (-60, 140) : (0.027404, 1.306610), # 12416
  (-60, 150) : (0.025092, 1.386904), # 7691
  (-60, 160) : (0.025178, 1.231195), # 2504
  (-60, 170) : (0.029920, 1.066913), # 317
  (-60, 180) : (0.021211, -1.648840), # 11
  (-50, -60) : (0.028839, -1.826821), # 674
  (-50, -50) : (0.010333, -2.013920), # 12819
  (-50, -40) : (0.015855, -2.148771), # 11461
  (-50, -30) : (0.018246, -2.380122), # 2947
  (-50, -20) : (0.027956, -2.327172), # 247
  (-50, 110) : (0.034032, 0.567008), # 69
  (-50, 120) : (0.028472, 0.611328), # 1079
  (-50, 130) : (0.026429, 1.014829), # 4589
  (-50, 140) : (0.029972, 1.218796), # 3531
  (-50, 150) : (0.037065, 1.292033), # 844
  (-50, 160) : (0.033157, 1.103617), # 92
  (-40, -70) : (0.037107, -0.679333), # 11
  (-40, -60) : (0.034581, -1.812735), # 139
  (-40, -50) : (0.028825, -1.920726), # 526
  (-40, -40) : (0.011173, -1.932204), # 213
  (-40, -30) : (0.008566, 3.095065), # 17
  (-40, 110) : (0.026658, 0.040784), # 32
  (-40, 120) : (0.033268, 0.691241), # 231
  (-40, 130) : (0.032042, 0.903993), # 434
  (-40, 140) : (0.048011, 1.198287), # 114
  (-40, 150) : (0.022540, 1.853840), # 11
  (-30, -60) : (0.043089, -2.280790), # 18
  (-30, -50) : (0.029539, -0.948622), # 10
  (-30, 110) : (0.022871, 0.551573), # 17
  (-30, 120) : (0.058660, 0.964802), # 33
  (-30, 130) : (0.051351, 0.857448), # 19
  (-20, 110) : (0.021884, -0.888302), # 13
  (30, 50) : (0.087046, -1.950943), # 12
  (30, 60) : (0.064968, -1.893247), # 47
  (30, 70) : (0.050543, -1.826870), # 33
  (40, -130) : (0.116805, -1.573698), # 39
  (40, -120) : (0.103515, -1.571604), # 46
  (40, -110) : (0.104618, -1.765322), # 14
  (40, 40) : (0.070075, -1.892134), # 129
  (40, 50) : (0.048435, -1.761842), # 524
  (40, 60) : (0.048351, -1.765530), # 465
  (40, 70) : (0.049892, -1.854185), # 65
  (50, -150) : (0.092632, -1.454294), # 30
  (50, -140) : (0.084238, -1.428013), # 169
  (50, -130) : (0.092903, -1.454323), # 359
  (50, -120) : (0.094209, -1.505790), # 275
  (50, -110) : (0.095798, -1.578098), # 60
  (50, -100) : (0.117008, -1.625376), # 11
  (50, 20) : (0.050073, -2.115467), # 82
  (50, 30) : (0.054220, -1.963639), # 1008
  (50, 40) : (0.049331, -1.818417), # 4130
  (50, 50) : (0.047627, -1.754331), # 3745
  (50, 60) : (0.047249, -1.753829), # 1251
  (50, 70) : (0.053395, -1.745506), # 132
  (50, 80) : (0.062963, -2.239019), # 17
  (60, -180) : (0.071636, -1.790798), # 10
  (60, -170) : (0.076051, -1.807459), # 36
  (60, -160) : (0.086812, -1.648340), # 74
  (60, -150) : (0.110777, -1.631110), # 92
  (60, -140) : (0.098213, -1.462889), # 102
  (60, -130) : (0.104930, -1.409280), # 159
  (60, -120) : (0.108994, -1.479907), # 167
  (60, -110) : (0.109699, -1.610965), # 59
  (60, -100) : (0.105828, -1.748579), # 21
  (60, 0) : (0.084257, -2.101035), # 25
  (60, 10) : (0.081466, -2.169148), # 398
  (60, 20) : (0.075166, -2.061571), # 1866
  (60, 30) : (0.067373, -1.991395), # 3721
  (60, 40) : (0.059583, -1.894650), # 3735
  (60, 50) : (0.058888, -1.843307), # 1697
  (60, 60) : (0.064435, -1.852643), # 557
  (60, 70) : (0.063465, -1.885295), # 169
  (60, 80) : (0.098783, -1.934860), # 39
  (60, 170) : (0.047206, -1.861841), # 11
  (70, -180) : (0.097652, -1.938965), # 24
  (70, -170) : (0.085374, -1.802051), # 41
  (70, -160) : (0.095100, -1.605208), # 24
  (70, -150) : (0.107550, -1.754276), # 15
  (70, -120) : (0.108318, -1.464899), # 21
  (70, -80) : (0.098926, -2.040947), # 10
  (70, -70) : (0.114760, -2.092220), # 45
  (70, -60) : (0.108090, -2.199770), # 96
  (70, -50) : (0.103206, -2.216593), # 100
  (70, -40) : (0.099284, -2.261739), # 94
  (70, -30) : (0.114308, -2.229911), # 31
  (70, -20) : (0.105514, -2.245892), # 24
  (70, -10) : (0.099843, -2.317367), # 89
  (70, 0) : (0.104636, -2.229997), # 537
  (70, 10) : (0.102481, -2.157693), # 1380
  (70, 20) : (0.096504, -2.078984), # 1289
  (70, 30) : (0.086296, -2.022910), # 834
  (70, 40) : (0.078740, -1.987873), # 279
  (70, 50) : (0.089010, -1.986364), # 82
  (70, 60) : (0.085736, -1.884112), # 65
  (70, 70) : (0.089585, -1.941997), # 60
  (70, 80) : (0.106244, -2.100553), # 41
  (70, 90) : (0.081410, -2.242746), # 26
  (70, 110) : (0.080845, -2.306957), # 22
  (70, 140) : (0.099814, -2.065304), # 10
  (70, 150) : (0.098552, -2.233347), # 12
  (70, 160) : (0.104861, -2.130572), # 53
  (70, 170) : (0.101809, -2.131538), # 50
  (70, 180) : (0.089699, -2.057429), # 25
  (80, -70) : (0.089577, -2.086815), # 13
  (80, -60) : (0.107326, -2.138645), # 35
  (80, -50) : (0.124977, -2.168144), # 50
  (80, -40) : (0.108885, -2.175857), # 35
  (80, -30) : (0.087800, -2.417618), # 17
  (80, -20) : (0.118585, -2.315026), # 44
  (80, -10) : (0.119801, -2.345491), # 185
  (80, 0) : (0.131362, -2.232846), # 375
  (80, 10) : (0.121937, -2.222599), # 231
  (80, 20) : (0.125962, -2.135559), # 96
  (80, 30) : (0.128941, -2.116507), # 36
  (80, 40) : (0.103835, -2.245398), # 14
  (80, 100) : (0.107078, -2.345395), # 15
  (80, 110) : (0.111767, -2.305369), # 30
  (80, 120) : (0.092212, -2.486929), # 18
  (80, 130) : (0.094011, -2.558750), # 11
  (80, 140) : (0.092913, -2.604048), # 19
  (80, 150) : (0.107580, -2.391215), # 22
  (80, 160) : (0.110199, -2.400704), # 46
  (80, 170) : (0.103633, -2.180440), # 32
  (80, 180) : (0.117708, -2.008535), # 14
  (90, -20) : (0.119502, -2.407177), # 21
  (90, -10) : (0.132172, -2.372219), # 40
  (90, 0) : (0.115742, -2.583950), # 25
  (90, 10) : (0.118406, -2.444673), # 13
  (170, 170) : (0.044211, 1.737079), # 14
  (180, 150) : (0.071429, 1.828557), # 10
  (180, 160) : (0.060106, 1.654180), # 43
  (180, 170) : (0.068469, 1.524977), # 40
}

if __name__ == '__main__':
  from mmtbx.validation.mean_devs_others_phi_psi import mean_devs as others_phi_psi
  for phi_psi in [(0,0), (-60,-60)]:
    print(phi_psi, mean_devs.get(phi_psi, None), others_phi_psi.get(phi_psi, None))

