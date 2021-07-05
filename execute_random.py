import argparse
import os
import numpy as np
import pfsp
import csv
import time

if __name__ == '__main__':
    a = pfsp.PFSP()
    jobs = (500,)
    machines = (20,)
    instances = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
    reps = 10
    size = 500
    it = 200

    files = a.generate_list_of_files_taillard(jobs, machines, instances)

    # os.system('./lanzar_pfsp.sh {} {} {} {}'.format(files[0], 0, 1.0, 1.0))
    p = pfsp.PFSP()
    k = 0
    best_result_max = -999999999
    best_known_results_20_5 = [-14033, -15151, -13301, -15447, -13529, -13123, -13548, -13948, -14295, -12943]
    best_known_results_20_10 = [-20911, -22440, -19833, -18710, -18641, -19245, -18363, -20241, -20330, -21320]
    best_known_results_20_20 = [-33623, -31587, -33920, -31661, -34557, -32564, -32922, -32412, -33600, -32262]

    best_known_results_50_5 = [-64803, -68062, -63162, -68226, -69392, -66841, -66258, -64359, -62981, -68898]
    best_known_results_50_10 = [-87207, -82820, -79987, -86581, -86450, -86637, -88866, -86824, -85526, -88077]
    best_known_results_50_20 = [-125831, -119259, -116459, -120712, -118184, -120703, -122962, -122489, -121872,
                                -124064]

    best_known_results_100_5 = [-253713, -242777, -238180, -227889, -240589, -232936, -240669, -231428, -248481,
                                -243360]
    best_known_results_100_10 = [-299431, -274593, -288630, -302105, -285340, -270817, -280649, -291665, -302624,-292230]
    best_known_results_100_20 = [-367267, -374032, -371417, -373822, -370459, -372768, -374483, -385456, -376063,-379899]



    best_known_results_200_10 = [-1047662, -1036042, -1047571, -1032095, -1037053, -1006650, -1053390, -1046246, -1025145,-1031176]
    best_known_results_200_20 = [-1226879, -1241811, -1266153, -1237053, -1223551, -1225254, -1241847, -1240820, -1229066,-1247156]

    best_known_results_500_20 = [-6708053, -6829668, -6747387, -6787054, -6755257, -6751496, -6708860, -6769821, -6720474,-6767645]

    mean = 0

    k = 0
    for file in files:
        best_result_max = -999999999999
        mean = 0
        for i in range(reps):
            best_result = p.random_execution(file=file, size=size, it=it)
            if best_result_max < best_result:
                best_result_max = best_result
            mean += best_result
        arpd = (mean / reps) / best_known_results_500_20[k] - 1
        print(arpd)
        k += 1
