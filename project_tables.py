"""Manifold projection: enforce tp·dp = tm·dm per (session, sp, imb) bucket.
Reduces calibration noise in sparse buckets. Writes *.proj.imbN.rates files
alongside the regular ones, so onestep can be switched via tables dir.
"""
import os, sys, glob
import numpy as np

TABLES = "data/tables"
SMOOTH = 0.5    # Laplace smoothing for zero counts

def project_one(ses):
    p = f"{TABLES}/{ses:02d}/sp_imb_rates.raw"
    if not os.path.exists(p): return
    data = np.loadtxt(p)
    if data.ndim == 1: data = data.reshape(1, -1)
    # cols: sp imb ntics n a_tp a_tm a_dp a_dm a_r b_tp b_tm b_dp b_dm b_r sum_aN0 sum_bN0 sum_aNd sum_bNd
    d = f"{TABLES}/{ses:02d}"
    # Collect by imb then write
    for im in range(6):
        # Open output files for this imb bin
        files = {
            'tp.a': open(f"{d}/tp.a.imb{im}.rates", "w"),
            'tm.a': open(f"{d}/tm.a.imb{im}.rates", "w"),
            'dp.a': open(f"{d}/dp.a.imb{im}.rates", "w"),
            'dm.a': open(f"{d}/dm.a.imb{im}.rates", "w"),
            'r.a':  open(f"{d}/r.a.imb{im}.rates", "w"),
            'tp.b': open(f"{d}/tp.b.imb{im}.rates", "w"),
            'tm.b': open(f"{d}/tm.b.imb{im}.rates", "w"),
            'dp.b': open(f"{d}/dp.b.imb{im}.rates", "w"),
            'dm.b': open(f"{d}/dm.b.imb{im}.rates", "w"),
            'r.b':  open(f"{d}/r.b.imb{im}.rates", "w"),
            'n':    open(f"{d}/n.imb{im}.rates", "w"),
        }
        for row in data:
            sp, rim = int(row[0]), int(row[1])
            if rim != im: continue
            nt = row[2]
            if nt <= 0: continue
            nn = row[3]
            # Ask side: a_tp, a_tm, a_dp, a_dm, a_r = cols 4, 5, 6, 7, 8
            a_tp, a_tm, a_dp, a_dm, a_r = row[4]+SMOOTH, row[5]+SMOOTH, row[6]+SMOOTH, row[7]+SMOOTH, row[8]
            # Project: tp·dp = tm·dm
            if a_tp*a_dp > 0 and a_tm*a_dm > 0:
                c_a = ((a_tm*a_dm)/(a_tp*a_dp)) ** 0.25
                a_tp *= c_a; a_dp *= c_a
                a_tm /= c_a; a_dm /= c_a
            b_tp, b_tm, b_dp, b_dm, b_r = row[9]+SMOOTH, row[10]+SMOOTH, row[11]+SMOOTH, row[12]+SMOOTH, row[13]
            if b_tp*b_dp > 0 and b_tm*b_dm > 0:
                c_b = ((b_tm*b_dm)/(b_tp*b_dp)) ** 0.25
                b_tp *= c_b; b_dp *= c_b
                b_tm /= c_b; b_dm /= c_b
            # Emit rates
            files['tp.a'].write(f"{sp} {a_tp/nt}\n"); files['tp.b'].write(f"{sp} {b_tp/nt}\n")
            files['tm.a'].write(f"{sp} {a_tm/nt}\n"); files['tm.b'].write(f"{sp} {b_tm/nt}\n")
            files['dp.a'].write(f"{sp} {a_dp/nt}\n"); files['dp.b'].write(f"{sp} {b_dp/nt}\n")
            files['dm.a'].write(f"{sp} {a_dm/nt}\n"); files['dm.b'].write(f"{sp} {b_dm/nt}\n")
            files['r.a'].write(f"{sp} {a_r/nt}\n");   files['r.b'].write(f"{sp} {b_r/nt}\n")
            files['n'].write(f"{sp} {nn/nt}\n")
        for f in files.values(): f.close()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        project_one(int(sys.argv[1]))
    else:
        for ses in range(63):
            project_one(ses)
            if ses % 10 == 0: print(f"done {ses}")
        print("all 63 sessions projected")
