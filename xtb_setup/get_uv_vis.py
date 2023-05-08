import pickle
import pandas as pd
import numpy as np
import subprocess
import sys
import os
from matplotlib import pyplot as plt

def shell(cmd, shell=False):
    if shell:
        p = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        p.wait()
    else:
        cmd = cmd.split()
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        p.wait()

    output, err = p.communicate()
    return output

def compute_absorbance_2(mol, charge, spin, n):
    print(f"{mol}, {charge}, {spin}, {n}")
    print("/home/jelholm/opt/stda/xtb4stda/exe/xtb4stda "+str(mol)+" -chrg "+str(charge)+" -uhf "+str(spin))
    shell("/home/jelholm/opt/xtb4stda-1.0/exe/xtb4stda "+str(mol)+" -chrg "+str(charge)+" -uhf "+str(spin)+" -gbsa toluene",shell=False)
    out = shell('/home/jelholm/opt/stda-1.6.3.1/stda -xtb -e 10',shell=False)
    wl = []
    osc = []
    data = str(out).split('Rv(corr)\\n')[1].split('(')[0]
    wavelength, osc_strength = float(data.split()[2]), float(data.split()[3])
    data2 = str(out).split('Rv(corr)\\n')[1].split('\\n')
    if len(data2) > 100:
        for i in range(0,n): #number of excited states
            wl.append(float(data2[i].split()[2]))
            osc.append(float(data2[i].split()[3]))
            print(float(data2[i].split()[2]), float(data2[i].split()[3]), file=open(f"{mol.split('.')[0]}plot_info.txt", "a"))
    else:
        print(mol, "does not have", n, "excitations, but", int(len(data2)-25), "excitations")
        for i in range(0,len(data2)-25):
            print(float(data2[i].split()[2]), float(data2[i].split()[3]), file=open(f"{mol.split('.')[0]}plot_info.txt", "a"))
    return wl,osc

def lorentzian(x, y, xmin, xmax, xstep, gamma):
    xi = np.arange(xmin,xmax,xstep); yi=np.zeros(len(xi))
    for i in range(len(xi)):
        for k in range(len(x)): yi[i] = yi[i] + y[k] * (gamma/2.) / ( (xi[i]-x[k])**2 + (gamma/2.)**2 )
    return xi,yi



if __name__ == "__main__":
    
    xyz = sys.argv[1]
    evtonm = (4.135667696*10**(-15.0))*299792458*10**(9.0)
    wl,osc = compute_absorbance_2(xyz, 0, 0, 75)
    freq = [evtonm/x for x in wl]
    plot_x,plot_y = lorentzian(freq,osc,min(freq)-1.0,max(freq)+1.0,0.01,0.4)
    plot_x = [evtonm/x for x in plot_x]
    plt.plot(plot_x,plot_y)
    plt.savefig(f"{xyz.split('.')[0]}_plot.pdf")
    os.remove('charges')
    os.remove('tda.dat')
    os.remove('wbo')
    os.remove('wfn.xtb')
    df = pd.DataFrame(plot_x,plot_y)
    df.to_csv(f"{xyz.split('.')[0]}_plot.csv")

