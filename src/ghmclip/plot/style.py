import matplotlib.pyplot as plt

def my_legend(axe=None,loc=0):
    if axe is None:
        axe = plt.gca()
    legend = axe.legend(loc=loc)
    legend.get_frame().set_linewidth(legend_linewidth)
    legend.get_frame().set_edgecolor('black')


plt.rcParams['axes.linewidth'] = 1
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
legend_linewidth = 1