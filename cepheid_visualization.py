from argparse import ArgumentParser
from os import path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as colors

import numpy
from sklearn.linear_model import LinearRegression
from plotypus.lightcurve import get_lightcurve_from_file, make_predictor
from plotypus.utils import make_sure_path_exists

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)


def get_args():
    parser = ArgumentParser(prog="cepheid_visualization")

    parser.add_argument("-i", "--input", type=str,
        help="Table file containing time, magnitude, and (optional) error "
             "in its columns.")
    parser.add_argument("-o", "--output", type=str,
        default=".",
        help="Directory to output demo plots.")
    parser.add_argument("-t", "--type", type=str,
        default="png",
        help="File type to output plots in. Default is png.")
    parser.add_argument("-n", "--name", type=str,
        default="",
        help="Name of star to use as prefix in output files.")
    parser.add_argument("-p", "--period", type=float,
        help="Period to phase observations by.")
    parser.add_argument("-d", "--fourier-degree", type=int, nargs=2,
        default=(2, 10), metavar=("MIN-DEGREE", "MAX-DEGREE"),
        help="Degree of fit")
    parser.add_argument("--use-cols", type=int, nargs="+",
        default=(0, 1, 2),
        help="Columns to read time, magnigude, and (optional) error from, "
             "respectively. "
             "Defaults to 0, 1, 2.")
    parser.add_argument("--radius", type=float, nargs=2,
        default=(0.2, 0.25), metavar=["R-MIN", "R-MAX"],
        help="Boundaries on radius of star visualization "
             "(default 0.2, 0.25)")
    parser.add_argument("--colors", type=str, nargs="+",
        default=["#FFFFFF", "#FFFF55"],
        help="Colors")

    args = parser.parse_args()

    args.prefix = (args.name + "-") if args.name else ""

    return args

def linear_map(x, x_min, x_max, y_min, y_max):
    return (x-x_min)*(y_max-y_min)/(x_max-x_min) + y_min

def display(index, d_radius,
            phases_fitted, mags_fitted,
            phases_observed, mags_observed,
            output, prefix, file_type,
            mag_min, mag_max, mag_mean,
            radius_min=0.2, radius_max=0.25,
            color_map=plt.get_cmap("Blues")):
    fig, axes = plt.subplots(1, 2)
    lc_axis, star_axis = axes

    lc_axis.invert_yaxis()

    star_axis.set_aspect("equal")
    star_axis.set_axis_bgcolor("black")
    star_axis.xaxis.set_visible(False)
    star_axis.yaxis.set_visible(False)

    phase, mag = phases_fitted[index], mags_fitted[index]
    rad = linear_map(d_radius,
                     mag_mean, -mag_mean,
                     radius_min, radius_max)
    mag_norm = (mag - mag_min) / (mag_max - mag_min)
    star = plt.Circle((0.5, 0.5), rad, color=color_map(mag_norm))

    lc_axis.scatter(phases_observed, mags_observed, color="k", s=0.1)
    lc_axis.plot(phases_fitted, mags_fitted, color="g")
    lc_axis.axvline(x=phase, linewidth=1, color="r")
    lc_axis.set_xlabel(r"Time (days)")
    lc_axis.set_ylabel(r"Magnitude")
    lc_axis.set_title("Light Curve")
    lc_axis.set_xlim(0, phases_fitted[-1])

    star_axis.add_artist(star)
    star_axis.set_title("Simulated Star")

    fig.savefig(path.join(output,
                          "{0}simulation-{1:02d}.{2}".format(prefix,
                                                             index,
                                                             file_type)))
    plt.close(fig)

def main():
    args = get_args()

    make_sure_path_exists(args.output)

    color_map = colors.LinearSegmentedColormap.from_list("StarColors",
                                                         args.colors)

    predictor = make_predictor(
        regressor=LinearRegression(fit_intercept=False),
        fourier_degree=args.fourier_degree,
        use_baart=True)

    phases = numpy.arange(0, 1, 0.01)

    result = get_lightcurve_from_file(args.input, period=args.period,
                                      n_phases=100,
                                      predictor=predictor,
                                      sigma=numpy.PINF)

    lightcurve = result["lightcurve"]
    phased_data = result["phased_data"]

    phase_observed, mag_observed, *err = phased_data.T

    mag_min, mag_max = lightcurve.min(), lightcurve.max()
    mag_mean = lightcurve.mean()

    d_radius = 0

    for i in range(100):
        d_radius += lightcurve[i] - mag_mean
        display(i, d_radius,
                phases*args.period, lightcurve,
                phase_observed*args.period, mag_observed,
                args.output, args.prefix, args.type,
                mag_min, mag_max, mag_mean,
                radius_min=args.radius[0], radius_max=args.radius[1],
                color_map=color_map)

    return 0

if __name__ == "__main__":
    exit(main())
