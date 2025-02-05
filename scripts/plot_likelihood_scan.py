import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from narf import ioutils

from combinetf2 import io_tools

plt.rcParams.update({"font.size": 14})


def writeOutput(fig, outfile, extensions=[], postfix=None, args=None, meta_info=None):
    name, _ = os.path.splitext(outfile)

    if postfix:
        name += f"_{postfix}"

    for ext in extensions:
        if ext[0] != ".":
            ext = "." + ext
        output = name + ext
        print(f"Write output file {output}")
        plt.savefig(output)

        output = name.rsplit("/", 1)
        output[1] = os.path.splitext(output[1])[0]
        if len(output) == 1:
            output = (None, *output)
    if args is None and meta_info is None:
        return
    ioutils.write_logfile(
        *output,
        args=args,
        meta_info=meta_info,
    )


def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "inputFile",
        type=str,
        help="fitresults output",
    )
    parser.add_argument(
        "-o",
        "--outpath",
        type=str,
        default="./test",
        help="Folder path for output",
    )
    parser.add_argument(
        "-p", "--postfix", type=str, help="Postfix for output file name"
    )
    parser.add_argument(
        "--params",
        type=str,
        nargs="*",
        default=[],
        help="Parameters to plot the likelihood scan",
    )
    parser.add_argument(
        "--title",
        default="CombineTF2",
        type=str,
        help="Title to be printed in upper left",
    )
    parser.add_argument(
        "--subtitle",
        default=None,
        type=str,
        help="Subtitle to be printed after title",
    )
    return parser.parse_args()


def plot_scan(
    h_scan,
    h_contours=None,
    param_value=0,
    param_variance=1,
    param="x",
    title=None,
    subtitle=None,
):

    x = np.array(h_scan.axes["scan"]).astype(float)
    y = h_scan.values() * 2

    fig, ax = plt.subplots(figsize=(6, 4))
    fig.subplots_adjust(left=0.12, bottom=0.14, right=0.99, top=0.99)

    ax.axhline(y=1, color="gray", linestyle="--", alpha=0.5)
    ax.axhline(y=4, color="gray", linestyle="--", alpha=0.5)

    parabola_vals = param_value + np.linspace(
        -3 * param_variance**0.5, 3 * param_variance**0.5, 100
    )
    parabola_nlls = 1 / param_variance * (parabola_vals - param_value) ** 2
    ax.plot(
        parabola_vals,
        parabola_nlls,
        marker="",
        markerfacecolor="none",
        color="red",
        linestyle="-",
        label="Hessian",
    )

    ax.plot(x, y, marker="x", color="blue", label="Likelihood scan")

    if h_contours is not None:
        for i, cl in enumerate(h_contours.axes["confidence_level"]):
            x = h_contours[{"confidence_level": cl}].values()[::-1] + param_value
            y = np.full(len(x), float(cl) ** 2)
            label = "Contour scan" if i == 0 else None
            ax.plot(
                x,
                y,
                marker="o",
                markerfacecolor="none",
                color="black",
                linestyle="",
                label=label,
            )
            for ix in x:
                ax.axvline(x=ix, color="gray", linestyle="--", alpha=0.5)

    ax.legend(loc="upper right")

    textsize = ax.xaxis.label.get_size()

    if title:
        ax.text(
            0.1,
            0.9,
            title,
            transform=ax.transAxes,
            fontweight="bold",
            fontsize=1.2 * textsize,
        )
    if subtitle:
        ax.text(0.1, 0.82, subtitle, transform=ax.transAxes, fontstyle="italic")

    ax.set_xlabel(param)
    ax.set_ylabel(r"$-2\,\Delta \log L$")

    return fig


if __name__ == "__main__":
    args = parseArgs()
    fitresult, meta = io_tools.get_fitresult(args.inputFile, meta=True)

    meta = {
        "combinetf2": meta["meta_info"],
    }

    h_params = fitresult["parms"].get()

    h_contour = None
    if "contour_scans" in fitresult.keys():
        h_contour = fitresult["contour_scans"].get()

    parms = h_params.axes["parms"] if len(args.params) == 0 else args.params

    for param in parms:
        p = h_params[{"parms": param}]
        param_value = p.value
        param_variance = p.variance
        h_scan = fitresult[f"nll_scan_{param}"].get()

        h_contour_param = None
        if h_contour is not None:
            h_contour_param = h_contour[{"parms": param, "impacts": param}]

        fig = plot_scan(
            h_scan,
            h_contour_param,
            param_value=param_value,
            param_variance=param_variance,
            param=param,
            title=args.title,
            subtitle=args.subtitle,
        )
        os.makedirs(args.outpath, exist_ok=True)
        outfile = os.path.join(args.outpath, f"nll_scan_{param}")
        writeOutput(
            fig,
            outfile=outfile,
            extensions=["png", "pdf"],
            meta_info=meta,
            args=args,
            postfix=args.postfix,
        )
