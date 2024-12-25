import argparse

import hist
import numpy as np

from combinetf2 import tensorwriter

parser = argparse.ArgumentParser()
parser.add_argument(
    "--sparse",
    default=False,
    action="store_true",
    help="Make sparse tensor",
)
args = parser.parse_args()

# Make histograms
h1_data = hist.Hist(hist.axis.Regular(10, -5, 5, name="x"))

h2_data = hist.Hist(
    hist.axis.Regular(12, 0, 5, name="a"),
    hist.axis.Variable([0, 1, 3, 6, 10, 20], name="b"),
)

h1_sig = h1_data.copy()
h2_sig = h2_data.copy()

h1_bkg = h1_data.copy()
h2_bkg = h2_data.copy()

# Generate random data for filling
np.random.seed(42)  # For reproducibility


def get_sig():
    # gaussian distributed signal
    x = np.random.normal(0, 1, 10000)
    a = np.random.normal(2, 1, 10000)
    b = np.random.normal(10, 2.5, 10000)
    return x, a, b


def get_bkg():
    # uniform distributed background
    x = np.random.uniform(-5, 5, 5000)
    a = np.random.uniform(0, 5, 5000)
    b = np.random.uniform(0, 20, 5000)
    return x, a, b


# Fill histograms
x, a, b = get_sig()
h1_data.fill(x)
h2_data.fill(a, b)

x, a, b = get_bkg()
h1_data.fill(x)
h2_data.fill(a, b)

x, a, b = get_sig()
h1_sig.fill(x)
h2_sig.fill(a, b)
# scale down signal by 50%
h1_sig.values()[...] = h1_sig.values() * 0.5
h2_sig.values()[...] = h2_sig.values() * 0.5

x, a, b = get_bkg()
h1_bkg.fill(x)
h2_bkg.fill(a, b)
# scale up background by 5%
h1_bkg.values()[...] = h1_bkg.values() * 1.05
h2_bkg.values()[...] = h2_bkg.values() * 1.05

# data covariance matrix
variances_flat = np.concatenate(
    [h1_data.values().flatten(), h2_data.values().flatten()]
)
cov = np.diag(variances_flat)
# Add correlations to off-diagonal elements
for i in range(len(variances_flat)):
    for j in range(i + 1, len(variances_flat)):
        # Compute the covariance using the specified correlation
        correlation = np.random.uniform(-0.25, 0.25)  # random correlation
        covariance = correlation * np.sqrt(cov[i, i] * cov[j, j])
        cov[i, j] = covariance
        cov[j, i] = covariance

# add bin by bin stat uncertainty on diagonal elements
cov += np.diag(np.concatenate([h1_sig.values().flatten(), h2_sig.values().flatten()]))
cov += np.diag(np.concatenate([h1_bkg.values().flatten(), h2_bkg.values().flatten()]))

# Build tensor
writer = tensorwriter.TensorWriter(sparse=args.sparse)

writer.add_data(h1_data, "ch0")
writer.add_data(h2_data, "ch1")

writer.add_data_covariance(cov)

writer.add_process(h1_sig, "signal", "ch0", signal=True)
writer.add_process(h2_sig, "signal", "ch1", signal=True)

writer.add_process(h1_bkg, "background", "ch0")
writer.add_process(h2_bkg, "background", "ch1")

# systematic uncertainties

writer.add_lnN_systematic("background_normalization", "background", "ch0", 1.1)
writer.add_lnN_systematic("background_normalization", "background", "ch1", 1.1)

# shape systematics for channel ch0

# Apply reweighting: linear function of axis value
# f(x) = a * x + b
a, b = 0.01, 0.1  # Linear coefficients
bin_centers = h1_bkg.axes[0].centers  # Get bin centers
bin_centers -= bin_centers[0]
weights = a * bin_centers + b  # Compute weights

# Reweight the histogram values
h1_bkg_syst0 = h1_bkg.copy()
h1_bkg_syst0.values()[...] = h1_bkg.values() * (1 + weights)

writer.add_systematic(
    h1_bkg_syst0,
    "slope_background",
    "background",
    "ch0",
    groups=["slopes", "slopes_background"],
)

h1_sig_syst1_up = h1_sig.copy()
h1_sig_syst1_dn = h1_sig.copy()
h1_sig_syst1_up.values()[...] = h1_sig.values() * (1 + weights)
h1_sig_syst1_dn.values()[...] = h1_sig.values() * (1 - weights)

writer.add_systematic(
    [h1_sig_syst1_up, h1_sig_syst1_dn],
    "slope_signal_ch0",
    "signal",
    "ch0",
    groups=["slopes", "slopes_signal"],
    symmetrize="average",
)

h1_sig_syst2_up = h1_sig.copy()
h1_sig_syst2_dn = h1_sig.copy()
h1_sig_syst2_up.values()[...] = h1_sig.values() * (1 + weights) ** 2
h1_sig_syst2_dn.values()[...] = h1_sig.values() * (1 - weights) ** 2

writer.add_systematic(
    [h1_sig_syst2_up, h1_sig_syst2_dn],
    "slope_2_signal_ch0",
    "signal",
    "ch0",
    groups=["slopes", "slopes_signal"],
    symmetrize="linear",
)

# shape systematics for channel ch1

bin_centers = h2_bkg.axes[0].centers  # Get bin centers
bin_centers -= bin_centers[0]
weights = (a * bin_centers + b)[..., None]  # Compute weights

h2_bkg_syst0 = h2_bkg.copy()
h2_bkg_syst0.values()[...] = h2_bkg.values() * (1 + weights)
writer.add_systematic(
    h2_bkg_syst0,
    "slope_background",
    "background",
    "ch1",
    groups=["slopes", "slopes_background"],
)

h2_sig_syst1_up = h2_sig.copy()
h2_sig_syst1_dn = h2_sig.copy()
h2_sig_syst1_up.values()[...] = h2_sig.values() * (1 + weights)
h2_sig_syst1_dn.values()[...] = h2_sig.values() * (1 - weights)

writer.add_systematic(
    [h2_sig_syst1_up, h2_sig_syst1_dn],
    "slope_signal_ch1",
    "signal",
    "ch1",
    groups=["slopes", "slopes_signal"],
    symmetrize="conservative",
)

h2_sig_syst2_up = h2_sig.copy()
h2_sig_syst2_dn = h2_sig.copy()
h2_sig_syst2_up.values()[...] = h2_sig.values() * (1 + weights) ** 2
h2_sig_syst2_dn.values()[...] = h2_sig.values() * (1 - weights) ** 2

writer.add_systematic(
    [h2_sig_syst2_up, h2_sig_syst2_dn],
    "slope_2_signal_ch1",
    "signal",
    "ch1",
    groups=["slopes", "slopes_signal"],
    symmetrize=None,
)


outfilename = "input_tensor"
if args.sparse:
    outfilename += "_sparse"
writer.write(outfolder="test_results", outfilename=outfilename)
