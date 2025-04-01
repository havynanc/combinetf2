#!/usr/bin/env python
import argparse
import os
import sys

import hist
import matplotlib.pyplot as plt
import numpy as np

from combinetf2 import debugdata, inputdata


def debug_input_data(input_file, output_dir=None, verbose=False):
    """
    Debug input data file and report potential issues

    Parameters:
    -----------
    input_file : str
        Path to input data file
    output_dir : str, optional
        Directory to save diagnostic plots
    verbose : bool
        Whether to print detailed information
    """
    print(f"Debugging input file: {input_file}")

    # Load the input data
    indata = inputdata.FitInputData(input_file)

    try:
        debug_data = debugdata.FitDebugData(indata)
        print("✓ Successfully created FitDebugData object")
    except Exception as e:
        print(f"✗ Failed to create FitDebugData: {str(e)}")
        return

    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Check for issues
    issues_found = 0

    # 1. Check for channels with no data observations
    if verbose:
        print("\nChecking channels with data observations:")
    channels_with_data = list(debug_data.data_obs_hists.keys())
    channels_without_data = [
        ch
        for ch in debug_data.indata.channel_info.keys()
        if ch not in channels_with_data
    ]

    if channels_without_data:
        issues_found += 1
        print(
            f"✗ Found {len(channels_without_data)} channels without data observations:"
        )
        for ch in channels_without_data:
            print(f"  - {ch}")
    else:
        print("✓ All channels have data observations")

    # 2. Check for empty bins in data
    empty_data_channels = []
    for channel, hist_obj in debug_data.data_obs_hists.items():
        if np.sum(hist_obj.values()) == 0:
            empty_data_channels.append(channel)

    if empty_data_channels:
        issues_found += 1
        print(
            f"✗ Found {len(empty_data_channels)} channels with empty data observations:"
        )
        for ch in empty_data_channels:
            print(f"  - {ch}")
    else:
        print("✓ All channels have non-empty data observations")

    # 3. Check for processes with zero normalization
    zero_norm_procs = {}
    for channel, hist_obj in debug_data.nominal_hists.items():
        proc_sums = np.sum(hist_obj.values(), axis=tuple(range(hist_obj.ndim - 1)))
        zero_procs = [
            debug_data.indata.procs[i] for i, val in enumerate(proc_sums) if val == 0
        ]
        if zero_procs:
            zero_norm_procs[channel] = zero_procs

    if zero_norm_procs:
        issues_found += 1
        print("✗ Found processes with zero normalization:")
        for channel, procs in zero_norm_procs.items():
            print(f"  - Channel {channel}: {', '.join(np.array(procs, dtype=str))}")
    else:
        print("✓ All processes have non-zero normalization")

    # 4. Check for systematics with only zeros
    all_systs = list(debug_data.axis_systs)
    nonzero_systs = debug_data.nonzeroSysts()
    zero_systs = [syst for syst in all_systs if syst not in nonzero_systs]

    if zero_systs:
        issues_found += 1
        print(f"✗ Found {len(zero_systs)} systematics with only zeros:")
        for syst in zero_systs:
            print(f"  - {syst}")
    else:
        print("✓ All systematics are nonzero")

    # 5. Check for processes with no systematic variations
    procs_without_systs = []
    for proc in debug_data.indata.procs:
        channels_with_syst = debug_data.channelsForNonzeroSysts(
            procs=[proc.decode("utf-8")]
        )
        if not channels_with_syst:
            procs_without_systs.append(proc)

    if procs_without_systs:
        issues_found += 1
        print(
            f"✗ Found {len(procs_without_systs)} processes with no systematic variations:"
        )
        for proc in procs_without_systs:
            print(f"  - {proc}")
    else:
        print("✓ All processes have systematic variations")

    # 6. Check for extreme systematic variations
    extreme_variations = {}
    threshold = 2.0  # Variation more than 100% up or down

    for channel, syst_hist in debug_data.syst_hists.items():
        nom_hist = debug_data.nominal_hists[channel]
        # Calculate ratio of systematic variation to nominal
        down_values = syst_hist[{"DownUp": "Down"}].values()
        up_values = syst_hist[{"DownUp": "Up"}].values()
        nom_values = nom_hist.values()[..., None]

        # Avoid division by zero
        with np.errstate(divide="ignore", invalid="ignore"):
            down_ratio = np.where(nom_values > 0, down_values / nom_values, np.nan)
            up_ratio = np.where(nom_values > 0, up_values / nom_values, np.nan)

        # Find extreme variations
        for isyst, syst in enumerate(debug_data.indata.systs):
            for iproc, proc in enumerate(debug_data.indata.procs):
                down_extreme = np.nanmin(down_ratio[..., iproc, isyst])
                up_extreme = np.nanmax(up_ratio[..., iproc, isyst])

                if down_extreme < 1 / threshold or up_extreme > threshold:
                    if channel not in extreme_variations:
                        extreme_variations[channel] = []
                    extreme_variations[channel].append(
                        (proc, syst, down_extreme, up_extreme)
                    )

    if extreme_variations:
        issues_found += 1
        print("✗ Found extreme systematic variations (>100%):")
        for channel, variations in extreme_variations.items():
            print(f"  Channel: {channel}")
            for proc, syst, down, up in variations:
                print(f"    - Process: {proc}, Systematic: {syst}")
                print(f"      Down: {down:.2f}x, Up: {up:.2f}x")
    else:
        print("✓ No extreme systematic variations found")

    # 7. Generate diagnostic plots if output_dir is specified
    if output_dir:
        # try:
        for channel, nom_hist in debug_data.nominal_hists.items():
            # Plot nominal distributions
            fig, ax = plt.subplots(figsize=(10, 6))

            # Get the first two axes for plotting (assuming they're binned)
            axes = debug_data.indata.channel_info[channel]["axes"]
            if len(axes) >= 1:
                x_axis = axes[0]
                y_axis = axes[1] if len(axes) >= 2 else None
                x_bins = len(x_axis)
                y_bins = 0

                # Sum over all processes
                summed_values = np.sum(nom_hist.values(), axis=-1)
                data_values = debug_data.data_obs_hists[channel].values()

                # If 2D histogram
                if x_bins > 1 and y_bins > 1:
                    im = ax.imshow(
                        summed_values.T,
                        origin="lower",
                        aspect="auto",
                        extent=[0, x_bins, 0, y_bins],
                    )
                    plt.colorbar(im, ax=ax, label="Events")
                    ax.set_xlabel(x_axis.name)
                    ax.set_ylabel(y_axis.name)
                    ax.set_title(f"Channel: {channel} - Nominal")
                else:  # 1D histogram
                    ax.bar(range(len(summed_values)), summed_values, label="prediction")
                    ax.errorbar(
                        range(len(summed_values)),
                        data_values,
                        yerr=np.sqrt(data_values),
                        color="black",
                        marker=".",
                        linestyle="None",
                        label="Data",
                    )
                    ax.set_xlabel(f"{x_axis.name} bin")
                    ax.set_ylabel("Events")
                    ax.set_title(f"Channel: {channel} - Nominal")

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{channel}_nominal.png"))
            plt.close()

            # Plot systematics impact
            if channel in debug_data.syst_hists:
                fig, ax = plt.subplots(figsize=(10, 6))

                # For each systematic, compute the total relative impact
                syst_impact = []
                syst_impact_down = []
                syst_names = []

                for isyst, syst in enumerate(debug_data.indata.systs):
                    # Skip inactive systematics
                    syst = syst.decode("utf-8")
                    if syst not in nonzero_systs:
                        continue

                    syst_hist = debug_data.syst_hists[channel]
                    down_values = syst_hist[
                        {"DownUp": "Down", "systs": syst, "processes": hist.sum}
                    ].values()
                    up_values = syst_hist[
                        {"DownUp": "Up", "systs": syst, "processes": hist.sum}
                    ].values()
                    nom_values = nom_hist[{"processes": hist.sum}].values()[..., None]

                    # Compute relative impact (maximum of up/down variations)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        rel_impact_down = np.where(
                            nom_values > 0, (down_values - nom_values) / nom_values, 0
                        )
                        rel_impact_up = np.where(
                            nom_values > 0, (up_values - nom_values) / nom_values, 0
                        )

                    max_impact = np.nanmax(np.maximum(rel_impact_down, rel_impact_up))
                    min_impact = np.nanmin(np.minimum(rel_impact_down, rel_impact_up))
                    syst_impact.append(max_impact)
                    syst_impact_down.append(min_impact)
                    syst_names.append(syst)

                # Sort by impact
                sorted_indices = np.argsort(syst_impact)[::-1]
                sorted_impact = [syst_impact[i] for i in sorted_indices]
                sorted_impact_down = [syst_impact_down[i] for i in sorted_indices]
                sorted_names = [syst_names[i] for i in sorted_indices]

                # Plot top 10
                top_n = min(10, len(sorted_impact))
                ax.barh(range(top_n), sorted_impact[:top_n])
                ax.barh(range(top_n), sorted_impact_down[:top_n])
                ax.set_yticks(range(top_n))
                ax.set_yticklabels(sorted_names[:top_n])
                ax.set_xlabel("Maximum relative impact")
                ax.set_title(f"Channel: {channel} - Top systematics impact")

                # ax.set_xlim(min(sorted_impact_down[:top_n]), max( sorted_impact[:top_n]))

                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"{channel}_syst_impact.png"))
                plt.close()

        print(f"✓ Generated diagnostic plots in {output_dir}")
        # except Exception as e:
        #     print(f"✗ Failed to generate diagnostic plots: {str(e)}")

    # Summary
    if issues_found == 0:
        print("\n✓ No issues found in the input data!")
    else:
        print(f"\n✗ Found {issues_found} potential issues in the input data")

    return issues_found == 0


def main():
    parser = argparse.ArgumentParser(description="Debug input data for fitting")
    parser.add_argument("input_file", help="Path to input data file")
    parser.add_argument("--output-dir", "-o", help="Directory to save diagnostic plots")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Print detailed information"
    )

    args = parser.parse_args()

    success = debug_input_data(args.input_file, args.output_dir, args.verbose)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
