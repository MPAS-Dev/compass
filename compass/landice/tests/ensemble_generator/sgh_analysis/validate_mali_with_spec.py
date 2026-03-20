#!/usr/bin/env python
"""
Validate MALI subglacial hydrology simulations using radar specularity content.

Calculates a validation score for MALI subglacial hydrology simulations
using radar specularity content data from Young 2016.

Values of 0 represent specularity content below 20%.
Values of 3.3 represent specularity content above 20% and energy 1 microsecond
below the bed 15 dB lower than the bed echo.
Values of 6.7 represent specularity content above 20% and energy 1 microsecond
below the bed 15 dB within the bed echo.
"""

import json
from argparse import ArgumentParser

import cmocean
import matplotlib.pyplot as plt
import numpy as np
import tifffile as tiff
import xarray as xr
from scipy.interpolate import griddata
from scipy.stats import binned_statistic_2d


class validateWithSpec:
    """
    Validator for MALI simulations using specularity content observations.
    """

    def __init__(self):
        """Initialize validator and parse command-line arguments."""
        print("Gathering Information ...")
        parser = ArgumentParser(
            prog='validate_mali_with_spec.py',
            description='Calculate validation score for MALI subglacial \
                    hydrology simulations using specularity content')
        parser.add_argument("--maliFile", dest="maliFile", required=True,
                            help="MALI output file to validate")
        parser.add_argument(
            "--specTiff",
            dest="specTiff",
            required=True,
            help='Tiff file containing specularity content (Young 2016)')
        parser.add_argument("--compRes", dest="compRes", type=float,
                            help="Grid resolution for interpolation (meters)",
                            default=5000.0)
        parser.add_argument(
            "--ba_threshold",
            dest="ba_threshold",
            type=float,
            help="Balanced accuracy threshold for data compatibility",
            default=0.65)
        parser.add_argument("--output_json", dest="output_json",
                            help="JSON file to save validation results",
                            default="validation_results.json")
        parser.add_argument("--plot", dest="plot", action='store_true',
                            help="Generate comparison plots")

        args = parser.parse_args()
        self.options = args

    def interpolate_to_common_grid(self):
        """Interpolate MALI and specularity data to common grid."""
        # Open MALI file and read bed roughness parameter
        ds_mali = xr.open_dataset(
            self.options.maliFile,
            decode_times=False,
            decode_cf=False)

        # Read Wr from global attribute config_SGH_bed_roughness_max
        try:
            Wr = float(ds_mali.attrs['config_SGH_bed_roughness_max'])
            print(f"Using bed roughness parameter Wr from MALI file: {Wr}")
        except (KeyError, ValueError, TypeError):
            print(
                "Warning: config_SGH_bed_roughness_max \
                        not found in MALI file attributes")
            print("Using default value Wr = 0.1")
            Wr = 0.1

        self.options.Wr = Wr

        # Establish common grid
        res = self.options.compRes

        xCell = ds_mali['xCell'][:].values
        yCell = ds_mali['yCell'][:].values

        xmin = np.min(xCell)
        xmax = np.max(xCell)
        ymin = np.min(yCell)
        ymax = np.max(yCell)

        x_edges = np.arange(xmin, xmax + res, res)
        y_edges = np.arange(ymin, ymax + res, res)

        x_centers = (x_edges[:-1] + x_edges[1:]) / 2
        y_centers = (y_edges[:-1] + y_edges[1:]) / 2

        # Remap MALI data
        Xgrid, Ygrid = np.meshgrid(x_centers, y_centers)

        W = ds_mali['waterThickness'][-1, :].values
        W_remapped = griddata(points=(xCell, yCell),
                              values=W,
                              xi=(Xgrid, Ygrid),
                              method='linear')

        Z = ds_mali['bedTopography'][-1, :].values
        Z_remapped = griddata(points=(xCell, yCell),
                              values=Z,
                              xi=(Xgrid, Ygrid),
                              method='linear')

        H = ds_mali['thickness'][-1, :].values
        H_remapped = griddata(points=(xCell, yCell),
                              values=H,
                              xi=(Xgrid, Ygrid),
                              method='linear')

        # Open and process specularity TIFF
        with tiff.TiffFile(self.options.specTiff) as tif:
            page = tif.pages[0]
            specData = page.asarray()
            scale = page.tags["ModelPixelScaleTag"].value
            tiepoint = page.tags["ModelTiepointTag"].value

            pixelWidth = scale[0]
            pixelHeight = scale[1]

            i, j, k, x0, y0, z0 = tiepoint

            rows, cols = specData.shape

            x = x0 + np.arange(cols) * pixelWidth
            y = y0 - np.arange(rows) * pixelHeight

        specData = specData.astype(float)
        specData[specData == 0] = np.nan

        [Xspec, Yspec] = np.meshgrid(x, y)
        specData = specData.ravel()
        Xspec = Xspec.ravel()
        Yspec = Yspec.ravel()

        mask = np.isfinite(specData)
        specData = specData[mask]
        Xspec = Xspec[mask]
        Yspec = Yspec[mask]

        spec_remapped, x_edges_out, y_edges_out, binnum = binned_statistic_2d(
            Xspec, Yspec, specData,
            statistic='mean',
            bins=[x_edges, y_edges]
        )
        spec_remapped = spec_remapped.T

        # Filter specularity data
        floating = (910 / 1028) * H_remapped + Z_remapped <= 0
        spec_remapped[floating] = np.nan
        spec_remapped[H_remapped == 0] = np.nan

        east_AIS = Xgrid >= 0

        east_valid = east_AIS & np.isfinite(
            spec_remapped) & np.isfinite(W_remapped)
        west_valid = ~east_AIS & np.isfinite(
            spec_remapped) & np.isfinite(W_remapped)

        # Calculate Rwt (relative water thickness)
        Rwt_e = W_remapped[east_valid] / Wr
        Rwt_w = W_remapped[west_valid] / Wr

        # Define comparison thresholds
        Sthresh = 3.33  # Physically-based specularity threshold
        Rthresh = np.arange(0.95, 1.0, 0.01)

        Strue_e = spec_remapped[east_valid] >= Sthresh
        Sfalse_e = spec_remapped[east_valid] < Sthresh
        Strue_w = spec_remapped[west_valid] >= Sthresh
        Sfalse_w = spec_remapped[west_valid] < Sthresh

        Strue_e = Strue_e[:, None]
        Sfalse_e = Sfalse_e[:, None]
        Strue_w = Strue_w[:, None]
        Sfalse_w = Sfalse_w[:, None]

        Rtrue_e = Rwt_e[:, None] >= Rthresh
        Rfalse_e = ~Rtrue_e
        Rtrue_w = Rwt_w[:, None] >= Rthresh
        Rfalse_w = ~Rtrue_w

        tp_e = np.sum(Strue_e & Rtrue_e, axis=0)
        tn_e = np.sum(Sfalse_e & Rfalse_e, axis=0)
        fp_e = np.sum(Sfalse_e & Rtrue_e, axis=0)
        fn_e = np.sum(Strue_e & Rfalse_e, axis=0)

        tp_w = np.sum(Strue_w & Rtrue_w, axis=0)
        tn_w = np.sum(Sfalse_w & Rfalse_w, axis=0)
        fp_w = np.sum(Sfalse_w & Rtrue_w, axis=0)
        fn_w = np.sum(Strue_w & Rfalse_w, axis=0)

        true_agree_e = tp_e / (tp_e + fn_e)
        false_agree_e = tn_e / (tn_e + fp_e)

        true_agree_w = tp_w / (tp_w + fn_w)
        false_agree_w = tn_w / (tn_w + fp_w)

        balanced_score_e = 0.5 * (true_agree_e + false_agree_e)
        balanced_score_w = 0.5 * (true_agree_w + false_agree_w)

        self.BA_e = np.max(balanced_score_e)
        self.BA_w = np.max(balanced_score_w)
        print(f"balanced accuracy east: {self.BA_e:.4f}")
        print(f"balanced accuracy west: {self.BA_w:.4f}")

        self.Xgrid = Xgrid
        self.Ygrid = Ygrid
        self.spec = spec_remapped
        self.W = W_remapped
        self.H = H_remapped
        self.Z = Z_remapped
        self.Rthresh = Rthresh
        self.ind_Rmax = np.argmax(balanced_score_e + balanced_score_w)
        self.Sthresh = Sthresh
        self.floating = floating

    def check_data_compatibility(self):
        """
        Determine if model results are data-compatible.

        Returns
        -------
        bool
            True if both east and west BA >= threshold

        dict
            Compatibility metrics
        """
        threshold = self.options.ba_threshold

        is_compatible = (self.BA_e >= threshold and self.BA_w >= threshold)

        compatibility_metrics = {
            'BA_east': float(self.BA_e),
            'BA_west': float(self.BA_w),
            'threshold': float(threshold),
            'bed_roughness_Wr': float(self.options.Wr),
            'is_compatible': is_compatible,
            'BA_east_passes': bool(self.BA_e >= threshold),
            'BA_west_passes': bool(self.BA_w >= threshold),
        }

        return is_compatible, compatibility_metrics

    def plot_comparison_maps(self):
        """Generate comparison maps."""
        fig, ax = plt.subplots(figsize=(8, 5))
        H = self.H.copy()
        H[self.floating] = np.nan
        ax.contourf(self.Xgrid, self.Ygrid, H, levels=[
                    0.1, np.nanmax(self.H)], colors=[[0.9, 0.9, 0.9]])

        cmap = cmocean.cm.matter
        cmap = cmocean.tools.crop_by_percent(cmap, 45, which='max', N=None)
        s = np.full(self.Xgrid.shape, np.nan)
        s[self.spec >= self.Sthresh] = 1
        s[self.spec < self.Sthresh] = 0
        ax.pcolor(self.Xgrid, self.Ygrid, s, cmap=cmap)

        lev = self.Rthresh[self.ind_Rmax]
        ax.contour(
            self.Xgrid,
            self.Ygrid,
            self.W /
            self.options.Wr,
            levels=[lev],
            colors='k',
            linewidths=0.75)

        ax.set_xlim(-2e6, 2.6e6)
        ax.set_ylim(-2e6, 0)
        ax.text(-1.5e6, -1.55e6,
                f"balanced accuracy west: \
                        {np.round(self.BA_w, 2)}", fontsize=10)
        ax.text(-1.5e6, -1.8e6,
                f"balanced accuracy east: \
                        {np.round(self.BA_e, 2)}", fontsize=10)
        ax.set_aspect('equal', adjustable='box')
        plt.savefig(
            "spec_subglacialHydro_validation.png",
            dpi=1000,
            bbox_inches="tight")
        print("Saved validation plot: spec_subglacialHydro_validation.png")

    def save_results(self, is_compatible, compatibility_metrics):
        """Save validation results to JSON."""
        results = {
            'validation_type': 'specularity_content',
            'is_data_compatible': is_compatible,
            'metrics': compatibility_metrics,
            'mali_file': self.options.maliFile,
            'spec_file': self.options.specTiff,
        }

        with open(self.options.output_json, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Validation results saved to {self.options.output_json}")


def main():
    """Main entry point."""
    validator = validateWithSpec()

    validator.interpolate_to_common_grid()

    is_compatible, metrics = validator.check_data_compatibility()

    print("\n" + "=" * 60)
    print("DATA COMPATIBILITY ASSESSMENT")
    print("=" * 60)
    print(f"Bed roughness (Wr): {metrics['bed_roughness_Wr']:.4f}")
    print(
        f"East BA:  {
            metrics['BA_east']:.4f} (threshold: {
            metrics['threshold']:.4f}) - {
                'PASS' if metrics['BA_east_passes'] else 'FAIL'}")
    print(
        f"West BA:  {
            metrics['BA_west']:.4f} (threshold: {
            metrics['threshold']:.4f}) - {
                'PASS' if metrics['BA_west_passes'] else 'FAIL'}")
    print(
        f"Overall:  {
            'DATA COMPATIBLE' if is_compatible else 'NOT DATA COMPATIBLE'}")
    print("=" * 60 + "\n")

    if hasattr(validator.options, 'plot') and validator.options.plot:
        validator.plot_comparison_maps()

    validator.save_results(is_compatible, metrics)


if __name__ == "__main__":
    main()
