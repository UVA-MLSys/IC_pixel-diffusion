import numpy as np
from tqdm import tqdm

import scipy.stats as stats
from typing import Dict, List, Tuple
import json
import Pk_library as PKL

import math
import torch

from torch import Tensor
from typing import Optional, Sequence, Tuple

def isotropic_binning(
    shape: Sequence[int],
    bins: Optional[int] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    r"""Computes an isotropic binning over the frequency domain.

    Arguments:
        shape: The domain shape :math:`(L_1, ..., L_N)`.
        bins: The number of bins :math:`B`.

    Returns:
        The bin edges, counts and indices, with shape :math:`(B + 1)`, :math:`(B + 1)`
        and :math:`(L_1 x ... x L_N)`, respectively.
    """

    k = []

    for s in shape:
        k_i = torch.fft.fftfreq(s)
        k.append(k_i)

    k2 = map(torch.square, k)
    k2_iso = sum(torch.meshgrid(*k2, indexing="ij"))
    k_iso = torch.sqrt(k2_iso)

    if bins is None:
        bins = math.floor(math.sqrt(k_iso.ndim) * min(k_iso.shape) / 2)

    edges = torch.linspace(0, k_iso.max(), bins + 1)

    indices = torch.bucketize(k_iso.flatten(), edges)
    counts = torch.bincount(indices, minlength=bins + 1)

    return edges, counts, indices


def isotropic_power_spectrum(x: Tensor, spatial: int = 2) -> Tuple[Tensor, Tensor]:
    r"""Computes the isotropic power spectrum of a field.

    Arguments:
        x: A field tensor, with shape :math:`(*, L_1, ..., L_N)`.
        spatial: The number of spatial dimensions :math:`N`.

    Returns:
        The binned power spectrum and the frequency bins (in cycles per pixel), with
        shape :math:`(*, B)` and :math:`(B)`, respectively.
    """

    x = torch.as_tensor(x)

    batch, shape = x.shape[:-spatial], x.shape[-spatial:]

    # Binning
    edges, counts, indices = isotropic_binning(shape)

    # Power spectrum
    s = torch.fft.fftn(x, dim=tuple(range(-spatial, 0)), norm="ortho")
    p = torch.square(torch.abs(s))
    p = torch.flatten(p, start_dim=-spatial)

    p_iso = torch.zeros((*batch, *edges.shape), dtype=x.dtype)
    p_iso = p_iso.scatter_add(dim=-1, index=indices.expand_as(p), src=p)
    p_iso = p_iso / torch.clip(counts, min=1)

    return p_iso[..., 1:], edges[1:]

class CosmologyMetrics:
    """Compute cosmology-specific metrics for dark matter density fields."""
    
    def __init__(
        self, boxsize: float = 1000.0, kmax=1.0
    ):
        self.boxsize = boxsize
        self.kmax = kmax
        
        try:
            from nbodykit.lab import ArrayMesh, FFTPower
            self.use_nbodykit = True
            print('Using nbodykit')
        except:
            self.use_nbodykit = False
            print(f'Nbodykit is not available. Using Pylians')
        
    def power_spectrum(self, field: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute 1D power spectrum P(k)."""
        field = field.squeeze()
        field = (field - np.mean(field))/np.std(field)
        
        if self.use_nbodykit:
            from nbodykit.lab import ArrayMesh, FFTPower
            mesh = ArrayMesh(field, BoxSize=self.boxsize)
            result = FFTPower(mesh, mode='1d', kmax=self.kmax)
            PS = result.power
            return PS['power'].real[1:], PS['k'][1:]  # Skip k=0
        else:
            delta = field.astype(np.float32)
    
            Pk_obj = PKL.Pk(delta, BoxSize=self.boxsize, axis=0, MAS='None', threads=1, verbose=False)
            
            # Extract k and Power
            k = Pk_obj.k1D
            PS = Pk_obj.Pk1D
            
            if self.kmax is None:
                return PS, k 
            else:
                mask = [i>0 and i <= self.kmax for i in k]
                return PS[mask], k[mask]
        
    
    def cross_correlation(self, field1: np.ndarray, field2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute cross-correlation C(k) = P_12(k) / sqrt(P_1(k) * P_2(k))."""
        field1 = field1.squeeze()
        field2 = field2.squeeze()
        
        field1 = (field1 - np.mean(field1))/np.std(field1)
        field2 = (field2 - np.mean(field2))/np.std(field2)
        
        if self.use_nbodykit:
            from nbodykit.lab import ArrayMesh, FFTPower
            mesh1 = ArrayMesh(field1, BoxSize=self.boxsize)
            mesh2 = ArrayMesh(field2, BoxSize=self.boxsize)
            
            result_12 = FFTPower(first=mesh1, mode='1d', second=mesh2, kmax=self.kmax)
            result_11 = FFTPower(first=mesh1, mode='1d', kmax=self.kmax)
            result_22 = FFTPower(first=mesh2, mode='1d', kmax=self.kmax)
            
            PS_12 = result_12.power['power'].real
            PS_11 = result_11.power['power'].real
            PS_22 = result_22.power['power'].real
            k = result_12.power['k']
            
            # Avoid division by zero
            denominator = np.sqrt(PS_11 * PS_22)
            denominator[denominator == 0] = 1e-10
            
            C_k = PS_12 / denominator
            return C_k[1:], k[1:]  # Skip k=0
        else:
            delta1 = field1.astype(np.float32)
            delta2 = field2.astype(np.float32)
            
            # Calculate auto and cross-power spectra
            # XPk calculates both auto (Pk1D) and cross (PkX1D) spectra
            Pk_obj = PKL.XPk(
                [delta1, delta2], BoxSize=self.boxsize, axis=0, 
                MAS=['None', 'None'], threads=1
            )
            
            # Extract k
            k = Pk_obj.k1D
            
            # Extract Power Spectra
            # Pk1D[:, 0] is Auto-Power of field 1 (x)
            # Pk1D[:, 1] is Auto-Power of field 2 (y)
            # PkX1D[:, 0] is Cross-Power of field 1 and 2
            PS_xx = Pk_obj.Pk1D[:, 0]
            PS_yy = Pk_obj.Pk1D[:, 1]
            PS_xy = Pk_obj.PkX1D[:, 0]
            
            # Calculate Cross-Correlation Coefficient
            # Avoid division by zero if necessary
            PS = PS_xy / np.sqrt(PS_xx * PS_yy)
            
            
            if self.kmax is None:
                return PS, k
            else:
                mask = [i>0 and i <= self.kmax for i in k]
                return PS[mask], k[mask]
    
    def transfer_function(self, field: np.ndarray, truth: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute transfer function T(k) = sqrt(P_sample(k) / P_truth(k))."""
        P_sample, k_sample = self.power_spectrum(field)
        P_truth, k_truth = self.power_spectrum(truth)
        
        # Avoid division by zero
        P_truth_safe = np.where(P_truth > 0, P_truth, 1e-10)
        T_k = np.sqrt(P_sample / P_truth_safe)
        return T_k, k_sample
    
    def power_spectrum_rmse_single(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        spatial: int = 3,
        n_bands: int = 3,
        eps: float = 1e-12,
    ) -> Dict[str, float]:
        """
        Power Spectrum RMSE as used in Ohana et al. (2023).

        u: ground-truth field, shape (..., N, N, N)
        v: generated field, same shape
        """

        # Isotropic power spectra
        Pu, k = isotropic_power_spectrum(u, spatial=spatial)
        Pv, _ = isotropic_power_spectrum(v, spatial=spatial)

        # Relative power
        ratio = Pv / (Pu + eps)

        # Log-spaced frequency bands
        log_k = torch.log(k)
        band_edges = np.linspace(
            log_k.min(), log_k.max(), n_bands + 1
        )

        rmse = {}

        for i in range(n_bands):
            mask = (log_k >= band_edges[i]) & (log_k < band_edges[i + 1])
            if mask.any():
                r = ratio[..., mask]
                rmse_val = torch.sqrt(torch.mean((r - 1.0) ** 2))
            else:
                rmse_val = torch.tensor(float("nan"))

            rmse[["low", "mid", "high"][i]] = rmse_val.item()

        return rmse


class ValidationMetrics:
    """Comprehensive validation metrics for posterior sampling."""
    
    def __init__(
        self, boxsize: float = 1000.0, kmax=None
    ):
        self.cosmo_metrics = CosmologyMetrics(boxsize, kmax=kmax)
        self.kmax = kmax
        
    def power_spectrum_accuracy(self, samples: np.ndarray, truth: np.ndarray) -> Dict[str, float]:
        """
        Compute power spectrum relative error.
        
        Args:
            samples: (n_samples, H, W, D)
            truth: (H, W, D)
        """
        P_truth, k_truth = self.cosmo_metrics.power_spectrum(truth)
        
        # Compute for all samples
        relative_errors = []
        for sample in samples:
            P_sample, k_sample = self.cosmo_metrics.power_spectrum(sample)
            
            rel_error = np.abs(P_sample - P_truth) / P_truth
            relative_errors.append(np.mean(rel_error))
        
        return {
            'mean': float(np.mean(relative_errors)),
            'std': float(np.std(relative_errors)),
            'max': float(np.max(relative_errors)),
            'score': float(1.0 - np.mean(relative_errors))  # Convert to 0-1 score
        }
    
    def cross_correlation_score(self, samples: np.ndarray, truth: np.ndarray) -> Dict[str, float]:
        """
        Compute cross-correlation C(k) between samples and truth.
        Higher C(k) means better reconstruction.
        """
        cross_corrs = []
        for sample in samples:
            C_k, k = self.cosmo_metrics.cross_correlation(sample, truth)
            
            # Average cross-correlation across all k
            cross_corrs.append(np.mean(C_k))

        return {
            'mean': float(np.mean(cross_corrs)),
            'std': float(np.std(cross_corrs)),
            'min': float(np.min(cross_corrs)),
            'score': float(np.mean(cross_corrs))  # Already in [0, 1] range
        }
    
    def transfer_function_accuracy(self, samples: np.ndarray, truth: np.ndarray) -> Dict[str, float]:
        """
        Compute transfer function T(k). Should be ~1.0 for perfect reconstruction.
        """
        transfer_deviations = []
        for sample in samples:
            T_k, k = self.cosmo_metrics.transfer_function(sample, truth)
            
            # Measure deviation from ideal T(k) = 1
            deviation = np.mean(np.abs(T_k - 1.0))
            transfer_deviations.append(deviation)
        
        return {
            'mean_deviation': float(np.mean(transfer_deviations)),
            'std': float(np.std(transfer_deviations)),
            'score': float(1.0 - np.mean(transfer_deviations))
        }
    
    def pearson_correlation(self, samples: np.ndarray, truth: np.ndarray) -> Dict[str, float]:
        """Compute Pearson correlation coefficient between samples and truth."""
        correlations = []
        for sample in samples:
            corr = np.corrcoef(sample.flatten(), truth.flatten())[0, 1]
            correlations.append(corr)
        
        return {
            'mean': float(np.mean(correlations)),
            'std': float(np.std(correlations)),
            'score': float(np.mean(correlations))  # Already in [-1, 1], but should be positive
        }
        
    def power_spectrum_rmse(self, samples: np.ndarray, truth: np.ndarray) -> Dict[str, Dict]:
        
        bands = ['low', 'mid', 'high']
        results = {band: [] for band in bands}
        scores = []
        for sample in samples:
            score = self.cosmo_metrics.power_spectrum_rmse_single(truth, sample)
            for band in bands:
                results[band].append(score[band])
        
        new_results = {}
        for band in bands:
            new_results[band] = {
                'mean': np.mean(results[band]),
                'std': np.std(results[band]),
                'score': np.mean(results[band])
            }
        
        return new_results
        
    def vrmse_score(
        self, samples: np.ndarray, truth: np.ndarray, epsilon=1e-6
    ) -> Dict[str, float]:
        """
        Computes Variance-Normalized RMSE (VRMSE) for a set of samples against a single truth.
        
        Formula: VRMSE = sqrt( MSE(u, v) / (Var(u) + epsilon) )
        
        Args:
            samples (list of np.ndarray or np.ndarray): 
                The predicted fields (v). Can be a list of 3D arrays or a 4D array [N, D, H, W].
            truth (np.ndarray): 
                The single ground truth field (u). Shape [D, H, W].
            epsilon (float): 
                Numerical stability term (default 10^-6).
                
        Returns:
            np.ndarray: An array of VRMSE scores (one per sample).
        """
        
        # 1. Pre-compute Truth Statistics (Denominator)
        # This is constant for all samples, so we compute it once.
        # Variance = <(u - <u>)^2>
        truth_mean = np.mean(truth)
        truth_var = np.mean((truth - truth_mean) ** 2)
        
        denominator = truth_var + epsilon
        
        # 2. Compute VRMSE for each sample
        scores = []
        
        # Iterate to save memory (avoid creating a massive difference tensor)
        # If samples is a numpy array [N, D, H, W], iterating yields [D, H, W] slices
        for v in samples:
            # MSE = <(u - v)^2>
            # We assume 'truth' and 'v' have the same shape
            mse = np.mean((truth - v) ** 2)
            
            # VRMSE calculation
            vrmse = np.sqrt(mse / denominator)
            scores.append(vrmse)
            
        return {
            'mean': float(np.mean(scores)),
            'std': float(np.std(scores)),
            'score': float(np.mean(scores))
        }
            
    
    def calibration_score(self, samples: np.ndarray, truth: np.ndarray) -> Dict[str, float]:
        """
        Check if posterior variance correctly captures uncertainty.
        Variance should correlate with squared error.
        """
        sample_mean = samples.mean(axis=0)
        sample_var = samples.var(axis=0)
        
        truth = truth.squeeze()
        squared_error = (sample_mean - truth) ** 2
        
        # Correlation between predicted variance and actual error
        valid_mask = sample_var > 0  # Avoid zeros
        calibration_corr = np.corrcoef(
            sample_var[valid_mask].flatten(),
            squared_error[valid_mask].flatten()
        )[0, 1]
        
        # Also check if normalized residuals are ~N(0,1)
        normalized_residuals = (truth - sample_mean) / (np.sqrt(sample_var) + 1e-10)
        ks_statistic, ks_pvalue = stats.kstest(
            normalized_residuals.flatten(),
            'norm'
        )
        
        return {
            'variance_error_corr': float(calibration_corr),
            'ks_statistic': float(ks_statistic),
            'ks_pvalue': float(ks_pvalue),
            'score': float(calibration_corr)  # Use correlation as score
        }
    
    def coverage_score(self, samples: np.ndarray, truth: np.ndarray, alpha: float = 0.95) -> Dict[str, float]:
        """
        Check if truth falls within predicted confidence intervals.
        For 95% CI, ~95% of voxels should contain truth.
        """
        lower = np.quantile(samples, (1 - alpha) / 2, axis=0)
        upper = np.quantile(samples, 1 - (1 - alpha) / 2, axis=0)
        
        coverage = np.mean((truth >= lower) & (truth <= upper))
        
        # Deviation from expected coverage
        coverage_error = np.abs(coverage - alpha)
        
        return {
            'coverage': float(coverage),
            'expected': float(alpha),
            'error': float(coverage_error),
            'score': float(1.0 - coverage_error)  # Penalize deviation
        }
    
    def diversity_score(self, samples: np.ndarray) -> Dict[str, float]:
        """
        Ensure samples are diverse (not mode collapse).
        Compute pairwise L2 distances between samples.
        """
        n_samples = len(samples)
        pairwise_dists = []
        
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                dist = np.mean((samples[i] - samples[j]) ** 2)
                pairwise_dists.append(dist)
        
        mean_diversity = np.mean(pairwise_dists)
        
        return {
            'mean_pairwise_distance': float(mean_diversity),
            'std': float(np.std(pairwise_dists)),
            'score': float(mean_diversity)  # Higher is more diverse
        }
    
    def compute_all_metrics(self, samples: np.ndarray, truth: np.ndarray) -> Dict[str, Dict]:
        """Compute all validation metrics for a single example."""
        return {
            'power_spectrum': self.power_spectrum_accuracy(samples, truth),
            'cross_correlation': self.cross_correlation_score(samples, truth),
            'transfer_function': self.transfer_function_accuracy(samples, truth),
            'pearson': self.pearson_correlation(samples, truth),
            'calibration': self.calibration_score(samples, truth),
            'coverage': self.coverage_score(samples, truth),
            'diversity': self.diversity_score(samples),
            'vrmse': self.vrmse_score(samples, truth),
            'power_spectrum_rmse': self.power_spectrum_rmse(samples, truth)
        }


class ValidationSuite:
    """Run validation across entire validation dataset."""
    
    def __init__(
        self, boxsize: float = 1000.0, kmax=None
    ):
        self.metrics_computer = ValidationMetrics(
            boxsize, kmax=kmax
        )
        
    def evaluate_dataset(
        self,
        val_examples: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
        weights: Dict[str, float] = None
    ) -> Dict:
        """
        Evaluate entire validation dataset.
        
        Args:
            val_examples: List of (observation, samples, truth) tuples
                - observation: (H, W, D)
                - samples: (n_samples, H, W, D)  # e.g., 10 samples
                - truth: (H, W, D)
            weights: Weights for computing composite score
        
        Returns:
            Dictionary with aggregate metrics and per-example results
        """
        if weights is None:
            weights = {
                'power_spectrum': 0.25,
                'cross_correlation': 0.20,
                'transfer_function': 0.15,
                'pearson': 0.10,
                'calibration': 0.05,
                'coverage': 0.10,
                'diversity': 0.05,
                'vrmse': 0.10
            }
        
        all_results = []
        
        print(f"Evaluating {len(val_examples)} validation examples...")
        for idx, (observation, samples, truth) in enumerate(tqdm(val_examples)):
            # Compute all metrics for this example
            example_metrics = self.metrics_computer.compute_all_metrics(samples, truth)
            all_results.append(example_metrics)
        
        # Aggregate across all examples
        aggregate_metrics = self._aggregate_results(all_results, weights)
        
        return {
            'aggregate_metrics': aggregate_metrics,
            'per_example_results': all_results,
            'weights': weights
        }
    
    def _aggregate_results(self, all_results: List[Dict], weights: Dict[str, float]) -> Dict:
        """Aggregate results across all validation examples."""
        aggregate_metrics = {}
        
        # Aggregate each metric
        for metric_name in all_results[0].keys():
            if 'score' in all_results[0][metric_name]:
                scores = []
                for result in all_results:
                    scores.append(result[metric_name]['score'])
                
                aggregate_metrics[metric_name] = {
                    'mean': float(np.mean(scores)),
                    'std': float(np.std(scores)),
                    'min': float(np.min(scores)),
                    'max': float(np.max(scores)),
                    'median': float(np.median(scores))
                }
            else:
                for sub_metric_name in all_results[0][metric_name].keys():
                    scores = []
                    for result in all_results:
                        scores.append(result[metric_name][sub_metric_name]['score'])
                    
                    aggregate_metrics[metric_name+'_'+sub_metric_name] = {
                        'mean': float(np.mean(scores)),
                        'std': float(np.std(scores)),
                        'min': float(np.min(scores)),
                        'max': float(np.max(scores)),
                        'median': float(np.median(scores))
                    }
        
        return aggregate_metrics
    
    def save_results(self, results: Dict, output_path: str):
        """Save validation results to JSON file."""
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_path}")
    
    def print_summary(self, results: Dict):
        """Print formatted summary of validation results."""
        print("\n" + "="*60)
        print("VALIDATION RESULTS SUMMARY")
        print("="*60)
        
        print("\nDETAILED METRICS:")
        print("-" * 60)
        
        metrics = results['aggregate_metrics']
        for metric_name, values in metrics.items():
            print(f"\n{metric_name.upper().replace('_', ' ')}:")
            print(f"  Mean:   {values['mean']:.4f} ± {values['std']:.4f}")
            print(f"  Range:  [{values['min']:.4f}, {values['max']:.4f}]")
            print(f"  Median: {values['median']:.4f}")
        
        print("\n" + "="*60)