from scipy import stats
from scipy.interpolate import interp1d
import numpy as np
import random
import matplotlib.pyplot as plt
import time
import os
import uuid
import imageio

from .utils import _filter_comparisons_by_ids, _group_comparisons_by_identity


def compute_results_unsupervised(
    EM_config,
    comparisons,
    groups=None,
    verbose=False,
    animation_root=None,
    person_id_to_was_uploaded=None,
    labeled_comparisons=None,
):
    """
    This is the entry-point for analyzing the comparisons without any annotations.

    The optional argument `labeled_comparisons` is only used to generate
    convergence gif's showing the iterations of the EM algorithm dynamically changing
    as the match/non-match confidence distributions steady out.
    It is not otherwise utilitized for any analysis.
    """
    # We will run the EM algorithm over each key-value mapping of sub-comparisons.
    sub_comparisons = {}

    # Include the aggregate performance of this provider (all comparisons).
    sub_comparisons["aggregate"] = comparisons

    # Include the person-level comparisons.
    comparisons_per_person = _group_comparisons_by_identity(comparisons)
    sub_comparisons.update(comparisons_per_person)

    # Include the group-level comparisons.
    for label, ids in groups.items():
        sub_comparisons[label] = _filter_comparisons_by_ids(comparisons, ids)

    # Run the EM algorithm.
    output = {}
    for label, comparisons in sub_comparisons.items():
        # Map EM specs to EM output. This can eventually include other
        # parameters besides `estimate_nonmatch_distribution_each_iteration`.
        point_estimate_em_output = {}
        bootstrapped_em_output = {}

        # TODO - remove the labeled_confidences ASAP once we no longer need
        # the convergence gif's to understand how the EM algorithm is evolving
        # the match/non-match distributions each iteration.
        unlabeled_results = ActualBenchmarkResults(
            comparisons, labeled_confidences=labeled_comparisons
        )

        for estimate_nonmatch_distribution_each_iteration in [False, True]:
            v = int(estimate_nonmatch_distribution_each_iteration)

            print(f"Starting EM version {v} point-estimate for", label)
            point_estimate_em_output[
                estimate_nonmatch_distribution_each_iteration
            ] = run_kernel_em(
                unlabeled_results,
                estimate_nonmatch_distribution_each_iteration,
                EM_config,
                person_id_to_was_uploaded,
                verbose=verbose,
                # Only include the EM gif's for the aggregate plots to save time/space and reduce clutter.
                dir_to_gen_EM_animation=animation_root
                if (label == "aggregate")
                else None,
            )

            print(f"Starting EM version {v} bootstrap for", label)
            bootstrapped_em_output[
                estimate_nonmatch_distribution_each_iteration
            ] = run_bootstrapper_kernel(
                unlabeled_results,
                estimate_nonmatch_distribution_each_iteration,
                EM_config,
                person_id_to_was_uploaded,
            )

        output[label] = {
            "point": point_estimate_em_output,
            "bootstrapped": bootstrapped_em_output,
        }
    return output


def run_kernel_em(
    unlabeled_results,
    estimate_nonmatch_distribution_each_iteration,
    EM_config,
    person_id_to_was_uploaded,
    verbose=False,
    dir_to_gen_EM_animation=None,
):
    """
    Produces a single EM estimate of Match (M) and Non-Match (NM) distributions, using all data
    in `unlabeled_results`.
    """
    em = KernelEMAlgorithm(
        prior_p_match_given_same_seed_identity_and_downloaded=EM_config[
            "EM_PRIOR_P_MATCH_GIVEN_SAME_SEED_IDENTITY_AND_DOWNLOADED"
        ],
        prior_p_match_given_same_seed_identity_and_uploaded=EM_config[
            "EM_PRIOR_P_MATCH_GIVEN_SAME_SEED_IDENTITY_AND_UPLOADED"
        ],
        prior_p_match_given_diff_seed_identity=EM_config[
            "EM_PRIOR_P_MATCH_GIVEN_DIFF_SEED_IDENTITY"
        ],
        delta_p_threshold=EM_config["EM_DELTA_P_STOPPING_THRESHOLD"],
        init_m_as_gaussan=EM_config["EM_INIT_MATCH_DISTRIBUTION_AS_GAUSSIAN"],
        bw_method=EM_config["EM_KDE_BW_METHOD"],
        # If we reestimate the non-match distribution each iteration, then return the final
        # non-match distribution from the EM algorithm.
        return_nm_iter_0=not estimate_nonmatch_distribution_each_iteration,
        estimate_nm_each_iteration=estimate_nonmatch_distribution_each_iteration,
        estimator_label="refit-NM"
        if estimate_nonmatch_distribution_each_iteration
        else "fixed-NM",
    )
    return em.estimate(
        unlabeled_results,
        person_id_to_was_uploaded,
        verbose=verbose,
        dir_to_gen_EM_animation=dir_to_gen_EM_animation,
    )


def run_bootstrapper_kernel(
    unlabeled_results,
    estimate_nonmatch_distribution_each_iteration,
    EM_config,
    person_id_to_was_uploaded,
):
    """
    Produces EM_NUM_BOOTSTRAPPED_DATASETS estimates of the M and NM distributions,
    each estimate using EM_FRAC_CONFIDENCES_TO_BOOTSTRAP% of the total data in
    `unlabeled_results`.
    """
    num_thresholds = EM_config["EM_BOOTSTRAP_NUM_INTERP_THRESHOLDS"]
    thresholds = [i / num_thresholds for i in range(num_thresholds + 1)]
    delta = thresholds[1]

    bootstrapped_results = []
    for i in range(EM_config["EM_NUM_BOOTSTRAPPED_DATASETS"]):
        if i % 10 == 0:
            print("Starting fit", i)

        results = SampledBenchmarkResults(
            unlabeled_results, EM_config["EM_FRAC_CONFIDENCES_TO_BOOTSTRAP"]
        )
        kde_nm, kde_m, nm_freq = run_kernel_em(
            results,
            estimate_nonmatch_distribution_each_iteration,
            EM_config,
            person_id_to_was_uploaded,
            verbose=False,
        )

        if i == 0:
            print("Bootstrapped data has size", len(results.get_confidences()))

        _, nonmatch_cdf = discrete_cdf(kde_nm, delta)
        _, match_cdf = discrete_cdf(kde_m, delta)

        fmr, fnmr = compute_FMR_FNMR_discrete_dist(nonmatch_cdf, match_cdf)
        bootstrapped_results.append((fmr, fnmr))

    fnmr_q, x_fmr_arr = compute_fmr_fnmr_quantiles(bootstrapped_results, EM_config)

    return fnmr_q, x_fmr_arr, bootstrapped_results


def key_has_matching_seed_people(key):
    ((_, person1), (_, person2)) = key
    return person1 == person2


def key_has_uploaded_people(key, person_id_to_was_uploaded):
    ((_, person1), (_, person2)) = key
    assert person1 == person2
    return person_id_to_was_uploaded[person1]


class KernelEMAlgorithm:
    def __init__(
        self,
        prior_p_match_given_same_seed_identity_and_downloaded=0.5,
        prior_p_match_given_same_seed_identity_and_uploaded=0.95,
        prior_p_match_given_diff_seed_identity=0.01,
        bw_method="scott",
        estimate_nm_each_iteration=False,
        return_nm_iter_0=False,
        delta_p_threshold=0.01,
        estimator_label=None,
        init_m_as_gaussan=False,
    ):
        # Note that the default parameter values should **never** be updated,
        # rather edit default_config.ini for package-wide updates OR for
        # trial-specific changes, create a config.ini to override the default config.

        self.p_match_same_downloaded = (
            prior_p_match_given_same_seed_identity_and_downloaded
        )
        self.p_match_same_uploaded = prior_p_match_given_same_seed_identity_and_uploaded
        self.p_match_diff = prior_p_match_given_diff_seed_identity
        self.bw_method = bw_method
        self.estimate_nm_each_iteration = estimate_nm_each_iteration
        self.return_nm_iter_0 = return_nm_iter_0
        self.delta_p_threshold = delta_p_threshold
        self.estimator_label = estimator_label
        self.init_m_as_gaussan = init_m_as_gaussan

    def _display_current_iteration(
        self, iteration, results, kde_nm, kde_m, max_change_p, start_time
    ):
        max_change_p = round(max_change_p, 5) if max_change_p is not None else "N/A"
        true_nonmatch_confidences = results.get_true_nonmatch_confidences()
        true_match_confidences = results.get_true_match_confidences()

        elapsed = time.time() - start_time

        x = np.linspace(0, 1, 150)

        fig, (ax, ax2) = plt.subplots(1, 2)
        ax.plot(x, kde_nm(x))
        ax.hist(true_nonmatch_confidences, bins=20, density=True, alpha=0.1)
        ax.set_xlabel("confidence")
        ax.set_ylabel("pdf")

        ax2.plot(x, kde_m(x))
        ax2.hist(true_match_confidences, bins=20, density=True, alpha=0.1)
        ax2.set_xlabel("confidence")

        fig.suptitle(
            f"EM ({self.estimator_label}) iteration {iteration} has "
            + f"Δp < {max_change_p} ({round(elapsed,1)}s elapsed)"
        )

        plt.tight_layout()
        fname = f"{self.dir}/{iteration:3}.png"
        fig.set_size_inches(8, 6)
        plt.savefig(fname, dpi=300)
        self.imgs.append(fname)

    def _gen_directory_for_animation(self, parent_dir):
        self.imgs = []
        self.dir = (
            parent_dir + os.sep + self.estimator_label + "-" + str(uuid.uuid4()).strip()
        )
        os.mkdir(self.dir)
        print("Animation will be prepared in", self.dir)

    def _gen_animation(self):
        movie_fname = f"{self.dir}/movie.gif"
        imageio.mimsave(
            movie_fname, [imageio.imread(filename) for filename in self.imgs], fps=1
        )

    def _get_prior_weight(self, key, person_id_to_was_uploaded):
        if key_has_matching_seed_people(key):
            if key_has_uploaded_people(key, person_id_to_was_uploaded):
                return self.p_match_same_uploaded
            else:
                return self.p_match_same_downloaded
        return self.p_match_diff

    def estimate(
        self,
        results,
        person_id_to_was_uploaded,
        verbose=False,
        dir_to_gen_EM_animation=None,
    ):
        """
        This algorithm learns the non-match and match distribution for a provider
        using the provider's confidence output during a benchmark.

        **Background:**

        During a benchmark, we run two types of comparisons:

        1. Same-seed-id (between two faces collected for the same identity)
        2. Diff-seed-id (between faces collected for different identities)

        If every face collected for an identity matched that identity,
        i.e., the scraping mechanism had 100% accuracy, then we could model

        1. Match distribution ~ Kernel(same-seed-id comparisons)
        2. Non-match distribution ~ Kernel(diff-seed-id comparisons)

        However, this is not the case. There are usually wrong faces scraped for
        each identity, which means

        1. Some same-seed-id comparisons are actually diff-id comparisons
        2. Some diff-seed-id comparisons are actually same-id comparisons

        **Priors:**

        Take two faces Fj, Fk.
        They are scraped for persons S(j), S(k).
        They bear true identities T(j), T(k).

        We assume the following priors on P(T(j) == T(k)):

        1. Same-seed-id comparisons, when S(j) == S(k):

           P(S(j) == T(j)) ~ 70% (scraper accuracy)

           P(T(j) == T(k) | S(j) == S(k)) ~ (70%)^2 ~ 50%

        2. Diff-seed-id comparisons, when S(j) != S(k):

           P(T(j) == T(k) | S(j) != S(k)) ~ 1% (very rare)

        **Bayes' Theorem:**

        Suppose we have approximated the match distribution M and the non-match distribution NM.
        Let the confidence value output for faces Fj, Fk be c(j, k).

        Bayes' Theorem gives::

            P(A | B)                   =   P(B | A)                 * P(A)            / P(B)
            P(T(j) == T(k) | c(j,k))   =   P(c(j,k) | T(j) == T(k)) * P(T(j) == T(k)) / P(c(j,k))

        To evaluate this formula, note that

        1. P(c(j,k) | T(j) == T(k)) ~ M(c(j,k)), which is known.
        2. By the priors, P(T(j) == T(k)) ~ 50% or 1% depending on (S(j), S(k)) which are known.

        This lets us compute P(T(j) == T(k) | c(j,k)) for each pair Fj, Fk.
        We can then re-estimate M, NM using these weights.

        **EM Algorithm:**

        0. Define M as match distribution, NM as non-match distribution.
        1. Initialize NM as kernel(diff-seed-id).
        2. Initialize M as gaussian(mean=1, sd=0.1) over [0, 1].
        3. Alternate between:

           3a. Compute p(match | j,k) using Bayes' Theorem for M,NM

           3b. Fit M (and optionally fit NM) using p(match | j,k).

        4. Stop once p(match | j,k) converges.

        **Extension:**

        Photos downloaded from news articles for some identity may have
        a ~70% likelihood of matching the true identity. This is denoted
        as "(scraper accuracy)" in the derivation above.

        However, photos uploaded from the user are more likely to
        match the true identity. So a separate prior can be specified
        depending on whether a person's set of photos was downloaded
        from news articles or uploaded locally.

        The `person_id_to_was_uploaded` map is responsible for this.
        """
        generate_EM_animation = dir_to_gen_EM_animation is not None
        if generate_EM_animation:
            self._gen_directory_for_animation(dir_to_gen_EM_animation)
            start_time = time.time()

        # Compute a kernel fit using all diff-seed comparisons, i.e.,
        # pairs of faces scraped for different identities.
        diff_confidences = list(results.get_diff_seed_id_confidences().values())
        kde_nm = stats.gaussian_kde(diff_confidences, bw_method=self.bw_method)
        original_kde_nm = kde_nm

        # This loads all confidences from the benchmark.
        mix_confidences_dict = results.get_confidences()
        mix_confidences = list(mix_confidences_dict.values())

        # We can either initialize M as a gaussian or as a mixture of
        # match/non-match distributions (using a kernel fit on all confidences).
        if self.init_m_as_gaussan:

            def kde_m(xs):
                return 2 * stats.norm.pdf(xs, 1, 0.1)

        else:
            kde_mix = stats.gaussian_kde(mix_confidences, bw_method=self.bw_method)
            kde_m = kde_mix

        # Set the priors for each comparison based on same-seed-id or diff-seed-id.
        weights = np.array(
            [
                self._get_prior_weight(k, person_id_to_was_uploaded)
                for k in mix_confidences_dict
            ]
        )

        # Stop when the max change in probabilities is sufficiently low, i.e.,
        # smaller than the user-specified `delta_p_threshold`.
        nm_probs = None
        prev_nm_probs = None
        iteration = 0
        max_change_p = None
        while prev_nm_probs is None or max_change_p > self.delta_p_threshold:

            if generate_EM_animation:
                self._display_current_iteration(
                    iteration, results, kde_nm, kde_m, max_change_p, start_time
                )

            iteration += 1

            # By saving the previous iteration's p(match | comparison)
            # we can see when these probabilities converge.
            prev_nm_probs = nm_probs

            # Assign priors and compute p(match) and p(non-match) densities for each comparison.
            nm_probs = kde_nm(mix_confidences) * (1 - weights)
            m_probs = kde_m(mix_confidences) * weights

            # Then normalize the densities into probabilities.
            nm_probs = nm_probs / (nm_probs + m_probs)
            m_probs = 1 - nm_probs

            if self.estimate_nm_each_iteration:
                # Estimate NM and M with a weighted kernel fit (using p(match) from above).
                kde_nm = stats.gaussian_kde(
                    mix_confidences, bw_method=self.bw_method, weights=nm_probs
                )
                kde_m = stats.gaussian_kde(
                    mix_confidences, bw_method=self.bw_method, weights=m_probs
                )
                nm_freq = np.mean(nm_probs)

            else:
                # Only estimate M with a weighted kernel fit (using p(match) from above).
                kde_m = stats.gaussian_kde(
                    mix_confidences, bw_method=self.bw_method, weights=m_probs
                )
                nm_freq = np.mean(nm_probs)

            if prev_nm_probs is not None:
                # Compute the greatest change in p(match | comparison) over all comparisons.
                max_change_p = np.max(np.abs(prev_nm_probs - nm_probs))
                if verbose:
                    print(f"Iteration {iteration}: max change in p is", max_change_p)

        if generate_EM_animation:
            self._display_current_iteration(
                iteration, results, kde_nm, kde_m, max_change_p, start_time
            )
            self._gen_animation()

        if verbose:
            print()

        return original_kde_nm if self.return_nm_iter_0 else kde_nm, kde_m, nm_freq


class BenchmarkResults:
    def get_confidences(self):
        raise NotImplemented("Base class cannot be called.")

    def get_diff_seed_id_confidences(self):
        raise NotImplemented("Base class cannot be called.")

    def get_same_seed_id_confidences(self):
        raise NotImplemented("Base class cannot be called.")


class ActualBenchmarkResults(BenchmarkResults):
    """
    This is a wrapper for the benchmark confidences.
    """

    def __init__(self, unlabeled_confidences, labeled_confidences=None):
        self.unlabeled_confidences = unlabeled_confidences
        self.labeled_confidences = labeled_confidences

        if self.labeled_confidences is not None:
            self.true_match_confidences = []
            self.true_nonmatch_confidences = []
            for key, value in self.labeled_confidences.items():
                ((_, person1), (_, person2)) = key
                if person1 == person2:
                    self.true_match_confidences.append(value)
                else:
                    self.true_nonmatch_confidences.append(value)

    def get_confidences(self):
        return dict(self.unlabeled_confidences)

    def get_diff_seed_id_confidences(self):
        return {
            k: v
            for k, v in self.get_confidences().items()
            if not key_has_matching_seed_people(k)
        }

    def get_same_seed_id_confidences(self):
        return {
            k: v
            for k, v in self.get_confidences().items()
            if key_has_matching_seed_people(k)
        }

    def get_true_match_confidences(self):
        if self.labeled_confidences is None:
            raise Exception("Ground truth labels not available.")
        return list(self.true_match_confidences)

    def get_true_nonmatch_confidences(self):
        if self.labeled_confidences is None:
            raise Exception("Ground truth labels not available.")
        return list(self.true_nonmatch_confidences)


class SampledBenchmarkResults(BenchmarkResults):
    """
    This is used to bootstrap a sample from the full benchmark.
    """

    def __init__(self, original_results, frac_to_sample):
        # TODO - could sample X% from same-id and X% from diff-id;
        # for now, will sample X% over all confidences
        # but this should be re-evaluated later!

        self.original_confidences = original_results.get_confidences()
        self.num_to_sample = int(len(self.original_confidences) * frac_to_sample)

        self._sample()

    def _sample(self):
        self.all_confidences = {
            k: v
            for k, v in random.sample(
                list(self.original_confidences.items()), self.num_to_sample
            )
        }
        self.same_seed_confidences = {
            k: v
            for k, v in self.all_confidences.items()
            if key_has_matching_seed_people(k)
        }
        self.diff_seed_confidences = {
            k: v
            for k, v in self.all_confidences.items()
            if not key_has_matching_seed_people(k)
        }

        if len(self.same_seed_confidences) < 2 or len(self.diff_seed_confidences) < 2:
            # Having <2 same or diff-seed comparisons is an edge case that could break EM
            # since the scipy.stats.gaussian_kde call requires 2+ values.
            self._sample()

    def get_confidences(self):
        return dict(self.all_confidences)

    def get_diff_seed_id_confidences(self):
        return dict(self.diff_seed_confidences)

    def get_same_seed_id_confidences(self):
        return dict(self.same_seed_confidences)


def compute_fmr_fnmr_quantiles(
    bootstrapped_results,
    EM_config,
    quantiles=[0.1, 0.25, 0.5, 0.75, 0.9],
    num_interp_pts=1000,
):
    """
    Input: bootstrapped_results is a list of (FMR, FNMR) parametric curves.

    Algorithm:
    1.  Each parametric curve (parameterized by discrete T) is interpolated
        using `interp1d` to output a continuous function FNMR(FMR).
    2.  For each FMR value `fmr` in [10^EM_BOOTSTRAP_LOG_MIN_FMR_VALUE, ..., 10^0]:
    2a.    Evaluate all interpolations interp_values = [FNMR_1(fmr), FNMR_2(fmr), ...]
    2b.    Compute certain quantiles of `interp_values`

    Output: error bars for `quantiles` on the FNMR/FMR curve.
    """
    interpolated_fnmr = []
    for fmr, fnmr in bootstrapped_results:
        interpolated_fnmr.append(interp1d(fmr, fnmr))

    poss_x_fmr_arr = 10 ** np.linspace(
        0,
        EM_config["EM_BOOTSTRAP_LOG_MIN_FMR_VALUE"],
        num=num_interp_pts,
        endpoint=True,
    )
    x_fmr_arr = []
    fnmr_q = {q: list() for q in quantiles}
    for x_fmr in poss_x_fmr_arr:

        all_fnmr = []
        for f in interpolated_fnmr:
            try:
                value = f(x_fmr)
            except:
                continue
            all_fnmr.append(value)

        if all_fnmr:
            qs = np.quantile(all_fnmr, quantiles)
            for q, val in zip(quantiles, qs):
                fnmr_q[q].append(val)
            x_fmr_arr.append(x_fmr)

    return fnmr_q, x_fmr_arr


def discrete_cdf(pdf, step_size):
    """
    Returns [cdf[0], cdf[step_size], ..., cdf[1]] based on `pdf`.
    """
    num_points = int(1 / step_size) + 1
    x = np.linspace(0, 1, num_points)
    p = pdf(x) * step_size
    cdf = np.cumsum(p)
    return x, cdf


def compute_FMR_FNMR_discrete_dist(nonmatch_cdf, match_cdf):
    """
    It is easy to compute the FMR/FNMR curve from the
    `discrete_cdf` output since FMR/FNMR are already parameterized.
    """
    fmr, fnmr = [], []
    for nm, m in zip(nonmatch_cdf, match_cdf):
        fmr.append(1 - nm)
        fnmr.append(m)
    return fmr, fnmr
