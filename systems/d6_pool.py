"""D6 dice pool system — engine, constants, and descriptions."""

import numpy as np

# ── Advantage thresholds ─────────────────────────────────────────────────────
THRESHOLD_MAP = {
    "Double Advantage": 2,
    "Advantage": 3,
    "Normal": 4,
    "Disadvantage": 5,
    "Double Disadvantage": 6,
}

ADVANTAGE_LEVELS = list(THRESHOLD_MAP.keys())
ADVANTAGE_DEFAULT_INDEX = 2  # "Normal"

# ── Posture configs for comparison charts ────────────────────────────────────
POSTURE_CONFIGS = {
    "Normal":           (False, False, False),
    "Safe":             (True,  False, False),
    "Blessed":          (False, True,  False),
    "Cursed":           (False, False, True),
    "Unnatural":        (False, True,  True),
    "Safe + Blessed":   (True,  True,  False),
    "Safe + Unnatural": (True,  True,  True),
}

POSTURE_COLORS = {
    "Normal": "#636EFA",
    "Safe": "#00CC96",
    "Blessed": "#FFA15A",
    "Cursed": "#EF553B",
    "Unnatural": "#AB63FA",
    "Safe + Blessed": "#19D3F3",
    "Safe + Unnatural": "#FF6692",
}

ADVANTAGE_COLORS = {
    "Double Disadvantage": "#EF553B",
    "Disadvantage": "#FFA15A",
    "Normal": "#636EFA",
    "Advantage": "#00CC96",
    "Double Advantage": "#19D3F3",
}

# ── Complication methods ─────────────────────────────────────────────────────
COMPLICATION_METHODS = {
    "Standard (strict): 1s > Marks": "standard",
    "Standard (loose): 1s >= Marks": "standard_loose",
    "Broad (strict): 1s + 4s > Marks": "broad_strict",
    "Broad (loose): 1s + 4s >= Marks": "broad_loose",
    "Proportional: 1s >= half Marks (round up)": "proportional",
    "Split: on fail 1s > Marks, on pass 4s > Marks": "split",
    "Split (DoS): on fail 1s > Marks, on pass 4s > DoS": "split_dos",
    "Split (Misses): on fail 1s > Marks, on pass 4s > non-mark dice": "split_misses",
}

COMPLICATION_DESCRIPTIONS = {
    "standard": (
        "Complication if you roll more 1s than marks.\n\n"
        "**On failure:** Very likely — few marks means even a couple 1s overwhelm them.\n\n"
        "**On success:** Rare — you need a lot of marks to succeed, so 1s almost never outnumber them.\n\n"
        "**Overall:** Complications mostly pile onto failures; successes are clean. Becomes very safe with large pools."
    ),
    "standard_loose": (
        "Complication if 1s are equal to or greater than marks.\n\n"
        "**On failure:** Very likely — ties now count, so even matching your few marks triggers it.\n\n"
        "**On success:** Uncommon but possible — a tight success where 1s match marks will complicate.\n\n"
        "**Overall:** Slightly more punishing than Standard across the board, especially on marginal rolls."
    ),
    "broad_strict": (
        "Counts both 1s and 4s together. Complication if that total exceeds your marks.\n\n"
        "**On failure:** Almost guaranteed — 4s and 1s together easily outnumber your few marks.\n\n"
        "**On success:** Noticeably more common — 4s are your weakest hits, and they count against you here.\n\n"
        "**Overall:** Complications are frequent at all skill levels. Even strong rollers face them regularly."
    ),
    "broad_loose": (
        "Counts both 1s and 4s together. Complication if that total equals or exceeds your marks.\n\n"
        "**On failure:** Nearly automatic.\n\n"
        "**On success:** Common — any roll heavy on 4s and 1s relative to marks triggers it.\n\n"
        "**Overall:** The most aggressive method. Complications are a constant presence."
    ),
    "proportional": (
        "Complication if your 1s reach at least half your marks (rounded up).\n\n"
        "**On failure:** Likely — few marks means even one or two 1s can hit the threshold.\n\n"
        "**On success:** Scales down — more marks raises the bar (6 marks needs 3+ ones).\n\n"
        "**Overall:** Complications fade as proficiency grows. Easy table math — just halve your marks."
    ),
    "split": (
        "Different rules for pass vs fail.\n\n"
        "**On failure:** Complication if 1s outnumber marks (same as Standard).\n\n"
        "**On success:** Complication if 4s outnumber marks. Since every 4 already counts as a mark, "
        "4s can never exceed total marks — so success complications essentially never happen.\n\n"
        "**Overall:** Failures get complicated, successes are always clean."
    ),
    "split_dos": (
        "Different rules for pass vs fail.\n\n"
        "**On failure:** Complication if 1s outnumber marks (same as Standard).\n\n"
        "**On success:** Complication if 4s outnumber your Degrees of Success (marks minus DR). "
        "A narrow win loaded with 4s will trigger, but a dominant success won't.\n\n"
        "**Overall:** Tight victories feel messy, dominant victories feel clean, failures punish bad luck."
    ),
    "split_misses": (
        "Different rules for pass vs fail.\n\n"
        "**On failure:** Complication if 1s outnumber marks (same as Standard).\n\n"
        "**On success:** Complication if 4s outnumber the dice that missed the threshold "
        "(e.g. rolls of 1/2/3 under Normal). How often this fires depends heavily on your advantage level — "
        "with Disadvantage most dice miss so 4s rarely outnumber them, but with Double Advantage only 1s miss, "
        "so a handful of 4s can easily outnumber the few misses.\n\n"
        "**Overall:** Success complications are tied to advantage level. Highly advantaged characters "
        "see more success complications from their weak hits; disadvantaged characters almost never do. "
        "Failures behave the same as Standard."
    ),
}

COMPLICATION_HELP = (
    "Determines what triggers a Complication on a roll. Different methods "
    "change how often complications appear and how they relate to your "
    "success or failure."
)

# ── Simulation defaults ──────────────────────────────────────────────────────
POOL_RANGE = np.arange(1, 31)
DR_RANGE = np.arange(0, 21)
COMPARE_TRIALS = 50_000


# ── Engine ───────────────────────────────────────────────────────────────────
def simulate(n_dice, thresh, is_safe, is_blessed, is_cursed, trials, rng,
             comp_method="standard", difficulty=0):
    """Run a single pool-size simulation. Returns (marks, complication) 1-D arrays."""
    rolls = rng.integers(1, 7, size=(trials, n_dice))
    if is_safe:
        mask = rolls == 1
        rolls = np.where(mask, rng.integers(1, 7, size=(trials, n_dice)), rolls)
    m = np.sum(rolls >= thresh, axis=1).astype(np.int64)
    if is_blessed:
        m += np.sum(rolls == 6, axis=1)
    if is_cursed:
        m -= np.sum(rolls == 1, axis=1)
    ones = np.sum(rolls == 1, axis=1)
    fours = np.sum(rolls == 4, axis=1)
    non_marks = np.sum(rolls < thresh, axis=1)
    comp = _resolve_complication(comp_method, m, ones, fours, non_marks, difficulty)
    return m, comp


def _resolve_complication(comp_method, m, cum_ones, cum_fours, cum_non_marks, difficulty):
    """Apply complication logic. All inputs are (trials, max_pool) shaped."""
    if comp_method == "broad_strict":
        return (cum_ones + cum_fours) > m
    elif comp_method == "broad_loose":
        return (cum_ones + cum_fours) >= m
    elif comp_method == "proportional":
        half_marks = np.ceil(np.maximum(m, 0) / 2).astype(np.int64)
        return cum_ones >= half_marks
    elif comp_method == "split":
        passed = m >= difficulty
        return np.where(passed, cum_fours > m, cum_ones > m)
    elif comp_method == "split_dos":
        passed = m >= difficulty
        return np.where(passed, cum_fours > (m - difficulty), cum_ones > m)
    elif comp_method == "split_misses":
        passed = m >= difficulty
        return np.where(passed, cum_fours > cum_non_marks, cum_ones > m)
    elif comp_method == "standard_loose":
        return cum_ones >= m
    else:
        return cum_ones > m


def simulate_all_pools(thresh, is_safe, is_blessed, is_cursed, trials, rng,
                       comp_method="standard", difficulty=0):
    """Simulate pool sizes 1-30 in a single vectorised pass.

    Rolls one (trials, 30) matrix and uses cumulative sums to derive
    marks and complications for every pool size simultaneously.

    Returns:
        marks: (trials, 30) — marks[:, n-1] = marks for pool size n
        comp:  (trials, 30) — complication bool for each pool size
    """
    max_pool = len(POOL_RANGE)
    rolls = rng.integers(1, 7, size=(trials, max_pool))
    if is_safe:
        mask = rolls == 1
        rolls = np.where(mask, rng.integers(1, 7, size=(trials, max_pool)), rolls)

    # Cumulative counts along the pool axis
    cum_hits = np.cumsum(rolls >= thresh, axis=1).astype(np.int64)
    cum_ones = np.cumsum(rolls == 1, axis=1)
    cum_fours = np.cumsum(rolls == 4, axis=1)
    cum_non_marks = np.cumsum(rolls < thresh, axis=1)

    m = cum_hits.copy()
    if is_blessed:
        m += np.cumsum(rolls == 6, axis=1)
    if is_cursed:
        m -= cum_ones

    comp = _resolve_complication(comp_method, m, cum_ones, cum_fours, cum_non_marks, difficulty)
    return m, comp


def success_curve(thresh, is_safe, is_blessed, is_cursed, dr, comp_method, rng):
    """Return (success%, complication%) arrays for pool sizes 1-30."""
    m, cc = simulate_all_pools(
        thresh, is_safe, is_blessed, is_cursed, COMPARE_TRIALS, rng, comp_method, dr,
    )
    succ = np.mean((m - dr) >= 0, axis=0) * 100
    comp = np.mean(cc, axis=0) * 100
    return succ, comp


def heatmap_data(thresh, is_safe, is_blessed, is_cursed, comp_method, rng):
    """Return a 2D array of success% indexed by [DR, pool_size]."""
    m, _ = simulate_all_pools(
        thresh, is_safe, is_blessed, is_cursed, COMPARE_TRIALS, rng, comp_method, 0,
    )
    # Broadcast: m is (trials, 30), DR_RANGE is (21,) → compare via (21, trials, 30)
    data = np.mean(m[np.newaxis, :, :] >= DR_RANGE[:, np.newaxis, np.newaxis], axis=1) * 100
    return data


def posture_summary_label(advantage_label, safe, blessed, cursed, unnatural):
    """Build a human-readable label like '6d6 Advantage + Safe'."""
    parts = []
    if advantage_label != "Normal":
        parts.append(advantage_label)
    if safe:
        parts.append("Safe")
    if unnatural:
        parts.append("Unnatural")
    else:
        if blessed:
            parts.append("Blessed")
        if cursed:
            parts.append("Cursed")
    return " + ".join(parts) if parts else "Normal"
