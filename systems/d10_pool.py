"""D10 dice pool system — engine, constants, and descriptions."""

import numpy as np

# ── Mark threshold ───────────────────────────────────────────────────────────
# Default: 6+ on a d10 produces a mark.  Advantage/Disadvantage shifts this.
DEFAULT_THRESHOLD = 6
ADVANTAGE_RANGE = (-3, 3)  # slider range: -3 (disadvantage) to +3 (advantage)

# ── Safe levels ──────────────────────────────────────────────────────────────
SAFE_OPTIONS = {
    "Off": 0,
    "Reroll 1s": 1,
    "Reroll 1s and 2s": 2,
}

# ── Blessed / Cursed tiers ───────────────────────────────────────────────────
BLESSED_OPTIONS = {
    "Off": None,
    "Blessed 10 (10s count double)": 10,
    "Blessed 9+ (9s and 10s count double)": 9,
    "Blessed 8+ (8, 9, 10s count double)": 8,
}

CURSED_OPTIONS = {
    "Off": None,
    "Cursed 1 (1s cancel a mark)": 1,
    "Cursed 2 (1s and 2s cancel a mark)": 2,
    "Cursed 3 (1s, 2s, and 3s cancel a mark)": 3,
}

# ── Risk Die complication ────────────────────────────────────────────────────
RISK_DIE_OPTIONS = {
    "Off": 0,
    "Complication on 1": 1,
    "Complication on 1–2": 2,
    "Complication on 1–3": 3,
}

RISK_DIE_HELP = (
    "One die in the pool is the Risk Die. It still produces marks normally, "
    "but also triggers a Complication if it rolls at or below the chosen threshold."
)

# ── Descriptions for sidebar info boxes ──────────────────────────────────────
BLESSED_DESCRIPTIONS = {
    None: "No Blessed effect active.",
    10: (
        "Results of 10 produce 2 marks instead of 1.\n\n"
        "**Impact:** Slight bonus on high rolls. Only 10% of dice are affected."
    ),
    9: (
        "Results of 9 or 10 produce 2 marks instead of 1.\n\n"
        "**Impact:** Moderate bonus. 20% of dice can double up, noticeably boosting strong rolls."
    ),
    8: (
        "Results of 8, 9, or 10 produce 2 marks instead of 1.\n\n"
        "**Impact:** Strong bonus. 30% of dice can double, significantly inflating mark counts."
    ),
}

CURSED_DESCRIPTIONS = {
    None: "No Cursed effect active.",
    1: (
        "Results of 1 each cancel one mark.\n\n"
        "**Impact:** Mild penalty. Only 10% of dice subtract, and bad luck has to overcome your marks."
    ),
    2: (
        "Results of 1 or 2 each cancel one mark.\n\n"
        "**Impact:** Noticeable penalty. 20% of dice can subtract, making low rolls more punishing."
    ),
    3: (
        "Results of 1, 2, or 3 each cancel one mark.\n\n"
        "**Impact:** Harsh penalty. 30% of dice actively work against you. Small pools become very risky."
    ),
}

RISK_DIE_DESCRIPTIONS = {
    0: "No Risk Die. Complications are disabled.",
    1: (
        "One die in the pool is the Risk Die. It produces marks normally, "
        "but a roll of 1 also triggers a Complication.\n\n"
        "**On any roll:** 10% base chance of complication, independent of success or failure.\n\n"
        "**Overall:** Complications are a flat random event — proficiency doesn't reduce them."
    ),
    2: (
        "The Risk Die triggers a Complication on a roll of 1 or 2.\n\n"
        "**On any roll:** 20% base chance of complication.\n\n"
        "**Overall:** Complications are fairly common. One in five rolls will have strings attached."
    ),
    3: (
        "The Risk Die triggers a Complication on a roll of 1, 2, or 3.\n\n"
        "**On any roll:** 30% base chance of complication.\n\n"
        "**Overall:** Complications are frequent. Nearly a third of all tests will have a wrinkle."
    ),
}

# ── Simulation defaults ──────────────────────────────────────────────────────
POOL_RANGE = np.arange(1, 31)
DR_RANGE = np.arange(0, 21)
COMPARE_TRIALS = 50_000


# ── Engine ───────────────────────────────────────────────────────────────────
def simulate(n_dice, thresh, safe_level, blessed_thresh, cursed_thresh,
             risk_die_thresh, trials, rng, difficulty=0):
    """Run a single pool-size simulation. Returns (marks, complication) 1-D arrays.

    Args:
        thresh: mark threshold (roll >= thresh produces a mark)
        safe_level: 0=off, 1=reroll 1s, 2=reroll 1s and 2s
        blessed_thresh: None=off, or int (e.g. 10, 9, 8) — rolls >= this double
        cursed_thresh: None=off, or int (e.g. 1, 2, 3) — rolls <= this cancel a mark
        risk_die_thresh: 0=off, or int — risk die complications on roll <= this
    """
    rolls = rng.integers(1, 11, size=(trials, n_dice))

    # Safe: reroll dice at or below safe_level once
    if safe_level > 0:
        mask = rolls <= safe_level
        rolls = np.where(mask, rng.integers(1, 11, size=(trials, n_dice)), rolls)

    # Base marks
    m = np.sum(rolls >= thresh, axis=1).astype(np.int64)

    # Blessed: extra mark for each die >= blessed_thresh
    if blessed_thresh is not None:
        m += np.sum(rolls >= blessed_thresh, axis=1)

    # Cursed: subtract a mark for each die <= cursed_thresh
    if cursed_thresh is not None:
        m -= np.sum(rolls <= cursed_thresh, axis=1)

    # Risk Die: first die in the pool is the risk die
    if risk_die_thresh > 0:
        comp = rolls[:, 0] <= risk_die_thresh
    else:
        comp = np.zeros(trials, dtype=bool)

    return m, comp


def simulate_all_pools(thresh, safe_level, blessed_thresh, cursed_thresh,
                       risk_die_thresh, trials, rng, difficulty=0):
    """Simulate pool sizes 1-30 in a single vectorised pass.

    Returns:
        marks: (trials, 30) — marks[:, n-1] = marks for pool size n
        comp:  (trials, 30) — complication bool for each pool size
    """
    max_pool = len(POOL_RANGE)
    rolls = rng.integers(1, 11, size=(trials, max_pool))

    if safe_level > 0:
        mask = rolls <= safe_level
        rolls = np.where(mask, rng.integers(1, 11, size=(trials, max_pool)), rolls)

    cum_hits = np.cumsum(rolls >= thresh, axis=1).astype(np.int64)
    m = cum_hits.copy()

    if blessed_thresh is not None:
        m += np.cumsum(rolls >= blessed_thresh, axis=1)

    if cursed_thresh is not None:
        m -= np.cumsum(rolls <= cursed_thresh, axis=1)

    # Risk die is always column 0 — same result for all pool sizes
    if risk_die_thresh > 0:
        risk_hit = rolls[:, 0] <= risk_die_thresh
        comp = np.broadcast_to(risk_hit[:, np.newaxis], (trials, max_pool)).copy()
    else:
        comp = np.zeros((trials, max_pool), dtype=bool)

    return m, comp


def success_curve(thresh, safe_level, blessed_thresh, cursed_thresh,
                  risk_die_thresh, dr, rng):
    """Return (success%, complication%) arrays for pool sizes 1-30."""
    m, cc = simulate_all_pools(
        thresh, safe_level, blessed_thresh, cursed_thresh,
        risk_die_thresh, COMPARE_TRIALS, rng, dr,
    )
    succ = np.mean((m - dr) >= 0, axis=0) * 100
    comp = np.mean(cc, axis=0) * 100
    return succ, comp


def heatmap_data(thresh, safe_level, blessed_thresh, cursed_thresh,
                 risk_die_thresh, rng):
    """Return a 2D array of success% indexed by [DR, pool_size]."""
    m, _ = simulate_all_pools(
        thresh, safe_level, blessed_thresh, cursed_thresh,
        risk_die_thresh, COMPARE_TRIALS, rng, 0,
    )
    data = np.mean(
        m[np.newaxis, :, :] >= DR_RANGE[:, np.newaxis, np.newaxis], axis=1,
    ) * 100
    return data


def posture_summary_label(threshold, safe_level, blessed_thresh, cursed_thresh,
                          risk_die_thresh):
    """Build a human-readable posture label."""
    parts = []
    if threshold != DEFAULT_THRESHOLD:
        diff = DEFAULT_THRESHOLD - threshold
        if diff > 0:
            parts.append(f"Adv +{diff}")
        else:
            parts.append(f"Disadv {diff}")
    if safe_level == 1:
        parts.append("Safe")
    elif safe_level == 2:
        parts.append("Safe 2")
    if blessed_thresh is not None:
        parts.append(f"Blessed {blessed_thresh}+")
    if cursed_thresh is not None:
        parts.append(f"Cursed {cursed_thresh}")
    if risk_die_thresh > 0:
        parts.append(f"Risk {risk_die_thresh}")
    return " + ".join(parts) if parts else "Normal"
