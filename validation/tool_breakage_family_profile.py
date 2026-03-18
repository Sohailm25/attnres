# ABOUTME: Builds a family-conditioned donor-arm profile from saved Gemma tool-breakage counterfactual checkpoints.
# ABOUTME: Keeps the post-run heterogeneity analysis reusable and prompt-level without launching another model run.

from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Mapping, Sequence

import numpy as np
from scipy.spatial.distance import jensenshannon


def _require_single_subcategory(
    *,
    prompt_id: str,
    tags: Sequence[str],
    composition_tag_prefix: str,
) -> str:
    subcategory_tags = [tag for tag in tags if tag.startswith(composition_tag_prefix)]
    if len(subcategory_tags) != 1:
        raise ValueError(
            f"prompt {prompt_id!r} must define exactly one {composition_tag_prefix} tag"
        )
    return subcategory_tags[0]


def _validated_source_labels(
    counterfactual_prompt_results: Sequence[Mapping[str, object]],
) -> tuple[str, ...]:
    if not counterfactual_prompt_results:
        raise ValueError("counterfactual_prompt_results must not be empty")

    first_baseline = counterfactual_prompt_results[0]["baseline_prompt_result"]
    source_labels = tuple(str(label) for label in first_baseline["source_labels"])
    for result in counterfactual_prompt_results[1:]:
        baseline = result["baseline_prompt_result"]
        candidate = tuple(str(label) for label in baseline["source_labels"])
        if candidate != source_labels:
            raise ValueError("all baseline prompt results must share source labels")
    return source_labels


def _js_distance(left: Sequence[float], right: Sequence[float]) -> float:
    return float(
        jensenshannon(np.asarray(left, dtype=float), np.asarray(right, dtype=float))
    )


def _target_word_count(target_text: str) -> int:
    return len([part for part in target_text.split() if part])


def _prompt_arm_profile(
    *,
    baseline_prompt_result: Mapping[str, object],
    arm_result: Mapping[str, object],
    prompt_lookup: Mapping[str, Mapping[str, object]],
    subcategory_by_prompt_id: Mapping[str, str],
) -> dict[str, object]:
    routed_traces = baseline_prompt_result["layer_traces"]
    arm_traces = arm_result["layer_traces"]
    if len(routed_traces) != len(arm_traces):
        raise ValueError("baseline and control traces must align per prompt")

    tuned_kl_deltas = [
        float(baseline_trace["tuned_routed_mean_kl_to_final"])
        - float(arm_trace["tuned_mean_kl_to_final"])
        for baseline_trace, arm_trace in zip(routed_traces, arm_traces, strict=True)
    ]
    tuned_final_position_kl_deltas = [
        float(baseline_trace["tuned_routed_final_position_kl_to_final"])
        - float(arm_trace["tuned_final_position_kl_to_final"])
        for baseline_trace, arm_trace in zip(routed_traces, arm_traces, strict=True)
    ]
    routed_target_ranks = [
        int(trace["tuned_routed_final_position_target_rank"]) for trace in routed_traces
    ]
    arm_target_ranks = [
        int(trace["tuned_final_position_target_rank"]) for trace in arm_traces
    ]
    alpha_source_prompt_id = arm_result["alpha_source_prompt_id"]
    alpha_source_target_text = None
    alpha_source_subcategory = None
    if alpha_source_prompt_id is not None:
        alpha_source_prompt = prompt_lookup[str(alpha_source_prompt_id)]
        alpha_source_target_text = str(alpha_source_prompt["target_text"])
        alpha_source_subcategory = subcategory_by_prompt_id[str(alpha_source_prompt_id)]

    return {
        "arm_name": str(arm_result["arm_name"]),
        "alpha_source_prompt_id": alpha_source_prompt_id,
        "alpha_source_target_text": alpha_source_target_text,
        "alpha_source_subcategory": alpha_source_subcategory,
        "alpha_js_distance_to_arm": _js_distance(
            baseline_prompt_result["oracle_alpha"],
            arm_result["alpha"],
        ),
        "mean_tuned_kl_delta_routed_minus_arm": float(mean(tuned_kl_deltas)),
        "mean_tuned_final_position_kl_delta_routed_minus_arm": float(
            mean(tuned_final_position_kl_deltas)
        ),
        "final_layer_tuned_final_position_kl_delta_routed_minus_arm": float(
            tuned_final_position_kl_deltas[-1]
        ),
        "final_target_rank_delta_routed_minus_arm": int(
            routed_target_ranks[-1] - arm_target_ranks[-1]
        ),
        "best_target_rank_delta_routed_minus_arm": int(
            min(routed_target_ranks) - min(arm_target_ranks)
        ),
        "target_rank_range_delta_routed_minus_arm": int(
            (max(routed_target_ranks) - min(routed_target_ranks))
            - (max(arm_target_ranks) - min(arm_target_ranks))
        ),
        "routed_worse_final_target_rank_vs_arm": (
            routed_target_ranks[-1] > arm_target_ranks[-1]
        ),
        "routed_worse_best_target_rank_vs_arm": (
            min(routed_target_ranks) > min(arm_target_ranks)
        ),
        "routed_increases_target_rank_range_vs_arm": (
            (max(routed_target_ranks) - min(routed_target_ranks))
            > (max(arm_target_ranks) - min(arm_target_ranks))
        ),
    }


def build_tool_breakage_family_profile_summary(
    *,
    counterfactual_prompt_results: Sequence[Mapping[str, object]],
    prompt_tags_by_id: Mapping[str, Sequence[str]],
    composition_tag_prefix: str = "subcategory_",
) -> dict[str, object]:
    source_labels = _validated_source_labels(counterfactual_prompt_results)
    prompt_lookup = {
        str(result["baseline_prompt_result"]["prompt_id"]): result[
            "baseline_prompt_result"
        ]
        for result in counterfactual_prompt_results
    }
    subcategory_by_prompt_id = {
        prompt_id: _require_single_subcategory(
            prompt_id=prompt_id,
            tags=prompt_tags_by_id[prompt_id],
            composition_tag_prefix=composition_tag_prefix,
        )
        for prompt_id in prompt_lookup
    }

    arm_names = sorted(
        {
            str(arm_result["arm_name"])
            for result in counterfactual_prompt_results
            for arm_result in result["counterfactual_results"]
        }
    )
    for result in counterfactual_prompt_results:
        current_arm_names = sorted(
            str(arm_result["arm_name"])
            for arm_result in result["counterfactual_results"]
        )
        if current_arm_names != arm_names:
            raise ValueError("counterfactual arms must align across prompts")

    prompt_profiles: list[dict[str, object]] = []
    for result in sorted(
        counterfactual_prompt_results,
        key=lambda item: str(item["baseline_prompt_result"]["prompt_id"]),
    ):
        baseline_prompt_result = result["baseline_prompt_result"]
        prompt_id = str(baseline_prompt_result["prompt_id"])
        target_text = str(baseline_prompt_result["target_text"])
        arm_profiles = {
            str(arm_result["arm_name"]): _prompt_arm_profile(
                baseline_prompt_result=baseline_prompt_result,
                arm_result=arm_result,
                prompt_lookup=prompt_lookup,
                subcategory_by_prompt_id=subcategory_by_prompt_id,
            )
            for arm_result in result["counterfactual_results"]
        }
        prompt_profiles.append(
            {
                "prompt_id": prompt_id,
                "prompt": str(baseline_prompt_result["prompt"]),
                "split": str(baseline_prompt_result["split"]),
                "subcategory": subcategory_by_prompt_id[prompt_id],
                "target_text": target_text,
                "target_word_count": _target_word_count(target_text),
                "target_contains_whitespace": (" " in target_text.strip()),
                "arm_profiles": arm_profiles,
            }
        )

    prompt_profiles_by_subcategory: dict[str, list[dict[str, object]]] = defaultdict(
        list
    )
    for prompt_profile in prompt_profiles:
        prompt_profiles_by_subcategory[str(prompt_profile["subcategory"])].append(
            prompt_profile
        )

    by_subcategory: dict[str, dict[str, object]] = {}
    for subcategory, profiles in sorted(prompt_profiles_by_subcategory.items()):
        arm_summaries = {}
        for arm_name in arm_names:
            arm_profiles = [profile["arm_profiles"][arm_name] for profile in profiles]
            arm_summaries[arm_name] = {
                "prompt_count": len(arm_profiles),
                "mean_alpha_js_distance_to_arm": float(
                    mean(
                        float(profile["alpha_js_distance_to_arm"])
                        for profile in arm_profiles
                    )
                ),
                "mean_tuned_kl_delta_routed_minus_arm": float(
                    mean(
                        float(profile["mean_tuned_kl_delta_routed_minus_arm"])
                        for profile in arm_profiles
                    )
                ),
                "mean_tuned_final_position_kl_delta_routed_minus_arm": float(
                    mean(
                        float(
                            profile[
                                "mean_tuned_final_position_kl_delta_routed_minus_arm"
                            ]
                        )
                        for profile in arm_profiles
                    )
                ),
                "mean_final_layer_tuned_final_position_kl_delta_routed_minus_arm": float(
                    mean(
                        float(
                            profile[
                                "final_layer_tuned_final_position_kl_delta_routed_minus_arm"
                            ]
                        )
                        for profile in arm_profiles
                    )
                ),
                "positive_prompt_fraction_tuned_kl_delta_routed_minus_arm": float(
                    mean(
                        1.0
                        if float(profile["mean_tuned_kl_delta_routed_minus_arm"]) > 0.0
                        else 0.0
                        for profile in arm_profiles
                    )
                ),
                "fraction_routed_worse_final_target_rank_vs_arm": float(
                    mean(
                        1.0
                        if bool(profile["routed_worse_final_target_rank_vs_arm"])
                        else 0.0
                        for profile in arm_profiles
                    )
                ),
                "fraction_routed_worse_best_target_rank_vs_arm": float(
                    mean(
                        1.0
                        if bool(profile["routed_worse_best_target_rank_vs_arm"])
                        else 0.0
                        for profile in arm_profiles
                    )
                ),
                "fraction_routed_increases_target_rank_range_vs_arm": float(
                    mean(
                        1.0
                        if bool(profile["routed_increases_target_rank_range_vs_arm"])
                        else 0.0
                        for profile in arm_profiles
                    )
                ),
                "mean_final_target_rank_delta_routed_minus_arm": float(
                    mean(
                        int(profile["final_target_rank_delta_routed_minus_arm"])
                        for profile in arm_profiles
                    )
                ),
                "mean_best_target_rank_delta_routed_minus_arm": float(
                    mean(
                        int(profile["best_target_rank_delta_routed_minus_arm"])
                        for profile in arm_profiles
                    )
                ),
                "mean_target_rank_range_delta_routed_minus_arm": float(
                    mean(
                        int(profile["target_rank_range_delta_routed_minus_arm"])
                        for profile in arm_profiles
                    )
                ),
            }

        target_word_counts = [int(profile["target_word_count"]) for profile in profiles]
        by_subcategory[subcategory] = {
            "prompt_count": len(profiles),
            "multiword_target_fraction": float(
                mean(
                    1.0 if word_count > 1 else 0.0 for word_count in target_word_counts
                )
            ),
            "mean_target_word_count": float(mean(target_word_counts)),
            "arm_summaries": arm_summaries,
        }

    return {
        "num_prompts": len(prompt_profiles),
        "source_labels": source_labels,
        "arm_names": arm_names,
        "prompt_profiles": prompt_profiles,
        "by_subcategory": by_subcategory,
    }
