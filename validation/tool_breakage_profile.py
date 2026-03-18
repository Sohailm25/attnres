# ABOUTME: Builds saved-artifact baseline profiles for Gemma tool-breakage runs grouped by prompt tags.
# ABOUTME: Keeps route-mode and family summaries reusable without relaunching the model run.

from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Mapping, Sequence


def _require_single_tag(
    *,
    prompt_id: str,
    tags: Sequence[str],
    tag_prefix: str,
) -> str:
    matching_tags = [tag for tag in tags if tag.startswith(tag_prefix)]
    if len(matching_tags) != 1:
        raise ValueError(
            f"prompt {prompt_id!r} must define exactly one {tag_prefix} tag"
        )
    return matching_tags[0]


def _mean(values: Sequence[float]) -> float:
    return float(mean(values))


def _target_ranks(
    prompt_result: Mapping[str, object],
    *,
    lens_name: str,
    routed: bool,
) -> list[int]:
    traces = prompt_result["layer_traces"]
    if lens_name == "tuned" and not routed:
        return [
            int(trace["tuned_original_final_position_target_rank"]) for trace in traces
        ]
    if lens_name == "tuned" and routed:
        return [
            int(trace["tuned_routed_final_position_target_rank"]) for trace in traces
        ]
    raise ValueError(f"unsupported lens name {lens_name!r}")


def _prompt_profile(
    *,
    prompt_result: Mapping[str, object],
    tags_by_prefix: Mapping[str, str],
) -> dict[str, object]:
    traces = prompt_result["layer_traces"]
    if not traces:
        raise ValueError("prompt_result layer_traces must not be empty")

    tuned_kl_deltas = [
        float(trace["tuned_routed_mean_kl_to_final"])
        - float(trace["tuned_original_mean_kl_to_final"])
        for trace in traces
    ]
    tuned_final_position_kl_deltas = [
        float(trace["tuned_routed_final_position_kl_to_final"])
        - float(trace["tuned_original_final_position_kl_to_final"])
        for trace in traces
    ]
    tuned_original_ranks = _target_ranks(
        prompt_result,
        lens_name="tuned",
        routed=False,
    )
    tuned_routed_ranks = _target_ranks(
        prompt_result,
        lens_name="tuned",
        routed=True,
    )

    return {
        "prompt_id": str(prompt_result["prompt_id"]),
        "prompt": str(prompt_result["prompt"]),
        "split": str(prompt_result["split"]),
        "target_text": str(prompt_result["target_text"]),
        "tags_by_prefix": dict(tags_by_prefix),
        "mean_tuned_kl_increase_under_routing": _mean(tuned_kl_deltas),
        "mean_tuned_final_position_kl_increase_under_routing": _mean(
            tuned_final_position_kl_deltas
        ),
        "final_target_rank_delta_under_routing": float(
            tuned_routed_ranks[-1] - tuned_original_ranks[-1]
        ),
        "best_target_rank_delta_under_routing": float(
            min(tuned_routed_ranks) - min(tuned_original_ranks)
        ),
        "target_rank_range_delta_under_routing": float(
            (max(tuned_routed_ranks) - min(tuned_routed_ranks))
            - (max(tuned_original_ranks) - min(tuned_original_ranks))
        ),
        "tuned_routing_worsens_final_target_rank": (
            tuned_routed_ranks[-1] > tuned_original_ranks[-1]
        ),
        "tuned_routing_worsens_best_target_rank": (
            min(tuned_routed_ranks) > min(tuned_original_ranks)
        ),
        "tuned_routing_increases_target_rank_range": (
            (max(tuned_routed_ranks) - min(tuned_routed_ranks))
            > (max(tuned_original_ranks) - min(tuned_original_ranks))
        ),
    }


def build_tool_breakage_baseline_tag_profile_summary(
    *,
    prompt_results: Sequence[Mapping[str, object]],
    prompt_tags_by_id: Mapping[str, Sequence[str]],
    group_tag_prefixes: Sequence[str] = ("subcategory_", "route_mode_"),
) -> dict[str, object]:
    if not prompt_results:
        raise ValueError("prompt_results must not be empty")
    if not group_tag_prefixes:
        raise ValueError("group_tag_prefixes must not be empty")

    prompt_profiles: list[dict[str, object]] = []
    grouped_profiles: dict[str, dict[str, list[dict[str, object]]]] = {
        tag_prefix: defaultdict(list) for tag_prefix in group_tag_prefixes
    }

    for prompt_result in sorted(
        prompt_results, key=lambda item: str(item["prompt_id"])
    ):
        prompt_id = str(prompt_result["prompt_id"])
        if prompt_id not in prompt_tags_by_id:
            raise ValueError(f"prompt_tags_by_id is missing prompt {prompt_id!r}")
        tags_by_prefix = {
            tag_prefix: _require_single_tag(
                prompt_id=prompt_id,
                tags=prompt_tags_by_id[prompt_id],
                tag_prefix=tag_prefix,
            )
            for tag_prefix in group_tag_prefixes
        }
        profile = _prompt_profile(
            prompt_result=prompt_result,
            tags_by_prefix=tags_by_prefix,
        )
        prompt_profiles.append(profile)
        for tag_prefix, tag_value in tags_by_prefix.items():
            grouped_profiles[tag_prefix][tag_value].append(profile)

    group_summaries = {}
    for tag_prefix, profiles_by_tag in grouped_profiles.items():
        group_summaries[tag_prefix] = {
            "by_tag": {
                tag_value: {
                    "prompt_count": len(profiles),
                    "mean_tuned_kl_increase_under_routing": _mean(
                        [
                            float(profile["mean_tuned_kl_increase_under_routing"])
                            for profile in profiles
                        ]
                    ),
                    "mean_tuned_final_position_kl_increase_under_routing": _mean(
                        [
                            float(
                                profile[
                                    "mean_tuned_final_position_kl_increase_under_routing"
                                ]
                            )
                            for profile in profiles
                        ]
                    ),
                    "positive_prompt_fraction_tuned_kl_increase_under_routing": _mean(
                        [
                            1.0
                            if float(profile["mean_tuned_kl_increase_under_routing"])
                            > 0.0
                            else 0.0
                            for profile in profiles
                        ]
                    ),
                    "fraction_tuned_routing_worsens_final_target_rank_prompts": _mean(
                        [
                            1.0
                            if bool(profile["tuned_routing_worsens_final_target_rank"])
                            else 0.0
                            for profile in profiles
                        ]
                    ),
                    "fraction_tuned_routing_worsens_best_target_rank_prompts": _mean(
                        [
                            1.0
                            if bool(profile["tuned_routing_worsens_best_target_rank"])
                            else 0.0
                            for profile in profiles
                        ]
                    ),
                    "fraction_tuned_routing_increases_target_rank_range_prompts": _mean(
                        [
                            1.0
                            if bool(
                                profile["tuned_routing_increases_target_rank_range"]
                            )
                            else 0.0
                            for profile in profiles
                        ]
                    ),
                    "mean_final_target_rank_delta_under_routing": _mean(
                        [
                            float(profile["final_target_rank_delta_under_routing"])
                            for profile in profiles
                        ]
                    ),
                    "mean_best_target_rank_delta_under_routing": _mean(
                        [
                            float(profile["best_target_rank_delta_under_routing"])
                            for profile in profiles
                        ]
                    ),
                    "mean_target_rank_range_delta_under_routing": _mean(
                        [
                            float(profile["target_rank_range_delta_under_routing"])
                            for profile in profiles
                        ]
                    ),
                }
                for tag_value, profiles in sorted(profiles_by_tag.items())
            }
        }

    return {
        "num_prompts": len(prompt_profiles),
        "group_tag_prefixes": list(group_tag_prefixes),
        "prompt_profiles": prompt_profiles,
        "group_summaries": group_summaries,
    }
