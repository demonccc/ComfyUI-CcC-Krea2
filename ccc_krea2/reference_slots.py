"""Slot assignment resolution, alias parsing, and reference validation."""

from typing import List, Tuple, Dict, Any
from .reference_specs import ReferenceChain


from .style_processing import get_style_processing_image_count


def parse_aliases(raw_aliases: str) -> Tuple[str, ...]:
    """Parse comma-separated alias strings, trimming whitespace and discarding duplicates."""
    if not raw_aliases:
        return ()

    tokens = [t.strip() for t in raw_aliases.split(",")]
    seen = set()
    result = []
    for token in tokens:
        if token and token not in seen:
            seen.add(token)
            result.append(token)

    return tuple(result)


def resolve_reference_slots_and_aliases(chain: ReferenceChain) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Resolve vision slots and expand aliases across all references in a chain.

    Returns:
        (resolved_reference_dicts, warnings)
    """
    resolved_unordered: List[Dict[str, Any]] = []
    warnings: List[str] = []

    used_slots = set()

    # Phase 1: Collect manual slots and check for duplicates among manual slots
    for idx, spec in enumerate(chain.references):
        req_slot = spec.requested_vision_slot
        if req_slot is not None and isinstance(req_slot, int) and req_slot > 0:
            if req_slot in used_slots:
                raise ValueError(f"Duplicate vision slot {req_slot} specified in reference chain.")
            used_slots.add(req_slot)

    # Phase 2: Assign auto slots consecutively starting from 1 into unused numbers
    next_auto_slot = 1
    slot_assignments = [None] * len(chain.references)

    for idx, spec in enumerate(chain.references):
        req_slot = spec.requested_vision_slot
        if req_slot is not None and isinstance(req_slot, int) and req_slot > 0:
            slot_assignments[idx] = req_slot
        else:
            while next_auto_slot in used_slots:
                next_auto_slot += 1
            slot_assignments[idx] = next_auto_slot
            used_slots.add(next_auto_slot)
            next_auto_slot += 1

    # Build initial list with assigned logical slots
    for idx, spec in enumerate(chain.references):
        resolved_unordered.append({"spec": spec, "resolved_slot": slot_assignments[idx]})

    # Phase 3: Sort references strictly by resolved logical slot
    resolved_sorted = sorted(resolved_unordered, key=lambda item: item["resolved_slot"])

    # Phase 4: Validate that resolved logical slots are consecutive 1..N with no gaps
    resolved_slots_list = [item["resolved_slot"] for item in resolved_sorted]
    expected_slots = list(range(1, len(resolved_sorted) + 1))
    if resolved_slots_list != expected_slots:
        raise ValueError(
            f"Invalid vision slot configuration: gaps detected in resolved logical slots "
            f"(found {resolved_slots_list}, expected consecutive slots 1..{len(resolved_sorted)}). "
            f"Physical Qwen images cannot contain an empty logical slot."
        )

    # Phase 3.5: Enforce non-Style references before Style
    seen_style = False
    for item in resolved_sorted:
        ref_path = getattr(item["spec"], "reference_path", item["spec"].role.lower())
        if ref_path == "style" or item["spec"].role.lower() == "style":
            seen_style = True
        elif seen_style:
            raise ValueError(
                "Invalid reference chain order: Style references expand into multiple physical Qwen images; "
                "placing Style before edit references makes Image N aliases ambiguous; "
                "place all edit references before Style."
            )

    # Phase 5: Expand aliases and assign physical indices/ranges in sorted logical slot order
    resolved: List[Dict[str, Any]] = []
    global_aliases = set()
    vae_frame_counter = 1
    physical_qwen_index = 1

    for item in resolved_sorted:
        spec = item["spec"]
        slot = item["resolved_slot"]
        expanded_aliases = []

        parsed_aliases = spec.parsed_aliases
        if not parsed_aliases and spec.alias:
            parsed_aliases = parse_aliases(spec.alias)

        for alias in parsed_aliases:
            exp_alias = alias.replace("{slot}", str(slot))
            if exp_alias in global_aliases:
                raise ValueError(f"Duplicate alias '{exp_alias}' specified across references in chain.")
            global_aliases.add(exp_alias)
            expanded_aliases.append(exp_alias)

            # Check literal Image N alias vs physical Qwen index mismatch
            if exp_alias.startswith("Image ") and exp_alias[6:].isdigit():
                lit_num = int(exp_alias[6:])
                if lit_num != physical_qwen_index:
                    raise ValueError(
                        f"Conflicting literal positional alias '{exp_alias}': physical Qwen index is {physical_qwen_index}."
                    )

        ref_path = getattr(spec, "reference_path", spec.role.lower())
        is_appearance = getattr(spec, "appearance_reference", True)

        include_in_vision = getattr(spec, "include_in_vision", True)

        if not include_in_vision:
            if ref_path == "style" or spec.role.lower() == "style":
                raise ValueError("Style references must participate in Qwen Vision conditioning.")
            vae_frame = vae_frame_counter if is_appearance else None
            if is_appearance:
                vae_frame_counter += 1
            physical_qwen_range = None
        elif ref_path == "style" or spec.role.lower() == "style":
            vae_frame = None
            style_proc = getattr(spec, "style_processing", "2x2")
            span_len = get_style_processing_image_count(style_proc)
            physical_qwen_range = (physical_qwen_index, physical_qwen_index + span_len - 1)
            physical_qwen_index += span_len
        elif not is_appearance:
            vae_frame = None
            physical_qwen_range = (physical_qwen_index, physical_qwen_index)
            physical_qwen_index += 1
        else:
            vae_frame = vae_frame_counter
            vae_frame_counter += 1
            physical_qwen_range = (physical_qwen_index, physical_qwen_index)
            physical_qwen_index += 1

        resolved.append(
            {
                "spec": spec,
                "resolved_slot": slot,
                "logical_reference_id": slot,
                "logical_role": getattr(spec, "reference_path", spec.role.lower()),
                "logical_vision_slot": slot,
                "expanded_aliases": tuple(expanded_aliases),
                "vae_reference_frame": vae_frame,
                "physical_qwen_range": physical_qwen_range,
            }
        )

    return resolved, warnings
