"""Slot assignment resolution, alias parsing, and reference validation."""

from typing import List, Tuple, Dict, Any
from ccc_krea2.reference_specs import ReferenceChain


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
    resolved: List[Dict[str, Any]] = []
    warnings: List[str] = []

    used_slots = set()
    auto_indices = []

    # Phase 1: Collect manual slots and identify auto references
    for idx, spec in enumerate(chain.references):
        req_slot = spec.requested_vision_slot
        if req_slot is not None and isinstance(req_slot, int) and req_slot > 0:
            if req_slot in used_slots:
                raise ValueError(f"Duplicate vision slot {req_slot} specified in reference chain.")
            used_slots.add(req_slot)
        else:
            auto_indices.append(idx)

    # Phase 2: Assign auto slots consecutively starting from 1
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

    # Phase 3: Expand aliases and validate uniqueness
    global_aliases = set()

    for idx, spec in enumerate(chain.references):
        slot = slot_assignments[idx]
        expanded_aliases = []

        for alias in spec.parsed_aliases:
            exp_alias = alias.replace("{slot}", str(slot))
            if exp_alias in global_aliases:
                warnings.append(f"Duplicate alias '{exp_alias}' resolved across references.")
            global_aliases.add(exp_alias)
            expanded_aliases.append(exp_alias)

            # Check positional alias mismatch
            if alias != exp_alias and f"Image {slot}" not in expanded_aliases:
                warnings.append(f"Reference in slot {slot} has custom alias '{alias}' expanded to '{exp_alias}'.")

        resolved.append({
            "spec": spec,
            "resolved_slot": slot,
            "expanded_aliases": tuple(expanded_aliases)
        })

    return resolved, warnings
