"""The terminology store.

Spec reference: §7.7, §8.4.3.

Technical vocabulary must survive intact in both directions. "Deployment"
stays "deployment"; "pull request" stays "pull request". A user-editable
store, persisted in SQLite and loaded at session start, feeds three places at
once:

    1. recogniser keyterm prompting  - improves recognition of exactly the
       words that matter most, which is measurably large (§8.5 step 3)
    2. the translation prompt        - stops the LLM helpfully translating
       "production" into something nobody says
    3. pronunciation hints           - where the voice engine supports them

The highest-cost translation errors in a business meeting are not awkward
phrasing; they are a wrong number, a wrong date or a mangled name. This file
plus the number-handling instruction in the translation prompt is the defence.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_LOCKED = [
    "deployment", "deploy", "staging", "production", "pull request",
    "merge request", "sprint", "backlog", "standup", "retro", "ticket",
    "API", "endpoint", "repository", "repo", "branch", "commit", "rollback",
    "Kubernetes", "Docker", "Jira", "Azure", "AWS", "database", "schema",
    "migration", "frontend", "backend", "pipeline", "build", "release",
    "milestone", "go live", "scope", "requirement", "integration",
    "software", "job", "developer", "engineer", "remote", "freelance",
    "office", "manager", "project", "client", "meeting", "company", "Pakistan",
]


#: Locked for TRANSLATION, useless as a recogniser hint. Keyterm prompting
#: biases the model towards a word; biasing it towards words it already knows
#: and that are common in ordinary speech buys nothing and costs two things -
#: budget (the cap is counted in tokens across the whole request, and these
#: were filling it ahead of the names that actually get mangled) and
#: precision, because a model pushed towards "job" will hear "job" in "jab".
#:
#: They stay in `locked`, where they belong: the point of "production" being
#: locked is that the TRANSLATOR must not render it as something else.
NOT_WORTH_HINTING = {
    "deploy", "deployment", "staging", "production", "sprint", "backlog",
    "ticket", "branch", "commit", "build", "release", "scope", "software",
    "job", "developer", "engineer", "remote", "freelance", "office",
    "manager", "project", "client", "meeting", "company", "database",
    "schema", "migration", "frontend", "backend", "pipeline", "endpoint",
    "repository", "repo", "integration", "requirement", "milestone",
    "go live", "standup", "retro", "rollback",
}


@dataclass
class Terminology:
    """Terms that must not be translated, entities, and forced renderings."""

    version: int = 1
    locked: list[str] = field(default_factory=lambda: list(DEFAULT_LOCKED))
    entities: list[str] = field(default_factory=list)
    preferred: dict[str, str] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, path: str | Path) -> "Terminology":
        p = Path(path)
        if not p.exists():
            return cls()
        data = json.loads(p.read_text(encoding="utf-8"))
        return cls(
            version=int(data.get("version", 1)),
            locked=list(data.get("locked", DEFAULT_LOCKED)),
            entities=list(data.get("entities", [])),
            preferred=dict(data.get("preferred", {})),
        )

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(
                {
                    "version": self.version,
                    "locked": self.locked,
                    "entities": self.entities,
                    "preferred": self.preferred,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    # ------------------------------------------------------------------
    # the three consumers
    # ------------------------------------------------------------------

    def keyterms(self, limit: int = 100) -> list[str]:
        """Terms to hand the recogniser as keyterm/phrase hints.

        Entities first: a mangled client name is more damaging than a mangled
        piece of jargon, and providers cap the list.
        """
        out: list[str] = []
        seen: set[str] = set()
        for term in [*self.entities, *self.preferred.keys(), *self.locked]:
            k = term.lower()
            if k in seen:
                continue
            seen.add(k)
            # Entities are always worth a hint - a name is exactly what
            # keyterm prompting is for. Generic vocabulary is not, however
            # firmly it is locked against translation.
            if k in NOT_WORTH_HINTING and term not in self.entities:
                continue
            out.append(term)
            if len(out) >= limit:
                break
        return out

    def prompt_fragment(self) -> str:
        """The block injected into the translation prompt."""
        parts: list[str] = []
        if self.locked:
            parts.append(
                "Carry these terms through EXACTLY as written, never "
                "translated or transliterated: " + ", ".join(sorted(set(self.locked)))
            )
        if self.entities:
            parts.append(
                "These are proper nouns; reproduce them exactly: "
                + ", ".join(sorted(set(self.entities)))
            )
        if self.preferred:
            pairs = "; ".join(f"{k} -> {v}" for k, v in sorted(self.preferred.items()))
            parts.append("Use these renderings: " + pairs)
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # editing
    # ------------------------------------------------------------------

    def lock(self, term: str) -> None:
        if term and term.lower() not in {t.lower() for t in self.locked}:
            self.locked.append(term)

    def add_entity(self, name: str) -> None:
        if name and name.lower() not in {e.lower() for e in self.entities}:
            self.entities.append(name)

    def prefer(self, source: str, target: str) -> None:
        self.preferred[source] = target

    def remove(self, term: str) -> None:
        low = term.lower()
        self.locked = [t for t in self.locked if t.lower() != low]
        self.entities = [e for e in self.entities if e.lower() != low]
        self.preferred.pop(term, None)

    # ------------------------------------------------------------------

    def is_locked(self, term: str) -> bool:
        low = term.lower()
        return any(t.lower() == low for t in self.locked)

    def find_in(self, text: str) -> list[str]:
        """Which locked terms and entities appear in `text`.

        Used to narrow the prompt fragment for a short unit, and to highlight
        entities in the transcript view.
        """
        hits: list[str] = []
        for term in [*self.entities, *self.locked]:
            pattern = r"\b" + re.escape(term) + r"\b"
            if re.search(pattern, text, flags=re.IGNORECASE):
                hits.append(term)
        return hits

    def __len__(self) -> int:
        return len(self.locked) + len(self.entities) + len(self.preferred)
