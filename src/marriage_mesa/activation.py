"""Mesa 3 activation adapter for NetLogo's ``ask agentset`` semantics."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any

from mesa.agent import AgentSet

if TYPE_CHECKING:
    import mesa


class NetLogoActivation:
    """Create a snapshot and activate it in a fresh random order for every ask.

    Mesa 3 removed the legacy scheduler classes.  ``AgentSet.shuffle_do`` is
    the supported equivalent of NetLogo's randomized ``ask``.  Rebuilding the
    AgentSet on each call also preserves NetLogo's snapshot semantics when an
    action creates or removes agents.
    """

    def __init__(self, model: "mesa.Model") -> None:
        self.model = model

    def snapshot(self, agents: Iterable[Any]) -> AgentSet:
        return AgentSet(list(agents), random=self.model.random)

    def select(self, predicate: Callable[[Any], bool]) -> AgentSet:
        return self.snapshot(agent for agent in self.model.agents if predicate(agent))

    def ask(
        self,
        agents: Iterable[Any],
        action: str | Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> AgentSet:
        snapshot = self.snapshot(agents)
        snapshot.shuffle_do(action, *args, **kwargs)
        return snapshot
