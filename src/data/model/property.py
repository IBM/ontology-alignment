from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Union, Any, Optional

from data.model.tbox_entity import TBoxEntity

if TYPE_CHECKING:
    from data.model.concept import Concept


@dataclass
class Property(TBoxEntity):
    domain: List["Concept"] = field(default_factory=list)
    range: List[Union["Concept", Any]] = field(default_factory=list)

    inverse: Optional["Property"] = field(default=None)

    def __str__(self) -> str:
        from data.model.concept import Concept  # avoid circular import by local import
        return (
            f"Property('name': {self.name}, 'uri': {self.uri}, "
            f"'comments': {[c for c in self.comments]}, "
            f"'labels': {[l for l in self.labels]}, "
            f"'domain': {[d.name if isinstance(d, Concept) else str(d) for d in self.domain]}, "
            f"'range': {[r.name if isinstance(r, Concept) else str(r) for r in self.range]}, "
            f"'parents': {[p.name for p in self.parents.values()]}), "
            f"'children': {[c.name for c in self.children.values()]})"
        )

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, Property):
            return False
        elif self.uri != o.uri:
            return False

        from data.model.concept import Concept  # avoid circular import by local import
        domain_eq = sorted(self.domain, key=lambda d: d.name if isinstance(d, Concept) else str(d)) == \
            sorted(o.domain, key=lambda d: d.name if isinstance(d, Concept) else str(d))
        range_eq = sorted(self.domain, key=lambda d: d.name if isinstance(d, Concept) else str(d)) == \
            sorted(o.domain, key=lambda d: d.name if isinstance(d, Concept) else str(d))

        return (domain_eq and
                range_eq and
                self.inverse == o.inverse and
                self.parents.keys() == o.parents.keys() and
                self.children.keys() == o.children.keys())
