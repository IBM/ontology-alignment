
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict

from data.model.tbox_entity import TBoxEntity


if TYPE_CHECKING:
    from data.model.property import Property


@dataclass
class Concept(TBoxEntity):
    outgoing_properties: Dict[str, "Property"] = field(default_factory=dict)
    incoming_properties: Dict[str, "Property"] = field(default_factory=dict)

    def __str__(self) -> str:
        return (
            f"Concept('name': {self.name}, 'uri': {str(self.uri)}, "
            f"'comments': {[c for c in self.comments]}, "
            f"'labels': {[l for l in self.labels]}, "
            f"'parents': {[p.name for p in self.parents.values()]}, "
            f"'children': {[c.name for c in self.children.values()]}, "
            f"'outgoing_properties': {[p.name for p in self.outgoing_properties.values()]}, "
            f"'incoming_properties': {[p.name for p in self.incoming_properties.values()]})"
        )

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, Concept):
            return False
        return (self.uri == o.uri and
                self.parents.keys() == o.parents.keys() and
                self.children.keys() == o.children.keys() and
                self.outgoing_properties.keys() == o.outgoing_properties.keys() and
                self.incoming_properties.keys() == o.incoming_properties.keys() and
                set(self.comments) == set(o.comments) and
                set(self.labels) == set(o.labels))
