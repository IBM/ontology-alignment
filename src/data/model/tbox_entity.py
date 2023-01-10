from dataclasses import dataclass, field

from typing import List, Dict


@dataclass
class TBoxEntity:
    name: str
    uri: str

    comments: List[str] = field(default_factory=list)
    labels: List[str] = field(default_factory=list)

    parents: Dict[str, "TBoxEntity"] = field(default_factory=dict)
    children: Dict[str, "TBoxEntity"] = field(default_factory=dict)

    def is_root(self):
        return len(self.parents) == 0

    def is_leaf(self):
        return len(self.children) == 0
