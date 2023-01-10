# coding: utf-8
from typing import Optional

from pydantic import BaseModel, Field
from api.models.rdf_alignment_map_inner_cell_entity import RDFAlignmentMapInnerCellEntity


class RDFAlignmentMapInnerCell(BaseModel):
    entity1: Optional[RDFAlignmentMapInnerCellEntity] = Field(alias="entity1", default=None)
    entity2: Optional[RDFAlignmentMapInnerCellEntity] = Field(alias="entity2", default=None)
    relation: Optional[str] = Field(alias="relation", default=None)
    measure: Optional[float] = Field(alias="measure", default=None)


RDFAlignmentMapInnerCell.update_forward_refs()
