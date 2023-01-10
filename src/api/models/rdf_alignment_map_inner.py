# coding: utf-8
from typing import Optional

from pydantic import BaseModel, Field
from api.models.rdf_alignment_map_inner_cell import RDFAlignmentMapInnerCell


class RDFAlignmentMapInner(BaseModel):
    cell: Optional[RDFAlignmentMapInnerCell] = Field(alias="Cell", default=None)


RDFAlignmentMapInner.update_forward_refs()
