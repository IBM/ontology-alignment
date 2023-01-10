# coding: utf-8
from typing import List, Optional

from pydantic import BaseModel, Field
from api.models.rdf_alignment_map_inner import RDFAlignmentMapInner


class RDFAlignment(BaseModel):
    xml: Optional[str] = Field(alias="xml", default=None)
    level: Optional[str] = Field(alias="level", default=None)
    type: Optional[str] = Field(alias="type", default=None)
    map: Optional[List[RDFAlignmentMapInner]] = Field(alias="map", default=None)


RDFAlignment.update_forward_refs()
