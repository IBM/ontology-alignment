# coding: utf-8
from typing import Optional

from pydantic import BaseModel, Field
from api.models.rdf_alignment import RDFAlignment


class RDFAlignmentFormat(BaseModel):
    alignment: Optional[RDFAlignment] = Field(alias="Alignment", default=None)


RDFAlignmentFormat.update_forward_refs()
