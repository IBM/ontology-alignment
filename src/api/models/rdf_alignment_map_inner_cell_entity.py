# coding: utf-8

from typing import Optional  # noqa: F401

from pydantic import BaseModel, Field


class RDFAlignmentMapInnerCellEntity(BaseModel):
    rdfresource: Optional[str] = Field(alias="rdf:resource", default=None)


RDFAlignmentMapInnerCellEntity.update_forward_refs()
