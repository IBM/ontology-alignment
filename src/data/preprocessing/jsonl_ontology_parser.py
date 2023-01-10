import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Union

import pandas as pd
from data.model.concept import Concept
from data.model.ontology import Ontology
from data.preprocessing.ontology_parser import IOntologyParser


class JSONLOntologyParser(IOntologyParser):

    def parse_from_file(self,
                        input_file: Union[str, Path],
                        concept_member_name_to_column_map: Dict[str, str],
                        missing_member_values: Dict[str, Any]) -> Ontology:
        input_file = Path(input_file)
        assert input_file.exists() and input_file.is_file(
        ) and input_file.suffix == '.jsonl', f"Cannot read Ontology from: {input_file}"

        self.concept_member_name_to_column_map = concept_member_name_to_column_map
        self.missing_member_values = missing_member_values

        logging.info(f"Parsing Ontology from {input_file} ...")
        start_parsing = time.time_ns()
        try:
            start = time.time_ns()
            concepts = self._get_concept_hierarchy(input_file)
            logging.debug(f"Getting all {len(concepts)} Concepts took: {(time.time_ns() - start) * 1e-9} seconds")

            # TODO: Properties, Concept Hierarchies, correct datatypes, optimize speed etc
            onto = Ontology(input_file=input_file,
                            concepts=concepts)
            logging.debug(f"Parsing Ontology took: {(time.time_ns() - start_parsing) * 1e-9} seconds")

        except Exception as e:
            msg = f"Cannot parse Ontology {input_file} because:\n{e}"
            logging.error(msg)
            raise SystemExit(msg)

        return onto

    @staticmethod
    def _jsonl_to_df(input_file: Path) -> pd.DataFrame:
        with open(input_file, 'r') as json_file:
            json_list = list(json_file)
        return pd.DataFrame([json.loads(json_str) for json_str in json_list])

    def _get_concept_hierarchy(self, input_file: Path) -> Dict[str, Concept]:
        df = self._jsonl_to_df(input_file)
        all_concepts: Dict[str, Concept] = {}

        for idx, row in df.iterrows():
            members = {}
            for member_name, column in self.concept_member_name_to_column_map.items():
                if column is None:
                    if member_name == 'name':
                        member_value = f'{self.missing_member_values[member_name]}#{idx}'
                    elif member_name == 'uri':
                        member_value = f'{self.missing_member_values[member_name]}/{idx}'.replace('//', '/')
                    else:
                        member_value = self.missing_member_values[member_name]
                else:
                    if (member_name == 'comments' or member_name == 'labels') and not isinstance(row[column], list):
                        member_value = row[column]
                        if member_value is None:
                            member_value = []
                        else:
                            member_value = [member_value]
                    else:
                        member_value = row[column]

                members[member_name] = member_value
            all_concepts[members['uri']] = Concept(**members)
        return all_concepts
