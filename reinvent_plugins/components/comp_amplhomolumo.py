"""AMPL Prediction Model

This component uses the AMPL (Accelerating Therapeutics for Opportunities in Medicine Machine Learning) model 
to predict properties of molecules.
"""

from __future__ import annotations

__all__ = ["AMPLPredictionModelhomolumo"]

from typing import List
import os
import pandas as pd
import numpy as np

from rdkit import Chem
from .amplscorer_homolumo import ampl_pred_model
from ..component_results import ComponentResults
from reinvent_plugins.mol_cache import molcache
from ..add_tag import add_tag
from pydantic.dataclasses import dataclass


ampl_image = "/home/overhulsejm/ampl16_sing/ampl.sif"
container_type = "singularity"
target_col_name =  "HOMO_LUMO_gap"
working_directory = "/home/overhulsejm/reinvent4/REINVENT4/contrib/reinvent_plugins/components/ampl2"

@add_tag("__parameters")
@dataclass
class Parameters:
    """Parameters for the scoring component

    Note that all parameters are always lists because components can have
    multiple endpoints and so all the parameters from each endpoint is
    collected into a list.  This is also true in cases where there is only one
    endpoint.
    """

    working_directory: List[str]  # should we type hint paths? List[Path]
    ampl_image: List[str]
    container_type: List[str]
    target_col_name: List[str]



@add_tag("__component")
class AMPLPredictionModelhomolumo:
    def __init__(self, params:Parameters):
        self.working_directory = params.working_directory[0]
        self.ampl_image = params.ampl_image[0]
        self.container_type = params.container_type[0]
        self.target_col_name = params.target_col_name[0]

        if self.container_type == 'singularity':
            os.system("module load singularity")

    @molcache
    def __call__(self, mols: List[Chem.Mol]) -> np.ndarray:
        # Convert molecules to SMILES
        smiles_list = [Chem.MolToSmiles(mol) for mol in mols]
        
        # Create a DataFrame
        population = pd.DataFrame({'SMILES': smiles_list})

        # Save unscored population
        population.to_csv(f"{self.working_directory}/unscored_population.csv", index=False)

        print("entering container")
        if self.container_type == 'singularity':
            os.system(f"singularity exec --bind {self.working_directory}:/data {self.ampl_image} /data/run_inference.sh")
       # elif self.container_type == 'docker':
        #    os.system(f"docker run -v {self.working_directory}:/data {self.ampl_image} /data/run_inference.sh")
        print("container complete")

        # Read scored population
        scored_population = pd.read_csv(f"{self.working_directory}/scored_population.csv")

        # Extract scores
        scores = scored_population[f'{self.target_col_name}_pred'].values

        return ComponentResults([np.array(scores)])