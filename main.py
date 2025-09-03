# Import necessary libraries
import pandas as pd
import numpy as np
import os
import multiprocessing
import time

import pyterrier as pt
from pyterrier_dr import FlexIndex, RetroMAE, SBertBiEncoder

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

from transformers import pipeline

# Initialize dataset
ground_truth_train = pd.read_pickle('ground_truth_train.pkl')
ground_truth_test = pd.read_pickle('ground_truth_test.pkl')
docs = pd.read_pickle('docs.pkl')

# Setting PyTerrier and ChromaDB


# Indexing


# Retrieval



# Evaluate Retrieval