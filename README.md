# snomed_entity_linking
Entity Linking from Clinical Notes to SNOMED Ontology

Entity Linking Pipeline has 2 steps - 
* A "Clinical Entity Recognizer" (CER) that is responsible for detecting candidate clinical entities from within a text. The CER Model is developed by finetuning deberta-v3-base using LoRA
* A "Linker" that is responsible for "linking" entities detected by the CER to concepts in the knowledge base. Often (as here) the linker's tasks are split into two steps:
  * In the Candidate Generation step, the Linker retrieves a handful of candidate concepts that it thinks may match the entity.
  * In the Candidate Selection step, the Linker selects the best candidate

Hardware Requirement - A GPU machine with at least 24GB of VRAM

Data Requirements - The Pipeline leverages 2 key datasets :
* SNOMED Graph Ontology
* Annotated Clinical Notes from MIMIC IV dataset

Training CER Model - train_cer.py
Training Linker Model - train_linker.py
Inference Pipeline - main.py


 
