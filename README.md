# HyperCOCO
HyperCOCO: Multi-sensory Hyper COgnitive COmputing for Learning Population-level Brain Connectivity
HyperCOCO is an extension of mCOCO that learns a high-order connectional brain template (CBT) from BOLD signals and endows it with cognitive memory traits using multi-sensory stimulation.

This work introduces a hyperconnectivity-aware framework to move beyond pairwise connectivity and better model complex population-level brain interactions.

Please contact mayssa.soussia@gmail.com for inquiries.
![Main Figure](Main_Figure.png)


# Introduction
This work  has been accepted in main MICCAI 2025, Daejon, South Korea. 




#Introduction
This work  has been accepted in Medical Image Analysis Journal 2026
> **HyperCOCO: Multi-sensory Hyper COgnitive COmputing for Learning Population Level Brain Connectivity
>
> Mayssa Soussia, Mohamed Ali Mahjoub, Islem Rekik
>
> BASIRA Lab, Imperial-X and Department of Computing, Imperial College London, UK
>
> LATIS Lab, National Engineering School of Sousse, University of Sousse, Tunisia
>
> **Abstract:** *Learning a high-order connectional brain template (CBT) endowed with cognitive capacities such as visual or auditory memory is crucial for iden- tifying cognition-related biomarkers and distinguishing between control and clinical populations. Higher-order CBTs provide a population-level repre- sentation that captures not only structural or topological regularities but also the multi-regional interactions and cognitive processes that conventional pairwise models fail to reflect. Because the brain operates through complex, coordinated dynamics, estimating CBTs that incorporate such higher-order and cognitively meaningful organization is essential for advancing our under- standing of neural function and dysfunction. While recent machine-learning and graph-neural-network approaches have improved CBT estimation, they remain limited by their focus on pairwise interactions and purely structural features, overlooking both higher-order organization and cognitive proper- ties. This gap raises a central question: How can we learn a high-order CBT that is well-centered at the population level and also endowed with cogni- tive capacities? We tackle this challenge using reservoir computing (RC), a biologically inspired framework that mimics how the brain processes infor- mation. RC exhibits dynamic properties similar to those of the prefrontal cortex, an area associated with working memory and features a fading mem- ory mechanism, known as the Echo State Property (ESP), which mirrors the brain’s short-term memory function. Building on these properties, we introduce HyperCOCO, a novel framework for generating high-order cogni- tively enhanced CBTs in two stages. First, BOLD signals are processed through a random reservoir to generate high-order individual functional con- nectomes, which are then aggregated into a population-level template. Sec- ond, this template is instantiated into a hyper-cognitive reservoir and stimu- lated with multi-sensory inputs (visual, auditory, and linguistic). Finally, we measure the memory capacity of the resulting CBT as a proxy for its ability to encode and retain cognitive information.*

## Key Contributions

- **HyperCognitive Templates**: CBTs are no longer static structural averages; they're endowed with memory and dynamic behavior.
- **Reservoir Computing**: Models hyper-cognitive reservoir instantiated with high-order functional connectome.
- **Multi-Sensory Inputs**: Evaluates CBTs using sensory streams (MNIST, speech, language) to evaluate memory capacity across different groups.
- **Population-Level Learning**: Builds a group-level hyper-CBT template from ASD and ADHD subjects using cross-validation.


## Dataset Description

The dataset "ABIDE_subset" provided in this repository is a portion of the full dataset used in our study. It consists of preprocessed functional data from:

- **73 individuals** with Autism Spectrum Disorder (ASD)
- **77 typically developing (TD)** controls
  
To access the complete dataset, please refer to the ABIDE (Autism Brain Imaging Data Exchange) website: [https://fcon_1000.projects.nitrc.org/indi/abide/](https://fcon_1000.projects.nitrc.org/indi/abide/)



## Project Structure

### 1. HyperCBT Generation with Reservoir Computing

- **File**: `Hypercbt_generation.py`
- **Goal**: Generate population-level HyperCBTs using RC

### 2. Multi-Sensory Memory Capacity Evaluation

- **File**: `multi_sensory_memory_capacity.py`
- **Goal**: Validate the memory/cognitive capacity of HyperCBTs using multi-sensory streams (visual, audio, language).

#### Sensory inputs: 
- **Visual**: we used MNIST dataset to meansure the visual memory capacity of the cognitive reservoir.
- **Audio**: We used two types of audio streams: a Beethoven soundtracck and a quranic recitation to measure the auditory memory of the cognitive reservoir. 
- **Language**: We used Gutenberg embeddings to measure the linguistic memory capacity of the cognitive reservoir. 

## Installation

Ensure you have Python 3.x installed. To install the required dependencies, use

```bash
pip install -r requirements.txt
```


## Run the Pipeline

### 1. CBT Generation
```bash
python cbt_generation.py
```
### 1. CBT Generation
```bash
python multi_sensory_memory_capacity.py
```


## Citation 
if you use this work, please cite:
```bibtex

@article{soussia2026hypercoco,
  title={HyperCOCO: Multi-sensory Hyper COgnitive COmputing for Learning Population Level Brain Connectivity},
  author={Soussia, Mayssa and Mahjoub, Mohamed Ali and Rekik, Islem},
  journal={Medical Image Analysis},
  year={2026},
  note={Accepted}
}


