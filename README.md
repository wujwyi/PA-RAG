# PA-RAG: RAG Alignment via Multi-Perspective Preference Optimization

# Dataset
## Training data
We provide the training data for PA-RAG, available at  
https://drive.google.com/file/d/1agP7fi1iX-3qFK7XFBvRu6rC5X_-M8Iy/view?usp=drive_link

Include
 * `sft_data.json`: 58.9k instruction fine-tuning data
 * `dpo_data_ri.json`: 11.8k response informative preference data
 * `dpo_data_rr.json`: 13.4k response robustness preference data
 * `dpo_data_cq.json`: 22.5k citation quality preference data

The question used for constracting our training data are sourced from  ASQA, WebQuestions, and Natural Questions. Detailed statistics are as follows:

||IFT|RI|RR|CQ|
|:--|:--:|:--:|:--:|:--:|
|ASQA|1,714|1,046|962|631|
|WebQ|1,681|326|357|653|
|NQ|55,463|10,416|12,080|21,241|
|Sum|58,858|11,788|13,399|22,525|

## Evaluation data
The data for evaluation is available at  
https://drive.google.com/file/d/1vn5O_PtUnV3rOC7CAbSsZITG6NQ1EZtx/view?usp=drive_link.  
The qustions are sourced from the test split from ASQA, WebQustions, Natural Questions, and TriviaQA. The retrieved documents are retrieved by dense retriever [GTR](https://huggingface.co/sentence-transformers/gtr-t5-xxl) from Wikipedia dump from December 20, 2018.
