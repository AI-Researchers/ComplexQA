# Think-On-Graph

This repository includes the code for paper "Think-on-Graph: Deep and Responsible Reasoning of Large Language Model on Knowledge Graph" The link to the original repo is [here](https://github.com/IDEA-FinAI/ToG).

## Project structure
* `CoT/` -- This folder contains experiments that correspond to the CoT and IO prompt. See `CoT/README.md` for details.
* `data/` -- This folder contains all datasets used in the paper. See `data/README.md` for details.
* `eval/` -- This folder contains evaluation scripts. See `eval/README.md` for details.
* `Freebase/` -- Freebase environment setting as mentioned in Think on Graph repo. See `Freebase/README.md` for details.
* `Wikidata/` -- Wikidata environment setting as mentioned in Think on Graph repo. See `Wikidata/README.md` for details.
* `ToG/` -- This folder contains source codes for Think on Graph approach. See `ToG/README.md` for details.
    - `client.py`: Pre-defined Wikidata APIs, copy from `Wikidata/`.
    - `server_urls.txt`: Wikidata server urls, copy from `Wikidata/`.
    - `main_freebase.py`: The main file of ToG where Freebase as KG source. See `ToG/README.md` for details.
    - `main_wiki.py`: Same as above but using Wikidata as KG source. See `ToG/README.md` for details.
    - `prompt_list.py`: The prompts for the ToG to pruning, reasoning and generating.
    - `freebase_func.py`: All the functions used in `main_freebase.py`.
    - `wiki_func.py`: All the functions used in `main_wiki.py`.
    - `utils.py`: All the functions used in ToG.
* `virtuoso/` -- This folder contains the code for managing virtoso service.
* `requirements.txt` -- Pip environment file.

## Requirement installation
```
conda create --name tog_env python=3.9
conda activate tog_env
pip install -r requirements.txt
```

## Get started
Before running ToG, please ensure that you have successfully installed either **Freebase** or **Wikidata** on your local machine. 

### Setting up Freebase + Virtuoso
To set up the SPARQL endpoint to Freebase via Virtuoso we use another code repository [link](https://github.com/dki-lab/Freebase-Setup). 

#### Requirements
* OpenLink Virtuoso 7.2.5 (download from this [link](https://sourceforge.net/projects/virtuoso/files/virtuoso/7.2.5/virtuoso-opensource.x86_64-generic_glibc25-linux-gnu.tar.gz/download))
* Python 3 (required if using the provided Python script)

#### Freebase data dump
Download the processed Virtuoso DB file via wget (WARNING: 53G+ disk space is needed):
```
wget https://www.dropbox.com/s/q38g0fwx1a3lz8q/virtuoso_db.zip
```

#### Managing the Virtuoso service
Unzip the OpenLink Virtuoso 7.2.5 software downloaded from the link mentioned above into the virtuoso folder `virtuoso/`. The wrapper script (adapted from [Sempre](https://github.com/percyliang/sempre)) `virtuoso.py` is used for managing the Virtuoso service. To use it, first change the virtuosoPath in the script to your local Virtuoso directory. Assuming the Virtuoso db file is located in a directory named `virtuoso_db` under the same directory as the script `virtuoso.py` and 3001 is the intended HTTP port for the service, to start the Virtuoso service:
```
python3 virtuoso.py start 3001 -d virtuoso_db
```
To stop the running Virtuoso service on the same port:
```
python3 virtuoso.py stop 3001
```
A server with at least 100 GB RAM is recommended. You may adjust the maximum amount of RAM the service may use and other configurations via the provided script.

## Running the experiments 
To reproduce results for Llama3 model, you just need to run below command on the terminal first:
```
python -m vllm.entrypoints.openai.api_server --model TechxGenus/Meta-Llama-3-70B-Instruct-GPTQ --dtype float16 --api-key token-abc123
```
To run Think on Graph using Freebase KG, navigate to the `ToG/` folder and then run below command for Meta-Llama-3-70B-Instruct-GPTQ model:
```
python main_freebase.py --dataset webqsp --max_length 256 --temperature_exploration 0.4 --temperature_reasoning 0.1 --width 3 --depth 3 --remove_unnecessary_rel True --LLM_type llama3 --num_retain_entity 5 --prune_tools llm
```
For Mixtral-8x7B-Instruct-v0.1 model, run this on a seperate terminal:
```
python -m vllm.entrypoints.openai.api_server --model TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ --dtype float16 --api-key token-abc123
```
Navigate to the `ToG/` folder and then run below command for Think on Graph with Mixtral model:
```
python main_freebase.py --dataset webqsp --max_length 256 --temperature_exploration 0.4 --temperature_reasoning 0.1 --width 3 --depth 3 --remove_unnecessary_rel True --LLM_type mixtral --num_retain_entity 5 --prune_tools llm
```
For Llama-2-70B-Chat-GPTQ model, run this on a seperate terminal:
```
python -m vllm.entrypoints.openai.api_server --model TheBloke/Llama-2-70B-Chat-GPTQ --dtype float16 --api-key token-abc123
```
For running the code, navigate to `ToG/` folder and then run below command:
```
python main_freebase.py --dataset webqsp --max_length 256 --temperature_exploration 0.4 --temperature_reasoning 0.1 --width 3 --depth 3 --remove_unnecessary_rel True --LLM_type llama2 --num_retain_entity 5 --prune_tools llm
```

See `ToG/README.md` for more details. 

## Experiment results

| **Model**                                     | **WebQSP** |
|-----------------------------------------------|------------|
| ToG - ChatGPT                                 | _76.2_     |
| ToG - GPT4                                    | _82.6_     |
| ToG - Llama2-70B-Chat                         | _63.7_     |
| **Ours** - Llama3-70B-Instruct-GPTQ           | 64.2       |
| **Ours** - Llama 2-70B-Chat-GPTQ              | 49.3       |
| **Ours** - Mixtral-8x7B-Instruct-v0.1-GPTQ    | 48.8       |