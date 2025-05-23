# Hydra: Structured Cross-Source Enhanced Large Language Model Reasoning (EMNLP Submit)
# Code Documentation 

Hydra unifies structured knowledge graphs, Wikipedia documents, and live web search so that large language models can **reason over verified multi‑source evidence**. 

---

## Directory layout

```text
Hydra/
├── answer/                 evaluation helpers and scoring scripts
├── data/                   benchmark datasets (CWQ, AdvHotpotQA, QALD10-en, SimpleQA, WebQSP, Webquestions, Zeroshot RE)
├── docs_pages/             extra documentation (MkDocs site)
├── Freebase/               Freebase environment setting. See Freebase/README.md for details.
├── freebase_subgraph/      Freebase subgraph KG
├── Hydra_run/              main source folder – run code from here
│   ├── hydra_main.py       entry point (invokes `build_parser`)
│   ├── cot_prompt_list.py  chain‑of‑thought prompts
│   ├── emb_model/          embedding model wrappers
│   ├── freebase_func.py    Freebase SPARQL helpers
│   ├── wiki_client.py      WikiKG client helpers
│   ├── subgraph_helper.py  subgraph construction utilities
│   ├── subgraph_utilts.py  extra graph helpers
│   ├── detected_kgsub.py   KG subgraph detection logic
│   ├── resp_process.py     response post‑processing
│   ├── utilts.py           shared utilities
│   └── utilts2.py          extra utilities
├── online_search/          live web search and documents caching
├── wiki_subgraph/          Wiki subgraph KG
├── Wikidata/               Wikidata environment setting. See Wikidata/README.md for details.
├── requirements.txt        Python dependencies
└── README.md               this file
```

---


## Get started
Before running Hydra, please ensure you have successfully installed **Freebase**, and **Wikidata** on your local machine. The comprehensive installation instructions and necessary configuration details can be found in the `/Freebase/README.md` and `/Wikidata/README.md`.

Once, Wikidata setup, copy the `server_urls.txt` files from the Wikidata directory into the Hydra_run folder.

You must use your own API in the `run_LLM` function of `utilts.py` for the APIs, and your own SerpAPI in `utilts2.py` for online search.

To set up the environment, install the required dependencies using:

```bash
cd Hydra
```

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```





---

## Command‑line interface

`hydra_main.py` exposes the options defined in `build_parser`.

```bash
cd Hydra/Hydra_run

python hydra_main.py \
  webqsp \                 # positional: dataset name prefix
  --depth 3 \              # maximum hop depth
  --allr \                 # Using Hydra instead of Hydra-E
  --model llama70b \       # LLM backend

```

### Positional argument

| Argument    | Meaning                                                                                                                             |
| ----------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| `file_name` | Dataset prefix. Supported values include `CWQ`, `hotpot` (AdvHotpotQA), `qald`, `simpleqa`, `webqsp`, `webquestions`, `zeroshotre`. |

### Main options

| Flag                                          | Description                                                                                  | Default    |
| --------------------------------------------- | -------------------------------------------------------------------------------------------- | ---------- |
| `--allsource`                                 | Enable all sources in every expansion step                                                   | off        |
| `--allr`                                      | (Hydra) rather than a single relation (Hydra‑E)                 | off        |
| `--incomplete`                                | Work with an incomplete KG split                                                             | off        |
| `--ratio {100,80,50,30}`                      | When `--incomplete` is on, sample the KG at the given percentage                             | `100`      |
| `--no-summary`                                | Disable summary generation                                                                   | on         |
| `--no-freebase`                               | Disable Freebase KG                                                                          | on         |
| `--no-wikikg`                                 | Disable WikiKG                                                                               | on         |
| `--no-web`                                    | Disable web search                                                                           | on         |
| `--no-wikidocu`                               | Disable retrieval from Wikipedia documents                                                   | on         |
| `--model {gpt3,gpt4,llama,deepseek,llama70b}` | LLM backend.<br>`llama` = Llama‑3.1‑8B, `deepseek` = DeepSeek‑v3, `llama70b` = Llama‑3.1‑70B | `llama70b` |
| `--depth {1,2,3,4}`                           | Maximum search depth                                                                         | `3`        |

> Modules remain active unless explicitly disabled with a `--no‑*` flag.

---

## Examples

Run Hydra with all sources and a four‑hop search on CWQ:

```bash
python hydra_main.py CWQ --allsource --depth 4 --model gpt3
```

Run an ablation without web evidence and using an incomplete KG sampled at 50 percent:

```bash
python hydra_main.py webqsp \
  --no-web \
  --incomplete --ratio 50 \
  --depth 3
```

Outputs are stored under `/Hydra/answer/` and include logs, intermediate paths, and final predictions.

---

## Evaluation

Accuracy is computed with the answer in `answer/`. The positional argument is same with hydra_main.py

```bash
python check_answer.py \
  webqsp \                 # positional: dataset name prefix
  --depth 3 \              # maximum hop depth
  --allr \                 # Using Hydra instead of Hydra-E
  --model llama70b \       # LLM backend
```


## Notes
- Ensure that the dataset files and model configurations are correctly set up before running the scripts.
- Use appropriate depth values based on the complexity of the dataset and required accuracy.

---


## Claims
This project uses the Apache 2.0 protocol. The project assumes no legal responsibility for any of the model's output and will not be held liable for any damages that may result from the use of the resources and output.