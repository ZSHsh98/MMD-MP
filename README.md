# Detecting Machine-Generated Texts by Multi-Population Aware Optimization for Maximum Mean Discrepancy


Official PyTorch implementation of the ICLR 2024 paper:

<!-- **Detecting Machine-Generated Texts by Multi-Population Aware Optimization for Maximum Mean Discrepancy** -->
**[Detecting Machine-Generated Texts by Multi-Population Aware Optimization for Maximum Mean Discrepancy](https://openreview.net/forum?id=3fEKavFsnv)**

Shuhai Zhang, Yiliao Song, Jiahao Yang, Yuanqing Li, Bo Han, Mingkui Tan.

Abstract: *Large language models (LLMs) such as ChatGPT have exhibited remarkable performance in generating human-like texts. However, machine-generated texts (MGTs) may carry critical risks, such as plagiarism issues, misleading informa- tion, or hallucination issues. Therefore, it is very urgent and important to detect MGTs in many situations. Unfortunately, it is challenging to distinguish MGTs and human-written texts because the distributional discrepancy between them is often very subtle due to the remarkable performance of LLMs. In this paper, we seek to exploit maximum mean discrepancy (MMD) to address this issue in the sense that MMD can well identify distributional discrepancies. However, directly training a detector with MMD using diverse MGTs will incur a significantly in- creased variance of MMD since MGTs may contain multiple text populations due to various LLMs. This will severely impair MMD’s ability to measure the differ-ence between two samples. To tackle this, we propose a novel multi-population aware optimization method for MMD called MMD-MP, which can avoid variance increases and thus improve the stability to measure the distributional discrep- ancy. Relying on MMD-MP, we develop two methods for paragraph-based and sentence-based detection, respectively. Extensive experiments on various LLMs, e.g., GPT2 and ChatGPT, show superior detection performance of our MMD-MP.*

## Requirements

- An NVIDIA RTX graphics card with 12 GB of memory.
- Python 3.7
- Pytorch 1.13.1

## Data and pre-trained models

For dataset, we use HC3, which can be downloaded by [download link](https://huggingface.co/datasets/Hello-SimpleAI/HC3). 
For the pre-trained language models, you need to first download them from the following links:
<!-- For data, here we did not put in all the baselines and data sets, only HC3 was selected.Use the fromPretrain method of transformers to load the HC3 data set. -->



- gpt2: : [download link](https://drive.google.com/file/d/16_-Ahc6ImZV5ClUc0vM5Iivf8OJ1VSif/view?usp=sharing)
- gpt2-large:  [download link](https://huggingface.co/openai-community/gpt2-large/tree/main)
- t5-large:  [download link](https://huggingface.co/t5-large)
- t5-small:  [download link](https://huggingface.co/t5-small)
- roberta-base-openai-detector: : [download link](https://huggingface.co/roberta-base-openai-detector/tree/main)
- minhtoan/gpt3-small-finetune-cnndaily-news: [download link](https://huggingface.co/minhtoan/gpt3-small-finetune-cnndaily-news/tree/main)
- EleutherAI/gpt-neo-125m: [download link](https://huggingface.co/EleutherAI/gpt-neo-125m/tree/main)


After the download, please complete the model_path_dit in the run file.

## Environment of MMD-MP
You have to create a virtual environment and set up libraries needed for training and evaluation.
```
conda env create -f detectGPT.yml
```

## Run experiments on HC3

**Training MMD-MP.**

- Select the best model through best_power:

<!-- # generate nature samples -->
```
CUDA_VISIBLE_DEVICES=0 \
python run_meta_mmd_trans.py \ 
--id 10001 \ 
--sigma0 55 \ 
--lr 0.00005 \ 
--no_meta_flag \   
--n_samples 3900 \ 
--target_senten_num 3000 \ 
--val_num 50 \ 
--sigma 30 \ 
--max_length  100 \ 
--trial_num 3 \ 
--num_hidden_layers 1 \ 
--target_datasets HC3 \ 
--text_generated_model_name chatGPT \ 
--base_model_name roberta-base-openai-detector \ 
--skip_baselines \ 
--mask_flag \ 
--transformer_flag \ 
--meta_test_flag \ 
--epochs 100 \ 
--two_sample_test \
```

- Select the best model through best_auroc:
```
CUDA_VISIBLE_DEVICES=1 \ 
python run_meta_mmd_trans_auroc.py \ 
--id 10002 \ 
--sigma0 40 \ 
--lr 0.00005 \  
--no_meta_flag \   
--n_samples 3900 \ 
--target_senten_num 3000 \ 
--val_num 50 \ 
--sigma 30 \ 
--max_length 100 \ 
--trial_num 3 \ 
--num_hidden_layers 1 \
--target_datasets HC3 \ 
--text_generated_model_name chatGPT \ 
--base_model_name roberta-base-openai-detector \ 
--skip_baselines \ 
--mask_flag \ 
--transformer_flag \ 
--meta_test_flag \ 
--epochs 100 \ 
--two_sample_test \
```

**Testing MMD-MP.**
- Add a command-line argument **--test_flag** to enable the testing functionality, allowing for the evaluation of the checkpoint corresponding to the specified **id**:

```
CUDA_VISIBLE_DEVICES=0 \ 
python run_meta_mmd_trans.py \ 
--test_flag \
--id 10001 \ 
--sigma0 55 \ 
--lr 0.00005 \  
--no_meta_flag \   
--n_samples 3900 \ 
--target_senten_num 3000 \ 
--val_num 50 \ 
--sigma 30 \ 
--max_length  100 \ 
--trial_num 3 \ 
--num_hidden_layers 1 \ 
--target_datasets HC3 \ 
--text_generated_model_name chatGPT \ 
--base_model_name roberta-base-openai-detector \ 
--skip_baselines \ 
--mask_flag \ 
--transformer_flag \ 
--meta_test_flag \
--epochs 100 \ 
--two_sample_test \
```

<!-- Training process and result records in ./two_sample_test/HC3-roberta-base-openai-detector/ id -->
## Citation


```
@inproceedings{zhangs2024MMDMP,
  title={Detecting Machine-Generated Texts by Multi-Population Aware Optimization for Maximum Mean Discrepancy},
  author={Zhang, Shuhai and Song, Yiliao and Yang, Jiahao and Li, Yuanqing and Han, Bo and Tan, Mingkui},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year={2024}
}
```
