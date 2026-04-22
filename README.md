# Mutilingual Optimal Transport
This repository implements Optimal Transport Alignment between English (Dominant Languge) and other target languages.

# Repository Structure

```bash
Mutilingual Optimal Transport/
├── dataloader/
│   ├── alignment_dataloader.py
│   ├── downstream_dataloader.py
│   └── finetune_dataloader.py
├── finetune/
│   └── Llama3-8B-Finetuning.py
├── optimal_transport/
│   ├── Llama3-8B-OT_no_L_LM.py <- L_OT loss only (main)
│   └── Llama3-8B-OT.py         <- L_LM + lambda*L_OT
├── optimal_transport_evaluation/
│   ├── MMMLU_evaluation.py
│   ├── XNLI_evaluation.py
│   └── XSQuAD_evaluation.py
├── venv/
├── zero-shot/
├── .env
├── .gitignore
├── download_data.py
├── README.md
└── requirements.txt
```

# Training Guidelines
1. Clone Repo & .env HF key

```bash
git clone https://github.com/TracDucAnh/Multilingual-Optimal-Transport.git

cd Multilingual-Optimal-Transport

echo 'HF_TOKEN = "YOUR HF TOKEN"' > .env
```



2. Install Requirements

```bash
pip install -r requirements.txt
```

3. Download All Required Data

```bash
python download_data.py --root raw_data/
```

4. Run Optimal Transport Version No Casual Language Modeling Loss

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun --nproc_per_node=2 optimal_transport/Llama3-8B-OT-only-distributed.py \
    --gpus 2 \
    --base_model ducanhdinh/Llama3-8B-Finetune \
    --data_root /raw_data/alignment/ \
    --output_dir ./ot_checkpoints_dist \
    --hub_repo Llama3-8B-OT \
    --epochs 3 \
    --batch_size 32 \
    --grad_accum 4 \
    --lr 1e-4 \
    --sinkhorn_eps 0.1 \
    --sinkhorn_iters 50 \
    --opus_ratio 0.02 \
    --eng_eng_ratio 0.3 \
    --mean_layer true \
    --max_length 256 \
    --seq_length 256 \
    --save_iter 200 \
    --num_workers 4 \
    --bf16 \
    --gradient_checkpointing
```

5. To run downstream evaluation

```bash
# Eval XNLI benchmark
python optimal_transport_evaluation/XNLI_evaluation.py --hub_repo ducanhdinh/Llama3-8B-OT_no_L_LM

# Eval MMMLU benchmark
python optimal_transport_evaluation/MMMLU_evaluation.py --hub_repo ducanhdinh/Llama3-8B-OT_no_L_LM

# Eval XSQuAD benchmark
python optimal_transport_evaluation/XSQuAD_evaluation.py --hub_repo ducanhdinh/Llama3-8B-OT_no_L_LM
```