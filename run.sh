# -------- ACTION STATE TRACKING -----------
# >>> Training <<<
python main.py --learning-rate 1e-5 --weight-decay 0 --batch-size 3 --epochs 21 --log-interval 400 \
    --grad-accum-steps 4 --model-type roberta --prefix 0524 --filename final --task ast
# >>> Examine Results <<<
python main.py --do-eval --quantify  --model-type roberta --prefix 0524 --filename final --task ast

# -------- CASCADING DIALOG SUCCESS ------------
# Preprocess utterances if running for the first time, change model-type as needed
# PYTHONPATH=. python utils/embed.py
# >>> Training <<<
python main.py --learning-rate 3e-5 --weight-decay 0 --batch-size 10 --epoch 14 --log-interval 600 \
    --model-type bert --prefix 0524 --filename final --task cds
# >>> Examine Results <<<
python main.py --do-eval --quantify --cascade --model-type bert --prefix 0524 --filename final --task cds

# ---- ABLATIONS ----
# Task Completion with Intent
python main.py --learning-rate 3e-5 --batch-size 10 --epoch 14 --use-intent \
    --model-type bert --prefix 0524 --filename intent_only --task cds 
# Task Completion with Agent Guidelines Only
python main.py --learning-rate 3e-5 --batch-size 10 --epoch 14 --use-kb \
    --model-type bert --prefix 0524 --filename kb_only --task cds 
# Full Task Completion with Intent and Agent Guideline KB
python main.py --learning-rate 3e-5 --batch-size 10 --epoch 14 --use-intent --use-kb \
    --model-type bert --prefix 0524 --filename intent_and_kb --task cds
