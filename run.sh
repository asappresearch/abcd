# -------- ACTION STATE TRACKING -----------
# >>> Training <<<
python main.py --learning-rate 1e-5 --weight-decay 0 --batch-size 3 --epochs 21 --eval-interval 800 \
    --grad-accum-steps 4 --hidden-dim 1024 --model-type roberta --prefix 0524 --filename final --task ast
# >>> Examine Results <<<
python main.py --do-eval --quantify \
	--hidden-dim 1024 --model-type roberta --prefix 0524 --filename final --task ast

# -------- CASCADING DIALOG SUCCESS ------------
# >>> Training <<<
python main.py --learning-rate 3e-5 --weight-decay 0 --batch-size 10 --epoch 14 --eval-interval 500 \
    --hidden-dim 768 --model-type bert --prefix 0524 --filename final --task cds
# >>> Examine Results <<<
python main.py --do-eval --quantify --cascade \
    --hidden-dim 768 --model-type bert --prefix 0524 --filename final --task cds

# ---- ABLATIONS ----
# Task Completion with Intent
python main.py --learning-rate 3e-5 --batch-size 10 --epoch 14 --eval-interval 500 --use-intent \
    --hidden-dim 768 --model-type bert --prefix 0524 --filename intent_only --task cds 
# Task Completion with Agent Guidelines Only
python main.py --learning-rate 3e-5 --batch-size 10 --epoch 14 --eval-interval 500 --use-kb \
    --hidden-dim 768 --model-type bert --prefix 0524 --filename kb_only --task cds 
# Full Task Completion with Intent and Agent Guideline KB
python main.py --learning-rate 3e-5 --batch-size 10 --epoch 14 --eval-interval 500 --use-intent --use-kb \
    --hidden-dim 768 --model-type bert --prefix 0524 --filename intent_and_kb --task cds