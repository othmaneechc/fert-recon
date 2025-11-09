python -m src.train --config configs/transformer_base.yaml

python -m src.train --config configs/transformer_base.yaml             # wheat
python -m src.train --config configs/lstm_base.yaml                    # wheat LSTM
python -m src.train --config configs/tcn_base.yaml                     # wheat TCN
python -m src.train --config configs/gbdt_base.yaml                    # wheat GBDT
# Tabular baselines
python -m src.train --config configs/elasticnet_log.yaml               # wheat ElasticNet (log-yield)
python -m src.train --config configs/ridge_enhanced.yaml               # wheat Ridge
python -m src.train --config configs/random_forest.yaml                 # wheat Random Forest
python -m src.train --config configs/xgboost.yaml                       # wheat XGBoost
python -m src.train --config configs/gbrt.yaml                          # wheat Gradient Boosting
# Switch crop to maize and repeat (or make maize copies of the YAMLs).

python -m src.train --config configs/transformer_base.yaml 2>&1 | tee /data/oe23/fert-recon/models/exp_logs/xf_wheat_transformer.log

bash scripts/run_sweep.sh configs/tuning/transformer_hparam.yaml
bash scripts/run_sweep.sh configs/tuning/regularization_strategies.yaml
bash scripts/run_sweep.sh configs/tuning/sequence_strategies.yaml

# After selecting tuned hyperparameters, update configs/cv/transformer_temporal_cv.yaml
# to point at the final config and sweep years as test folds.
bash scripts/run_sweep.sh configs/cv/transformer_temporal_cv.yaml


python -m src.recommend \
  --config configs/transformer_base.yaml \
  --ckpt /data/oe23/fert-recon/models/exp_logs/best_wheat_transformer.pt \
  --out_csv /data/oe23/fert-recon/models/recs_wheat_2014.csv
