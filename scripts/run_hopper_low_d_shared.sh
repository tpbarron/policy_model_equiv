LOG_DIR="/media/trevor/22c63957-b0cc-45b6-9d8f-173d9619fb73/outputs/pol_model/11_12_17_fix/";
DEFAULT_ARGS="--max-episode-steps 1000 --num-processes 1 --num-steps 2048 --entropy-coef 0 --ppo-epoch 10 --lr 3e-4 --gamma 0.99 --tau 0.95 --batch-size 64 --num-frames 1000000 --use-gae --shared-actor-critic"

for i in 1 2 3 4 5 6 7 8 9 10
do
  EXP_PATH="${LOG_DIR}/ppo_baseline/hopper_t1000_shared/${i}/";
  mkdir -p $EXP_PATH
  python main.py ${DEFAULT_ARGS} --seed ${i} --log-dir ${EXP_PATH} --env-name HopperBulletEnv-v0 &

  EXP_PATH="${LOG_DIR}/ppo_model_baseline/hopper_model_t1000_shared/${i}/";
  mkdir -p $EXP_PATH
  python main.py ${DEFAULT_ARGS} --model --seed ${i} --log-dir ${EXP_PATH} --env-name HopperBulletEnv-v0 &

  wait;
done
