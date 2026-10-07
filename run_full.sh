export LEADERBOARD_ROOT=/app/leaderboard
export SCENARIO_RUNNER_ROOT=/app/scenario_runner
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8,max_split_size_mb:128
export PORT=2000
export PYTHONPATH="app/PythonAPI/carla/":"${SCENARIO_RUNNER_ROOT}":"${LEADERBOARD_ROOT}":${PYTHONPATH}
export IS_BENCH2DRIVE=True 
export CLEAR_LANE=1
export ROUTES=$LEADERBOARD_ROOT/data/bench2drive220.xml
export ROUTES_SUBSET=""         
export REPETITIONS=1
export DEBUG_CHALLENGE=0        
export CHALLENGE_TRACK_CODENAME=SENSORS
export CHECKPOINT_ENDPOINT="${LEADERBOARD_ROOT}/results.json"
export RECORD_PATH=
export RESUME=1                 
export TEAM_AGENT=tfv6_regnet
export TEAM_CONFIG=""

cd ${LEADERBOARD_ROOT}
ln -sfn ${SCENARIO_RUNNER_ROOT}/srunner ${LEADERBOARD_ROOT}/srunner

echo "Starting Leaderboard Evaluator..."
python3 -m leaderboard.leaderboard_evaluator \
  --routes=${ROUTES} \
  --routes-subset=${ROUTES_SUBSET} \
  --repetitions=${REPETITIONS} \
  --track=${CHALLENGE_TRACK_CODENAME} \
  --checkpoint=${CHECKPOINT_ENDPOINT} \
  --agent=${TEAM_AGENT} \
  --agent-config=${TEAM_CONFIG} \
  --debug=${DEBUG_CHALLENGE} \
  --record=${RECORD_PATH} \
  --resume=${RESUME} \
  --port=${PORT}

echo "Evaluation sequence ended. Stopping CARLA..."
