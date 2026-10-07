#!/bin/bash
export CARLA_ROOT=/home/omar-shaat/carla_0.9.15
export LEADERBOARD_ROOT=/home/omar-shaat/thesis_final/leaderboard
export SCENARIO_RUNNER_ROOT=/home/omar-shaat/thesis_final/scenario_runner
export PORT=2000
export TM_PORT=8000
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8,max_split_size_mb:128

echo "Starting CARLA Simulator..."
${CARLA_ROOT}/CarlaUE4.sh -opengl -carla-rpc-port=${PORT} -quality-level=Epic -RenderOffScreen &
CARLA_PID=$!

sleep 15 

export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/":"${SCENARIO_RUNNER_ROOT}":"${LEADERBOARD_ROOT}":${PYTHONPATH}
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
kill $CARLA_PID
