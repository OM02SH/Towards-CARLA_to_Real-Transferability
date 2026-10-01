 #!/bin/bash

export LEADERBOARD_ROOT=/home/omar-shaat/thesis_final/leaderboard

export ROUTES=$LEADERBOARD_ROOT/data/routes_devtest.xml
export ROUTES_SUBSET=0
export REPETITIONS=1

export DEBUG_CHALLENGE=1
export CHALLENGE_TRACK_CODENAME=SENSORS
export CHECKPOINT_ENDPOINT="${LEADERBOARD_ROOT}/results.json"
export RECORD_PATH=
export RESUME=

export TEAM_AGENT=tfv6_regnet
export TEAM_CONFIG=""

python3 ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py \
  --routes=${ROUTES} \
  --routes-subset=${ROUTES_SUBSET} \
  --repetitions=${REPETITIONS} \
  --track=${CHALLENGE_TRACK_CODENAME} \
  --checkpoint=${CHECKPOINT_ENDPOINT} \
  --agent=${TEAM_AGENT} \
  --agent-config=${TEAM_CONFIG} \
  --debug=${DEBUG_CHALLENGE} \
  --record=${RECORD_PATH} \
  --resume=${RESUME}