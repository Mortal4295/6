#!/bin/bash
# Run one practice trial (0, 1 or 2) manually: start the kit env server, run the
# bot against it, SIGTERM the env for its state export, score with the sealed scorer.
set -u
KIT="$HOME/workspace/makefun/free-model-aggregator-01-kit/free-model-aggregator-01-kit"
BOT="$HOME/workspace/makefun/free-model-aggregator-01/main.ts"
N="$1"
DATA="$KIT/practice/trial-$N/input/env"
REF="$KIT/practice/trial-$N/reference"
STATE="/tmp/agg-trial$N-state"
PORT=$((18080 + N))
rm -rf "$STATE" "/tmp/agg-trial$N-scores"
mkdir -p "$STATE"
MAKEFUN_STATE_DIR="$STATE" MAKEFUN_ENV_DATA_DIR="$DATA" MAKEFUN_ENV_PORT="$PORT" \
  node "$KIT/program/env/server.mjs" > "/tmp/env$N.log" 2>&1 &
ENVPID=$!
for _ in $(seq 1 100); do [ -f "$STATE/ready" ] && break; sleep 0.1; done
if [ ! -f "$STATE/ready" ]; then echo "env did not become ready"; kill $ENVPID 2>/dev/null; cat "/tmp/env$N.log"; exit 1; fi
EPORT=$(python3 -c "import json;print(json.load(open('$STATE/ready'))['port'])")
MAKEFUN_PROMPT_FILE="$KIT/practice/trial-$N/input/prompt.txt" MAKEFUN_ENV_URL="http://127.0.0.1:$EPORT" \
  node "$BOT" > "/tmp/bot$N.log" 2>&1
echo "trial-$N bot exit=$?"
kill -TERM $ENVPID; wait $ENVPID 2>/dev/null
echo "trial-$N env log:"; cat "/tmp/env$N.log"
MAKEFUN_OUTPUT_DIR="$STATE/export" MAKEFUN_REFERENCE_DIR="$REF" MAKEFUN_SCORES_DIR="/tmp/agg-trial$N-scores" \
  node "$KIT/program/scorer/score.mjs"
echo "--- trial-$N state ---"
python3 -c "
import json
s=json.load(open('$STATE/export/state.json'))
print('totals:', s['totals'])
for r in s['requests']:
    bad=[c['name'] for c in r['checks'] if not c['pass']]
    print(r['id'], 'OK' if r['valid'] else 'INVALID', r['provider'], r['model'], 'rating', r['rating_milli'], 'calls', r['upstream_calls'], 'failures', len(r['failures']), 'BAD', bad)
"
