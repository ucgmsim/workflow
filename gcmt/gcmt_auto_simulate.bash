#!/usr/bin/env bash

NEW_GCMT="/mnt/hypo_scratch/jfa92/automated_gcmt/gcmt_solutions.json.new"
OLD_GCMT="/mnt/hypo_scratch/jfa92/automated_gcmt/gcmt_solutions.json"
GCMT_DB_KEY=$(cat /mnt/hypo_scratch/jfa92/automated_gcmt/gcmt_db_key)

wget -O $NEW_GCMT "https://gcmt-realtime-database-default-rtdb.asia-southeast1.firebasedatabase.app/$GCMT_DB_KEY/earthquakes.json"

if [[ ! -f $OLD_GCMT ]]; then
    cp $NEW_GCMT $OLD_GCMT
fi

MOST_RECENT_OLD_GCMT=$(jq 'map(.date) | max' $OLD_GCMT)
source $HOME/.pyenv/versions/workflow/bin/activate
jq -r --arg most_recent "$MOST_RECENT_OLD_GCMT" '.[] | select(.date > $most_recent) | .eventID' $NEW_GCMT | while IFS= read -r gcmt_id; do
    echo "Will simulate GCMT event id: $gcmt_id"
    mkdir -p ~/cylc-src/"gcmt_$gcmt_id"/input/$gcmt_id
    plan-workflow "$gcmt_id" "$HOME/cylc-src/gcmt_$gcmt_id/flow.cylc" --goal im_calc --source gcmt --defaults-version 24.2.2.2 --target-host hypocentre
    echo "(bash -c 'source $HOME/.pyenv/versions/workflow/bin/activate && cylc vip gcmt_$gcmt_id') >> /mnt/hypo_scratch/jfa92/automated_gcmt/log 2>&1" | batch
done
mv $NEW_GCMT $OLD_GCMT
