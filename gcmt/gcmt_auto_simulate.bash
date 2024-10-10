#!/usr/bin/env bash

NEW_GCMT="/mnt/hypo_scratch/jfa92/automated_gcmt/gcmt_solutions.csv.new"
OLD_GCMT="/mnt/hypo_scratch/jfa92/automated_gcmt/gcmt_solutions.csv"

wget -O $NEW_GCMT "https://raw.githubusercontent.com/GeoNet/data/main/moment-tensor/GeoNet_CMT_solutions.csv"

if [[ ! -f $OLD_GCMT ]]; then
    cp $NEW_GCMT $OLD_GCMT
fi

MOST_RECENT_OLD_GCMT=$(awk -F',' '$2 > $id && $1 !="PublicID" {id=$2} END{print $2}' $OLD_GCMT)
source $HOME/.pyenv/versions/workflow/bin/activate
while IFS= read -r gcmt_id; do
    echo "Will simulate GCMT event id: $gcmt_id"
    mkdir -p ~/cylc-src/"gcmt_$gcmt_id"/input
    plan-workflow "$gmct_id" "$HOME/cylc-src/gcmt_$gcmt_id/flow.cylc" --goal im_calc --goal plot_ts --source gcmt --defaults-version 24.2.2.4 --target-host hypocentre
    echo "source $HOME/.pyenv/versions/workflow/bin/activate &&bash -c 'cylc vip gcmt_$gcmt_id'" | batch
done < <(awk -F',' "\$2 > $MOST_RECENT_OLD_GCMT &&\$12 >= 4.5 && \$1 !=\"PublicID\" { print \$1 }" $NEW_GCMT)

mv $NEW_GCMT $OLD_GCMT
