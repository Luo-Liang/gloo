#!/bin/sh
git remote update
UPSTREAM=${1:-'@{u}'}
LOCAL=$(git rev-parse @)
REMOTE=$(git rev-parse "$UPSTREAM")
BASE=$(git merge-base @ "$UPSTREAM")
git reset --hard origin/master
if [ $LOCAL = $REMOTE ]; then
    echo "Up-to-date"
elif [ $LOCAL = $BASE ]; then
    echo "Need to pull"
    git pull
    cd build
    rm * -rf
    cmake .. -DUSE_CUDA=1 -DUSE_REDIS=1 -DBUILD_SHARED_LIBS=ON -DUSE_IBVERBS=0 -DBUILD_BENCHMARK=1
    make -j32
elif [ $REMOTE = $BASE ]; then
    echo "Need to push"
else
    echo "Diverged"
fi
