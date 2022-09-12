#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

set -e
set -u

ROOD_DIR="$(realpath $(dirname "$0"))"
DST_DIR="$ROOD_DIR/pre-trained_language_models"

mkdir -p "$DST_DIR"
cd "$DST_DIR"

echo "ELMO ORIGINAL 5.5B"
if [[ ! -f elmo/original5.5B/vocab-enwiki-news-500000.txt ]]; then
  mkdir -p 'elmo'
  cd elmo
  mkdir -p 'original5.5B'
  cd original5.5B
  wget -c "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5"
  wget -c "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json"
  wget -c "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_softmax_weights.hdf5"
  wget -c "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/tf_checkpoint/vocab-enwiki-news-500000.txt"
  cd ../../
fi

echo "ELMO ORIGINAL"
if [[ ! -f elmo/original/vocab-2016-09-10.txt ]]; then
  mkdir -p 'elmo'
  cd elmo
  mkdir -p 'original'
  cd original
  wget -c "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
  wget -c "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
  wget -c "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_softmax_weights.hdf5"
  wget -c "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/vocab-2016-09-10.txt"
  cd ../../
fi

cd "$ROOD_DIR"
echo 'Building common vocab'
if [ ! -f "$DST_DIR/common_vocab_cased.txt" ]; then
  python lama/vocab_intersection.py
else
  echo 'Already exists. Run to re-build:'
  echo 'python util_KB_completion.py'
fi

